"""
Heterogeneous Graph Neural Network (HGNN) Teacher Model.

Owner: Member 1 (Data & Graph Learning)

Paper reference: Section 2, Equation 1
- Uses SAGEConv from PyG
- Architecture: [512, 256, 128, 64] with ReLU
- Separate GNN per edge type, mean aggregation across relations
- Trained with contrastive loss (Eq. 2-3)
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)


class HGNNLayer(nn.Module):
    """
    A single HGNN layer with separate SAGEConv per relation type.

    For each relation type, applies SAGEConv independently, then
    aggregates across relation types (mean by default).
    """

    def __init__(self, in_channels: int, out_channels: int, relation_types: List[str], aggr: str = "mean"):
        """
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            relation_types: List of edge type names (e.g., ['copurchased'] or ['light', 'medium', 'heavy']).
            aggr: Aggregation across relation types: 'mean' or 'sum'.
        """
        super().__init__()
        self.relation_types = relation_types
        self.aggr = aggr

        # One SAGEConv per relation type
        self.convs = nn.ModuleDict({
            rel: SAGEConv(in_channels, out_channels) for rel in relation_types
        })

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels].
            edge_indices: Dict mapping relation_type -> edge_index tensor [2, E].

        Returns:
            Updated node features [N, out_channels].
        """
        outputs = []

        for rel in self.relation_types:
            if rel in edge_indices and edge_indices[rel].size(1) > 0:
                out = self.convs[rel](x, edge_indices[rel])
                outputs.append(out)

        if not outputs:
            # No edges for any relation: return projected features
            return self.convs[self.relation_types[0]](x, torch.zeros(2, 0, dtype=torch.long, device=x.device))

        # Aggregate across relation types
        stacked = torch.stack(outputs, dim=0)
        if self.aggr == "mean":
            return stacked.mean(dim=0)
        elif self.aggr == "sum":
            return stacked.sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggr}")


class HGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network — Teacher Model.

    Architecture: Multi-layer SAGEConv with ReLU activation.
    Projects CNN embeddings (512-dim) to structural embeddings (64-dim)
    through the graph structure.

    Paper: "The HGNN produces a structural embedding h_s_p = HGNN(h_CNN_p, E+)"
    """

    def __init__(
        self,
        layer_dims: List[int] = None,
        relation_types: List[str] = None,
        relation_aggr: str = "mean",
        activation: str = "relu",
    ):
        """
        Args:
            layer_dims: List of layer dimensions, e.g., [512, 256, 128, 64].
                        First element is the input dim, last is output dim.
            relation_types: List of edge type names.
            relation_aggr: Aggregation across relation types.
            activation: Activation function name.
        """
        super().__init__()

        if layer_dims is None:
            layer_dims = [512, 256, 128, 64]
        if relation_types is None:
            relation_types = ["copurchased"]

        self.layer_dims = layer_dims
        self.relation_types = relation_types
        self.output_dim = layer_dims[-1]

        # Build HGNN layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                HGNNLayer(layer_dims[i], layer_dims[i + 1], relation_types, aggr=relation_aggr)
            )

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through all HGNN layers.

        Args:
            x: Node features [N, input_dim].
            edge_indices: Dict mapping relation_type -> edge_index [2, E].

        Returns:
            Structural embeddings [N, output_dim].
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_indices)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = self.activation(x)

        return x

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_indices: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get structural embeddings (same as forward, explicit naming).

        Args:
            x: Node features [N, 512].
            edge_indices: Dict of relation -> edge_index.

        Returns:
            Structural embeddings [N, 64].
        """
        return self.forward(x, edge_indices)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for HGNN training (Paper Equations 2-3).

    L_ci = (1/|E+|) * sum_{(p,q) in E+} a_{p,q} * ||h_p - h_q||^2
         - (1/|E-|) * sum_{(p,q) in E-} ||h_p - h_q||^2

    L_total = gamma_1 * L_c1 + gamma_2 * L_c2 + ...

    For H&M (single relation): L_total = gamma_1 * L_copurchased
    """

    def __init__(self, gamma: List[float] = None, margin: float = 1.0):
        """
        Args:
            gamma: Loss weights per relation type. For H&M: [1.0].
            margin: Contrastive margin.
        """
        super().__init__()
        self.gamma = gamma or [1.0]
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_edges: Dict[str, torch.Tensor],
        positive_weights: Dict[str, torch.Tensor],
        negative_edges: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Node embeddings [N, D].
            positive_edges: Dict of rel -> edge_index [2, E+].
            positive_weights: Dict of rel -> edge_weights [E+].
            negative_edges: Dict of rel -> edge_index [2, E-] (|E-| = |E+|).

        Returns:
            Scalar loss.
        """
        total_loss = torch.tensor(0.0, device=embeddings.device)

        for i, rel in enumerate(positive_edges.keys()):
            gamma_i = self.gamma[i] if i < len(self.gamma) else self.gamma[-1]

            # Positive loss: weighted distance for connected pairs
            pos_edge = positive_edges[rel]
            pos_weight = positive_weights[rel]

            h_p = embeddings[pos_edge[0]]
            h_q = embeddings[pos_edge[1]]
            pos_dist = (h_p - h_q).pow(2).sum(dim=-1)
            pos_loss = (pos_weight * pos_dist).mean()

            # Negative loss: distance for random pairs (should be large)
            neg_edge = negative_edges[rel]
            h_p_neg = embeddings[neg_edge[0]]
            h_q_neg = embeddings[neg_edge[1]]
            neg_dist = (h_p_neg - h_q_neg).pow(2).sum(dim=-1)
            neg_loss = torch.clamp(self.margin - neg_dist, min=0).mean()

            total_loss = total_loss + gamma_i * (pos_loss + neg_loss)

        return total_loss

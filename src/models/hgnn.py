import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)

class HGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network (HGNN) — Teacher Model.
    Enforces Layer-Wise Aggregation as per the paper. No Late-Fusion.
    """
    def __init__(
        self,
        layer_dims: List[int] = None,
        relation_types: List[str] = None,
        neighbor_aggr: str = "mean",
        relation_aggr: str = "mean",
    ):
        super().__init__()
        self.layer_dims = layer_dims or [512, 256, 128, 64]
        self.relation_types = relation_types or ["click", "favorite", "cart", "purchase"]
        self.relation_aggr = relation_aggr
        self.activation = nn.ReLU()

        # Build layers sequentially. Each layer contains a module dictionary for relations.
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_dims) - 1):
            conv_dict = nn.ModuleDict()
            for rel in self.relation_types:
                conv_dict[rel] = SAGEConv(self.layer_dims[i], self.layer_dims[i + 1], aggr=neighbor_aggr)
            self.layers.append(conv_dict)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        
        h = x
        
        for i, layer_dict in enumerate(self.layers):
            relation_outputs = []
            
            for rel in self.relation_types:
                if rel in edge_index_dict and edge_index_dict[rel].size(1) > 0:
                    edge_index = edge_index_dict[rel]
                    h_rel = layer_dict[rel](h, edge_index)
                    relation_outputs.append(h_rel)
            
            if not relation_outputs:
                # Fallback if a subgraph lacks any edges in the current batch
                relation_outputs.append(torch.zeros((h.size(0), self.layer_dims[i + 1]), device=h.device))

            # Aggregate across all relation-specific representations
            stacked = torch.stack(relation_outputs, dim=0)
            if self.relation_aggr == "sum":
                h = stacked.sum(dim=0)
            else:
                h = stacked.mean(dim=0)
            
            if i < len(self.layers) - 1:
                h = self.activation(h)

        # L2 normalization constraints Euclidean distance to prevent negative infinity in ContrastiveLoss
        return F.normalize(h, p=2, dim=-1)


class ContrastiveLoss(nn.Module):
    """
    Modified implementation of Equation (2) and (3) for HGNN pre-training.
    Introduces a margin to the negative term to prevent unbounded negative loss.
    L_ci = (1/|E+|) * Σ (a_{p,q} * ||h_p − h_q||²) - (1/|E-|) * Σ max(0, margin - ||h_p − h_q||²)
    """
    def __init__(
        self, 
        gamma_dict: Optional[Dict[str, float]] = None,
        margin: float = 2.0
    ):
        super().__init__()
        self.gamma_dict = gamma_dict or {
            "click":    1.0,
            "favorite": 0.5,
            "cart":     0.5,
            "purchase": 0.1,
        }
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_edges: Dict[str, torch.Tensor],
        positive_weights: Dict[str, torch.Tensor],
        negative_edges: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        
        total_loss = torch.tensor(0.0, device=embeddings.device)

        for rel, pos_edge in positive_edges.items():
            if pos_edge.size(1) == 0:
                continue
                
            gamma_i = self.gamma_dict.get(rel, 1.0)

            # --- Positive Term (Weighted by a_{p,q}) ---
            pos_w = positive_weights[rel]
            
            pos_w = pos_w / (pos_w.mean() + 1e-8)

            h_pos_src = embeddings[pos_edge[0]]
            h_pos_dst = embeddings[pos_edge[1]]
            
            pos_sq_dist = ((h_pos_src - h_pos_dst) ** 2).sum(dim=-1)
            pos_loss = (pos_w * pos_sq_dist).mean() 

            # --- Negative Term (Unweighted, WITH Margin) ---
            neg_edge = negative_edges.get(rel)
            if neg_edge is not None and neg_edge.size(1) > 0:
                h_neg_src = embeddings[neg_edge[0]]
                h_neg_dst = embeddings[neg_edge[1]]
                
                neg_sq_dist = ((h_neg_src - h_neg_dst) ** 2).sum(dim=-1)
                neg_loss = F.relu(self.margin - neg_sq_dist).mean() 
            else:
                neg_loss = torch.tensor(0.0, device=embeddings.device)

            # Tổng Loss = Khoảng cách Pos + Hình phạt khi Neg ở quá gần
            relation_loss = pos_loss + neg_loss
            
            # L_tot = sum(gamma * L_ci)
            total_loss = total_loss + (gamma_i * relation_loss)

        return total_loss
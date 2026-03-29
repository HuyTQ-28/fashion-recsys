"""
Student MLP — Attribute Encoder.

Owner: Member 1 (Data & Graph Learning)

Paper reference: Section 2, Equation 4
- Simple MLP: [512, 256, 128, 64] with ReLU
- Input: FashionCLIP embedding (512-dim)
- Output: 64-dim embedding aligned to HGNN structural space
- Trained via knowledge distillation (alignment loss: MSE with frozen HGNN)

Also serves as the template for Personal MLP (Member 2).
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StudentMLP(nn.Module):
    """
    Student MLP (Attribute Encoder).

    Learns to map CNN embeddings into the HGNN embedding space
    without access to graph structure. Performs hybrid filtering
    by integrating content-based (CNN) with collaborative (HGNN) signals.

    Paper: "MLP(h_CNN_p) -> embedding aligned with HGNN(h_CNN_p, E+)"
    """

    def __init__(self, layer_dims: List[int] = None, activation: str = "relu"):
        """
        Args:
            layer_dims: List of layer dimensions, e.g., [512, 256, 128, 64].
            activation: Activation function name.
        """
        super().__init__()

        if layer_dims is None:
            layer_dims = [512, 256, 128, 64]

        self.layer_dims = layer_dims
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]

        # Build MLP layers
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No activation on last layer
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: CNN/CLIP embeddings [batch_size, input_dim].

        Returns:
            Projected embeddings [batch_size, output_dim].
        """
        return self.network(x)

    def project_all(self, embeddings_dict: dict) -> dict:
        """
        Project all article embeddings through the MLP.

        Args:
            embeddings_dict: Dict of article_id -> CLIP embedding tensor [512].

        Returns:
            Dict of article_id -> MLP embedding tensor [64].
        """
        self.eval()
        projected = {}

        with torch.no_grad():
            article_ids = list(embeddings_dict.keys())
            embeddings = torch.stack([embeddings_dict[aid] for aid in article_ids])

            # Process in batches to avoid OOM
            batch_size = 1024
            for i in range(0, len(article_ids), batch_size):
                batch_ids = article_ids[i : i + batch_size]
                batch_emb = embeddings[i : i + batch_size].to(
                    next(self.parameters()).device
                )
                batch_proj = self.forward(batch_emb).cpu()

                for aid, proj in zip(batch_ids, batch_proj):
                    projected[aid] = proj

        return projected


class AlignmentLoss(nn.Module):
    """
    Alignment loss for Student MLP distillation (Paper Equation 4).

    L_al = (1/|V|) * sum_{p in V} ||MLP(h_CNN_p) - HGNN(h_CNN_p, E+)||^2

    HGNN is frozen; only MLP weights are updated.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alignment loss.

        Args:
            student_embeddings: MLP output [N, D].
            teacher_embeddings: HGNN output [N, D] (detached/frozen).

        Returns:
            Scalar MSE loss.
        """
        return self.mse(student_embeddings, teacher_embeddings.detach())

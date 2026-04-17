import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StudentMLP(nn.Module):
    """
    Student MLP (Attribute Encoder).

    Projects raw CLIP embeddings into the HGNN embedding space
    without needing graph edges at inference time.
    """

    def __init__(self, layer_dims: List[int] = None):
        """
        Args:
            layer_dims: Layer dimensions [in, h1, ..., out].
                        Default [512, 256, 128, 64].
        """
        super().__init__()

        self.layer_dims = layer_dims or [512, 256, 128, 64]

        layers = []
        for i in range(len(self.layer_dims) - 1):
            layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            # ReLU after every layer EXCEPT the last
            if i < len(self.layer_dims) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return F.normalize(out, p=2, dim=-1)


class AlignmentLoss(nn.Module):
    """
    Alignment Loss for Student MLP distillation.

    L_al = (1/|V|) Σ_p ||MLP(h_p^CNN) - HGNN(h_p^CNN, E+)||_2^2
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mlp_output: torch.Tensor,
        hgnn_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mlp_output:  Student MLP predictions [N, out_dim].
            hgnn_target: Frozen HGNN embeddings   [N, out_dim].

        Returns:
            Scalar loss matching Equation 4 exactly.
        """
        sq_l2_norm = ((mlp_output - hgnn_target) ** 2).sum(dim=-1)
        
        return sq_l2_norm.mean()

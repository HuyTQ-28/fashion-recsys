"""
Triplet Loss Continual Personalization.

Owner: Member 2 (Personalization Engine)

Paper reference: Section 2, Equations 6-7
- Collects batch of recent user interactions
- Computes weighted centroid of interacted article embeddings
- Selects negatives (random for H&M since no click data)
- Runs SGD steps on the user's Personal MLP with triplet loss:
  L_tri = max(0, ||MLP_u(h_wc) - MLP_u(h_pos)||^2 - ||MLP_u(h_wc) - MLP_u(h_neg)||^2 + epsilon)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.personal_mlp import PersonalMLP

logger = logging.getLogger(__name__)


class TripletAdaptation:
    """
    Triplet Loss adaptation for Personal MLP.

    Adapts a user's Personal MLP in-place using recent interactions.
    """

    def __init__(
        self,
        sgd_steps: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        margin: float = 100.0,
    ):
        """
        Args:
            sgd_steps: Number of SGD steps per adaptation.
            learning_rate: Learning rate for adaptation SGD.
            weight_decay: Weight decay for adaptation.
            margin: Triplet loss margin epsilon.
                    Use float('inf') for margin=infinity (removes the margin constraint).
        """
        self.sgd_steps = sgd_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.margin = margin

    def compute_weighted_centroid(
        self,
        clip_embeddings: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Compute weighted centroid of interacted article CLIP embeddings (Eq. 6).

        Args:
            clip_embeddings: List of CLIP embeddings for interacted articles.
            weights: Optional weights (e.g., by interaction type or recency).
                     For H&M (purchase only): uniform weights.

        Returns:
            Weighted centroid tensor [512].
        """
        stacked = torch.stack(clip_embeddings)  # [B, 512]
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float).unsqueeze(1)  # [B, 1]
            return (stacked * w).sum(dim=0) / w.sum()
        return stacked.mean(dim=0)

    def adapt(
        self,
        personal_mlp: PersonalMLP,
        positive_embeddings: List[torch.Tensor],
        negative_embeddings: List[torch.Tensor],
        interaction_weights: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Run triplet loss adaptation on a Personal MLP (Equation 7).

        Modifies the MLP weights IN-PLACE.

        Args:
            personal_mlp: User's Personal MLP (modified in-place).
            positive_embeddings: CLIP embeddings of positively interacted articles.
            negative_embeddings: CLIP embeddings of negative articles.
            interaction_weights: Optional weights for positive articles.

        Returns:
            Tuple of (final_loss, adaptation_time_ms).
        """
        start_time = time.time()

        personal_mlp.train()

        # Compute weighted centroid
        h_wc = self.compute_weighted_centroid(positive_embeddings, interaction_weights)

        # Prepare tensors
        h_wc = h_wc.unsqueeze(0)  # [1, 512]
        h_pos = torch.stack(positive_embeddings)  # [B_pos, 512]
        h_neg = torch.stack(negative_embeddings)  # [B_neg, 512]

        # SGD optimizer (fresh for each adaptation)
        optimizer = optim.SGD(
            personal_mlp.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        final_loss = 0.0

        for step in range(self.sgd_steps):
            optimizer.zero_grad()

            # Project through personal MLP
            proj_wc = personal_mlp(h_wc)     # [1, 64]
            proj_pos = personal_mlp(h_pos)   # [B_pos, 64]
            proj_neg = personal_mlp(h_neg)   # [B_neg, 64]

            # Triplet loss (Eq. 7)
            # For each (positive, negative) pair:
            # L = max(0, ||MLP(h_wc) - MLP(h_pos)||^2 - ||MLP(h_wc) - MLP(h_neg)||^2 + margin)
            dist_pos = ((proj_wc - proj_pos) ** 2).sum(dim=-1)  # [B_pos]
            dist_neg = ((proj_wc - proj_neg) ** 2).sum(dim=-1)  # [B_neg]

            # Cross all positive-negative pairs
            # Use mean of positive distances vs mean of negative distances
            loss_raw = dist_pos.mean() - dist_neg.mean()

            if self.margin == float("inf"):
                loss = dist_pos.mean()  # No margin: just pull positives closer
            else:
                loss = torch.clamp(loss_raw + self.margin, min=0)

            if loss.item() > 0:
                loss.backward()
                optimizer.step()

            final_loss = loss.item()

        personal_mlp.eval()
        adaptation_time_ms = (time.time() - start_time) * 1000

        return final_loss, adaptation_time_ms

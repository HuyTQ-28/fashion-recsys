import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.personal_mlp import PersonalMLP

logger = logging.getLogger(__name__)

def compute_weighted_centroid(interactions: List[Dict]) -> torch.Tensor:
    """Compute weighted centroid of interaction CLIP embeddings"""
    embeddings = torch.stack([item["clip_embedding"] for item in interactions])  # [B, 512]
    weights = torch.tensor(
        [item.get("weight", 1.0) for item in interactions], dtype=torch.float
    ).unsqueeze(1)  # [B, 1]

    B = len(interactions)   # |B_u| — the count, per paper Eq. 6
    return (embeddings * weights).sum(dim=0) / B


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet loss for a single (anchor, positive, negative) triple"""

    dist_pos = ((anchor - positive) ** 2).sum()
    dist_neg = ((anchor - negative) ** 2).sum()
    loss = torch.clamp(dist_pos - dist_neg + margin, min=0.0)
    return loss


def adapt_personal_mlp(
    personal_mlp: PersonalMLP,
    positive_clip_embeddings: List[torch.Tensor],
    negative_clip_embeddings: List[torch.Tensor],
    interaction_weights: Optional[List[float]] = None,
    sgd_steps: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    margin: float = float("inf"),
) -> Tuple[float, float]:
    """Run triplet loss adaptation on a Personal MLP in-place"""

    adapter = TripletAdaptation(
        sgd_steps=sgd_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        margin=margin,
    )
    return adapter.adapt(
        personal_mlp,
        positive_clip_embeddings,
        negative_clip_embeddings,
        interaction_weights,
    )

class TripletAdaptation:
    """
    Triplet Loss adaptation for Personal MLP.

    Adapts a user's Personal MLP in-place using recent interactions.
    """

    def __init__(
        self,
        sgd_steps: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        margin: float = float("inf"),
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
            weights: Optional weights (e.g., by interaction type).

        Returns:
            Weighted centroid tensor [512].
        """
        stacked = torch.stack(clip_embeddings)  # [B, D]
        B = stacked.shape[0]  # |B_u|
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float, device=stacked.device).unsqueeze(1)  # [B, 1]
            return (stacked * w).sum(dim=0) / B
        return stacked.mean(dim=0)

    def adapt(
        self,
        personal_mlp: PersonalMLP,
        positive_embeddings: List[torch.Tensor],
        negative_embeddings: List[torch.Tensor],
        interaction_weights: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Run triplet loss adaptation on a Personal MLP.

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

        # SGD optimizer
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

            # Paper Eq. 7 — pairwise triplet loss, summed over all (pos, neg) pairs.
            # Pair each positive with ONE negative (min-length pairing).
            n_pairs = min(proj_pos.shape[0], proj_neg.shape[0])
            proj_pos_paired = proj_pos[:n_pairs]   # [P, 64]
            proj_neg_paired = proj_neg[:n_pairs]   # [P, 64]

            dist_pos = ((proj_wc - proj_pos_paired) ** 2).sum(dim=-1)  # [P]
            dist_neg = ((proj_wc - proj_neg_paired) ** 2).sum(dim=-1)  # [P]

            if self.margin == float("inf"):
                # Degenerate margin=∞ case: only push positives closer to centroid
                loss = dist_pos.sum()
            else:
                # Standard triplet: max(0, d_pos - d_neg + ε), summed over pairs
                loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0).sum()

            if loss.item() > 0:
                loss.backward()
                optimizer.step()

            final_loss = loss.item()

        personal_mlp.eval()
        adaptation_time_ms = (time.time() - start_time) * 1000

        return final_loss, adaptation_time_ms

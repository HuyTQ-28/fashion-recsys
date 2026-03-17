"""
EMA User State Management.

Owner: Member 2 (Personalization Engine)

Paper reference: Section 2, Equation 5
u_t = (1 - alpha) * u_{t-1} + alpha * MLP_u(h_CNN_{t,u})
u_0 = MLP_u(h_CNN_{0,u})

The EMA vector represents the user's evolving interest.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class UserState:
    """
    Manages the EMA (Exponential Moving Average) user representation.

    The user vector is an EMA of the MLP-projected embeddings of interacted articles.
    K-nearest-neighbors to u_t in the MLP space are the recommendations.
    """

    def __init__(self, alpha: float = 0.5, embedding_dim: int = 64):
        """
        Args:
            alpha: EMA decay factor. Higher = more weight on recent interactions.
                   Search range: {0.1, 0.2, ..., 1.0}
            embedding_dim: Dimension of the MLP output space.
        """
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self.ema_vector: Optional[torch.Tensor] = None
        self.interaction_count: int = 0

    def update(self, mlp_projected_embedding: torch.Tensor) -> torch.Tensor:
        """
        Update the EMA user vector with a new interaction.

        Equation 5: u_t = (1 - alpha) * u_{t-1} + alpha * MLP_u(h_CNN_{t,u})

        Args:
            mlp_projected_embedding: MLP output for the interacted article [64].

        Returns:
            Updated EMA vector [64].
        """
        if self.ema_vector is None:
            # First interaction: u_0 = MLP_u(h_CNN_{0,u})
            self.ema_vector = mlp_projected_embedding.clone().detach()
        else:
            self.ema_vector = (
                (1 - self.alpha) * self.ema_vector
                + self.alpha * mlp_projected_embedding.detach()
            )

        self.interaction_count += 1
        return self.ema_vector

    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the current EMA vector, or None if no interactions yet."""
        return self.ema_vector

    def reset(self) -> None:
        """Reset user state (for day-by-day cold start mode)."""
        self.ema_vector = None
        self.interaction_count = 0

    def to_dict(self) -> dict:
        """Serialize user state for storage."""
        return {
            "alpha": self.alpha,
            "ema_vector": self.ema_vector.tolist() if self.ema_vector is not None else None,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserState":
        """Deserialize user state from storage."""
        state = cls(alpha=data["alpha"], embedding_dim=64)
        if data["ema_vector"] is not None:
            state.ema_vector = torch.tensor(data["ema_vector"])
        state.interaction_count = data["interaction_count"]
        return state

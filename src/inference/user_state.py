import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Interaction type -> weight mapping (H&M: purchase >> click)
INTERACTION_WEIGHTS: Dict[str, int] = {
    "purchase": 4,
    "cart": 3,
    "favorite": 2,
    "click": 1,
    "view": 1,
}


class UserState:
    """
    Manages the EMA (Exponential Moving Average) user representation.

    The user vector is an EMA of the MLP-projected embeddings of interacted articles.
    K-nearest-neighbors to u_t in the MLP space are the recommendations.
    """

    def __init__(self, user_id: str = "", alpha: float = 0.5, embedding_dim: int = 64):
        """
        Args:
            user_id: User identifier.
            alpha: EMA decay factor. Higher = more weight on recent interactions.
                   Search range: {0.1, 0.2, ..., 1.0}
            embedding_dim: Dimension of the MLP output space.
        """
        import time
        self.user_id = user_id
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self.ema_vector: Optional[torch.Tensor] = None
        self.interaction_count: int = 0
        self.interaction_history: List[Dict] = []
        self.last_updated_at: float = time.time()

    def update_ema(self, mlp_projected_embedding: torch.Tensor) -> torch.Tensor:
        """
        Update the EMA user vector with a new MLP-projected embedding.

        Equation 5: u_t = (1 - alpha) * u_{t-1} + alpha * MLP_u(h_CNN_{t,u})

        Args:
            mlp_projected_embedding: MLP output for the interacted article [64].

        Returns:
            Updated EMA vector [64].
        """
        import time
        emb = mlp_projected_embedding.detach().clone()
        if self.ema_vector is None:
            # First interaction: u_0 = MLP_u(h_CNN_{0,u})
            self.ema_vector = emb
        else:
            self.ema_vector = (1 - self.alpha) * self.ema_vector + self.alpha * emb

        self.interaction_count += 1
        self.last_updated_at = time.time()
        return self.ema_vector

    # Keep old name as alias for backward compat with SessionSimulator
    def update(self, mlp_projected_embedding: torch.Tensor) -> torch.Tensor:
        return self.update_ema(mlp_projected_embedding)

    def record_interaction(
        self,
        article_id: str,
        interaction_type: str,
        clip_embedding: torch.Tensor,
    ) -> None:
        """
        Record a raw interaction with its CLIP embedding and weight.

        Args:
            article_id: Article identifier.
            interaction_type: "purchase", "click", or "view".
            clip_embedding: Raw CLIP embedding [512] (before MLP projection).
        """
        import time
        weight = INTERACTION_WEIGHTS.get(interaction_type, 1)
        self.interaction_history.append({
            "article_id": article_id,
            "interaction_type": interaction_type,
            "clip_embedding": clip_embedding.detach().clone(),
            "weight": weight,
            "ts": time.time(),
        })

    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the current EMA vector, or None if no interactions yet."""
        return self.ema_vector

    def reset(self) -> None:
        """Reset user state (for day-by-day cold start mode)."""
        import time
        self.ema_vector = None
        self.interaction_count = 0
        self.interaction_history = []
        self.last_updated_at = time.time()

    def to_dict(self) -> dict:
        """Serialize user state for storage."""
        # Convert interaction history without large clip_embeddings
        recent_events = []
        for event in self.interaction_history[-50:]:  # Keep last 50
            recent_events.append({
                "article_id": event["article_id"],
                "interaction_type": event["interaction_type"],
                "weight": event.get("weight", 1),
                "ts": event.get("ts", self.last_updated_at)
            })
            
        return {
            "user_id": self.user_id,
            "alpha": self.alpha,
            "ema_vector": self.ema_vector.tolist() if self.ema_vector is not None else None,
            "interaction_count": self.interaction_count,
            "recent_events": recent_events,
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserState":
        """Deserialize user state from storage."""
        state = cls(
            user_id=data.get("user_id", ""),
            alpha=data.get("alpha", 0.5),
            embedding_dim=64,
        )
        if data.get("ema_vector") is not None:
            state.ema_vector = torch.tensor(data["ema_vector"])
        state.interaction_count = data.get("interaction_count", 0)
        state.last_updated_at = data.get("last_updated_at", 0.0)
        
        # Note: clip_embedding is omitted during serialization. It can be populated
        # later by looking up article_id in clip_embeddings if full history needed.
        recent_events = data.get("recent_events", [])
        state.interaction_history = []
        for ev in recent_events:
            state.interaction_history.append({
                "article_id": ev["article_id"],
                "interaction_type": ev["interaction_type"],
                "weight": ev.get("weight", 1),
                "ts": ev.get("ts", 0.0),
                "clip_embedding": torch.zeros(512),  # Placeholder, should be re-hydrated if needed
            })
            
        return state

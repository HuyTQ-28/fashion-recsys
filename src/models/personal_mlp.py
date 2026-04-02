"""
Personal MLP — Per-User Meta-Model.

Owner: Member 2 (Personalization Engine)

Paper reference: Section 2
- Architecture identical to Student MLP: [512, 256, 128, 64]
- Initialized as a deep copy of the trained Student MLP
- Continually adapted per user via Triplet Loss (Eq. 6-7)
- Serialized to ~700KB for Redis storage
"""

import copy
import io
import base64
import logging
from typing import Optional, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PersonalMLP(nn.Module):
    """
    Personal MLP — per-user copy of the Student MLP.

    Each user gets their own instance, initialized from the global Student MLP.
    The personal MLP projects products into a user-specific space that is
    continually adapted based on interactions.
    """

    def __init__(self, base_model=None, user_id: str = "", layer_dims: List[int] = None, base_mlp=None):
        """
        Initialize Personal MLP.

        Args:
            base_model: Trained Student MLP (or PersonalMLP) to deep-copy from.
                        If None, creates a new MLP with layer_dims (for deserialization).
            user_id: User identifier.
            layer_dims: Layer dimensions (used only if base_model is None).
        """
        super().__init__()

        self.user_id = user_id
        self.interaction_count = 0

        # Support both kwarg names
        base_model = base_model or base_mlp

        if base_model is not None:
            # Deep copy weights from base model.
            # Normalize: StudentMLP (real) uses .network, mock uses .net.
            # We always store as self.network so state_dict keys are consistent.
            if hasattr(base_model, "network"):
                src = base_model.network
                self.layer_dims = getattr(base_model, "layer_dims", [512, 256, 128, 64])
            elif hasattr(base_model, "net"):
                src = base_model.net
                self.layer_dims = layer_dims or [512, 256, 128, 64]
            elif isinstance(base_model, nn.Sequential):
                src = base_model
                self.layer_dims = layer_dims or [512, 256, 128, 64]
            else:
                # PersonalMLP copying another PersonalMLP
                src = base_model.network
                self.layer_dims = getattr(base_model, "layer_dims", [512, 256, 128, 64])
            self.network = copy.deepcopy(src)
            self.input_dim = self.layer_dims[0]
            self.output_dim = self.layer_dims[-1]
        else:
            dims = layer_dims or [512, 256, 128, 64]
            self.layer_dims = dims
            self.input_dim = dims[0]
            self.output_dim = dims[-1]

            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: CNN/CLIP embeddings [batch_size, 512].

        Returns:
            User-specific projected embeddings [batch_size, 64].
        """
        return self.network(x)

    # ------------------------------------------------------------------ #
    # Serialization (instance methods — used by tests and lifecycle mgr)  #
    # ------------------------------------------------------------------ #

    def serialize(self) -> str:
        """
        Serialize this Personal MLP to a base64-encoded string.

        Saves state_dict + metadata (user_id, interaction_count, layer_dims).
        Size: ~700KB.

        Returns:
            Base64-encoded string.
        """
        payload = {
            "state_dict": self.state_dict(),
            "user_id": self.user_id,
            "interaction_count": self.interaction_count,
            "layer_dims": self.layer_dims,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @classmethod
    def deserialize(cls, encoded: str, base_model=None) -> "PersonalMLP":
        """
        Deserialize a Personal MLP from a base64-encoded string.

        Args:
            encoded: Base64-encoded string produced by serialize().
            base_model: Optional base model (unused, kept for API compat).

        Returns:
            PersonalMLP with loaded weights and metadata.
        """
        raw_bytes = base64.b64decode(encoded.encode("ascii"))
        buffer = io.BytesIO(raw_bytes)
        payload = torch.load(buffer, weights_only=False)

        layer_dims = payload.get("layer_dims", [512, 256, 128, 64])
        mlp = cls(base_model=None, user_id=payload.get("user_id", ""), layer_dims=layer_dims)
        mlp.load_state_dict(payload["state_dict"])
        mlp.interaction_count = payload.get("interaction_count", 0)
        return mlp

    def get_size_kb(self) -> float:
        """Get approximate serialized size in KB."""
        return len(self.serialize()) / 1024


# ============================================================
# Factory — loads Student MLP checkpoint and creates copies
# ============================================================

class PersonalMLPFactory:
    """
    Creates PersonalMLP instances by deep-copying a loaded Student MLP.

    Usage:
        factory = PersonalMLPFactory("src/models/mlp_student.pt")
        mlp = factory.create("user_001")
    """

    def __init__(self, checkpoint_path: str, layer_dims: List[int] = None):
        """
        Args:
            checkpoint_path: Path to Student MLP state_dict (.pt file).
            layer_dims: Architecture dimensions (default [512, 256, 128, 64]).
        """
        self.checkpoint_path = checkpoint_path
        self.layer_dims = layer_dims or [512, 256, 128, 64]
        self._base_mlp = self._load(checkpoint_path)

    def _load(self, path: str) -> PersonalMLP:
        """Load Student MLP from checkpoint."""
        state_dict = torch.load(path, weights_only=True)
        mlp = PersonalMLP(base_model=None, layer_dims=self.layer_dims)
        # Handle both raw state_dict and wrapped dicts
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            mlp.load_state_dict(state_dict["state_dict"])
        else:
            # Student MLP may use "network.*" or "net.*" keys
            # Remap "net.*" -> "network.*" if needed
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("net."):
                    remapped["network." + k[4:]] = v
                else:
                    remapped[k] = v
            mlp.load_state_dict(remapped)
        return mlp

    def create(self, user_id: str) -> "PersonalMLP":
        """
        Create a new PersonalMLP for a user by deep-copying the base model.

        Args:
            user_id: User identifier.

        Returns:
            Fresh PersonalMLP initialized from Student MLP weights.
        """
        mlp = PersonalMLP(base_model=self._base_mlp, user_id=user_id, layer_dims=self.layer_dims)
        return mlp


# ============================================================
# Legacy standalone helpers (kept for backward compat)
# ============================================================

def serialize_mlp(mlp: PersonalMLP) -> str:
    return mlp.serialize()


def deserialize_mlp(encoded: str, layer_dims=None) -> PersonalMLP:
    return PersonalMLP.deserialize(encoded)


def get_serialized_size_kb(mlp: PersonalMLP) -> float:
    return mlp.get_size_kb()

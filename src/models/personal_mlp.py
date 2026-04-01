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
from typing import Optional

import torch
import torch.nn as nn

from src.models.student_mlp import StudentMLP

logger = logging.getLogger(__name__)


class PersonalMLP(nn.Module):
    """
    Personal MLP — per-user copy of the Student MLP.

    Each user gets their own instance, initialized from the global Student MLP.
    The personal MLP projects products into a user-specific space that is
    continually adapted based on interactions.
    """

    def __init__(self, base_mlp: Optional[StudentMLP] = None, layer_dims=None):
        """
        Initialize Personal MLP.

        Args:
            base_mlp: Trained Student MLP to deep-copy from.
                      If None, creates a new MLP (for deserialization).
            layer_dims: Layer dimensions (used only if base_mlp is None).
        """
        super().__init__()

        if base_mlp is not None:
            # Deep copy from trained Student MLP
            self.network = copy.deepcopy(base_mlp.network)
            self.layer_dims = base_mlp.layer_dims
            self.input_dim = base_mlp.input_dim
            self.output_dim = base_mlp.output_dim
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


# ============================================================
# Serialization helpers (for Redis storage)
# ============================================================

def serialize_mlp(mlp: PersonalMLP) -> str:
    """
    Serialize a Personal MLP to a base64-encoded string.

    Serializes the state_dict to bytes, then base64 encodes for
    safe storage in Redis REST API (which works with strings).

    Size: ~700KB per MLP.

    Args:
        mlp: PersonalMLP instance.

    Returns:
        Base64-encoded string of the serialized state_dict.
    """
    buffer = io.BytesIO()
    torch.save(mlp.state_dict(), buffer)
    raw_bytes = buffer.getvalue()
    return base64.b64encode(raw_bytes).decode("ascii")


def deserialize_mlp(encoded: str, layer_dims=None) -> PersonalMLP:
    """
    Deserialize a Personal MLP from a base64-encoded string.

    Args:
        encoded: Base64-encoded string.
        layer_dims: Layer dimensions for the MLP architecture.

    Returns:
        PersonalMLP instance with loaded weights.
    """
    raw_bytes = base64.b64decode(encoded.encode("ascii"))
    buffer = io.BytesIO(raw_bytes)
    state_dict = torch.load(buffer, weights_only=True)

    mlp = PersonalMLP(base_mlp=None, layer_dims=layer_dims)
    mlp.load_state_dict(state_dict)
    return mlp


def get_serialized_size_kb(mlp: PersonalMLP) -> float:
    """Get the approximate serialized size in KB."""
    encoded = serialize_mlp(mlp)
    return len(encoded) / 1024

import torch
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ClipRamIndex:
    """
    In-memory 512-dim CLIP embedding dictionary, used for fast O(1) RAM lookups
    during Stage-2 Reranking to avoid fetching large vectors from Weaviate.
    """
    def __init__(self, clip_dict: Dict[str, torch.Tensor]):
        """
        Args:
            clip_dict: Dict of article_id -> 512-dim tensor
        """
        # Normalize keys once so lookups are robust across int/str/zero-padded forms.
        self.clip_dict = {}
        for key, value in clip_dict.items():
            key_str = str(key)
            self.clip_dict[key_str] = value
            if key_str.isdigit():
                self.clip_dict[key_str.zfill(10)] = value
                self.clip_dict[str(int(key_str))] = value
        self.dim = next(iter(clip_dict.values())).shape[0] if clip_dict else 512
        logger.info(f"Loaded {len(self.clip_dict)} normalized vectors into RAM index")

    def gather(self, article_ids: List[str]) -> torch.Tensor:
        """
        Given a list of N article_ids, returns a [N, 512] tensor.
        Fallbacks to zero vector if ID is missing.
        """
        tensors = []
        for aid in article_ids:
            if aid in self.clip_dict:
                tensors.append(self.clip_dict[aid].detach().clone())
            else:
                tensors.append(torch.zeros(self.dim, dtype=torch.float32))
        
        if not tensors:
            return torch.empty((0, self.dim))
            
        return torch.stack(tensors)
        
    def get(self, article_id: str) -> torch.Tensor:
        """Get single vector or zero vector if missing."""
        key_str = str(article_id)
        key_candidates = [key_str]
        if key_str.isdigit():
            key_candidates.extend([key_str.zfill(10), str(int(key_str))])

        for key in key_candidates:
            if key in self.clip_dict:
                return self.clip_dict[key].detach().clone()
        return torch.zeros(self.dim, dtype=torch.float32)

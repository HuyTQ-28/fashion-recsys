import torch
import logging
from typing import List

from src.models.personal_mlp import PersonalMLP

logger = logging.getLogger(__name__)

class PersonalReranker:
    """
    Stage-2 Reranker: Projects candidate 512-dim CLIP vectors into the 
    user's Personal Space (64-dim) and calculates Euclidean distance to u_t_personal.
    """
    def __init__(self, clip_ram_index):
        self.clip_ram = clip_ram_index

    def rerank(
        self, 
        candidate_ids: List[str], 
        personal_weights_b64: str, 
        u_t_personal: List[float], 
        top_k: int = 10
    ) -> List[str]:
        if not candidate_ids:
            return []
            
        # [N, 512] O(1) RAM gather (N <= 500)
        x512 = self.clip_ram.gather(candidate_ids)

        # Deserialize the PersonalMLP weights
        model = PersonalMLP.deserialize(personal_weights_b64)
        model.eval()
        
        with torch.no_grad():
            # Project into Personal Space
            y64 = model(x512)  # Tensor[N,64]
            
            # Query vector
            q64 = torch.tensor(u_t_personal, dtype=torch.float32).view(1, 64)  # Tensor[1,64]
            
            # L2 distances
            d = torch.cdist(q64, y64, p=2).squeeze(0)  # Tensor[N]
            
            k = min(top_k, len(candidate_ids))
            _, idx = torch.sort(d)
            idx = idx[:k]

        return [candidate_ids[i] for i in idx.tolist()]

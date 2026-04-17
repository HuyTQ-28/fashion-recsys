import torch
from typing import List

from src.data_access.redis.session_store import InteractionEvent
from src.data_access.catalog.clip_ram_index import ClipRamIndex
from src.models.personal_mlp import PersonalMLP

def recompute_u_personal_from_history(
    model: PersonalMLP,
    history: List[InteractionEvent],
    alpha: float,
    clip_lookup: ClipRamIndex
) -> List[float]:
    """
    CRUCIAL INVALIDATION RULE: 
    Recomputes u_t_personal from scratch using the updated PersonalMLP 
    and the bounded interaction history.
    """
    model.eval()
    u = None
    
    # Ensure history is in chronological order
    sorted_history = sorted(history, key=lambda e: e.ts)
    
    with torch.no_grad():
        for ev in sorted_history:
            x512 = clip_lookup.get(ev.article_id).view(1, 512)
            z64 = model(x512).squeeze(0)
            
            if u is None:
                u = z64
            else:
                u = (1 - alpha) * u + alpha * z64
                
    if u is not None:
        return u.cpu().tolist()
        
    return [0.0] * 64

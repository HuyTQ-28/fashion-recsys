import torch
from typing import List, Optional

def update_ema_vector(
    old_vector: Optional[List[float]],
    new_embedding: torch.Tensor,
    alpha: float
) -> List[float]:
    """
    Update an Exponential Moving Average (EMA) vector.
    Equation 5: u_t = (1 - alpha) * u_{t-1} + alpha * MLP_u(h_CNN_{t,u})

    Args:
        old_vector: Previous EMA state (if any).
        new_embedding: Newly projected 64-dim interaction tensor.
        alpha: Decay factor.

    Returns:
        Updated EMA vector as a float list.
    """
    emb = new_embedding.detach().clone().squeeze()
    
    if not old_vector:
        # First interaction: u_0 = MLP_u(h_CNN_{0,u})
        return emb.cpu().tolist()
        
    old_vec = torch.tensor(old_vector, dtype=torch.float32)
    new_vec = (1 - alpha) * old_vec + alpha * emb
    
    return new_vec.cpu().tolist()

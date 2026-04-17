import pytest
import torch

from src.core.personalization.ema import update_ema_vector
from src.core.personalization.state_rebuilder import recompute_u_personal_from_history
from src.models.personal_mlp import PersonalMLP
from src.data_access.catalog.clip_ram_index import ClipRamIndex
from src.data_access.redis.session_store import InteractionEvent

def test_update_ema_vector_first_interaction():
    alpha = 0.7
    new_embedding = torch.ones(64)
    old_vector = None

    result = update_ema_vector(old_vector, new_embedding, alpha)
    
    assert len(result) == 64
    assert all(x == 1.0 for x in result), "First interaction should just be the new embedding"

def test_update_ema_vector_subsequent_interaction():
    alpha = 0.5
    old_vector = [1.0] * 64
    new_embedding = torch.zeros(64)

    result = update_ema_vector(old_vector, new_embedding, alpha)
    
    assert len(result) == 64
    assert all(abs(x - 0.5) < 1e-6 for x in result), "EMA should blend exactly by alpha"

def test_recompute_u_personal_from_history():
    # Setup fake model and clip index
    model = PersonalMLP(user_id="test", layer_dims=[512, 64])
    
    # We want a predictable output from the model to verify history reconstruction
    # Let's manually set weights to something trivial or just use the forward output
    
    # Fake clip ram index
    clip_dict = {
        "article1": torch.ones(512),
        "article2": torch.zeros(512),
    }
    clip_index = ClipRamIndex(clip_dict)
    
    history = [
        InteractionEvent(article_id="article1", interaction_type="click", weight=1, ts=100.0),
        InteractionEvent(article_id="article2", interaction_type="purchase", weight=4, ts=110.0),
    ]

    result = recompute_u_personal_from_history(
        model=model,
        history=history,
        alpha=0.5,
        clip_lookup=clip_index
    )
    
    assert len(result) == 64
    # Just checking it computes without error and returns 64-dim list
    # As the model is randomly initialized, the exact values will vary,
    # but the shape and type must be correct.
    assert isinstance(result[0], float)

def test_recompute_empty_history():
    model = PersonalMLP(user_id="test", layer_dims=[512, 64])
    clip_index = ClipRamIndex({})
    
    result = recompute_u_personal_from_history(
        model=model,
        history=[],
        alpha=0.5,
        clip_lookup=clip_index
    )
    
    assert len(result) == 64
    assert all(x == 0.0 for x in result), "Empty history should return zero vector"

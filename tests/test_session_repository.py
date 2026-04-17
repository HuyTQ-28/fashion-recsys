import json
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.inference.session_repository import SessionBundle, RedisSessionRepository
from src.inference.user_state import UserState
from src.models.personal_mlp import PersonalMLP

def test_user_state_serialization():
    state = UserState(user_id="u1", alpha=0.8)
    state.ema_vector = torch.ones(64)
    state.record_interaction("i1", "click", torch.zeros(512))
    
    data = state.to_dict()
    assert data["user_id"] == "u1"
    assert data["alpha"] == 0.8
    assert data["interaction_count"] == 0 # no update_ema called
    assert len(data["recent_events"]) == 1
    assert data["recent_events"][0]["article_id"] == "i1"
    
    # ensure no large tensors in recent_events
    assert "clip_embedding" not in data["recent_events"][0]
    
    restored = UserState.from_dict(data)
    assert restored.user_id == "u1"
    assert restored.alpha == 0.8
    assert torch.allclose(restored.ema_vector, state.ema_vector)
    assert len(restored.interaction_history) == 1
    assert restored.interaction_history[0]["article_id"] == "i1"
    
@patch("redis.Redis.from_url")
def test_redis_session_repository_save_load(mock_redis_cls):
    mock_redis = MagicMock()
    mock_redis_cls.return_value = mock_redis
    
    repo = RedisSessionRepository("redis://localhost", prefix="test")
    
    state = UserState("u1")
    state.ema_vector = torch.zeros(64)
    mlp = PersonalMLP(user_id="u1", layer_dims=[512, 64])
    
    bundle = SessionBundle(
        user_state=state,
        personal_mlp=mlp,
        adaptation_batch=["i1", "i2"]
    )
    
    # Test Save
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value.__enter__.return_value = mock_pipeline
    
    repo.save("u1", bundle, 3600)
    
    assert mock_pipeline.set.call_count == 2
    args_state, kwargs_state = mock_pipeline.set.call_args_list[0]
    args_model, kwargs_model = mock_pipeline.set.call_args_list[1]
    
    assert args_state[0] == "test:state:u1"
    assert kwargs_state["ex"] == 3600
    assert json.loads(args_state[1])["adaptation_batch"] == ["i1", "i2"]
    
    assert args_model[0] == "test:model:u1"
    
    # Test Load
    mock_pipeline.execute.return_value = (args_state[1], args_model[1])
    
    restored_bundle = repo.load("u1")
    assert restored_bundle is not None
    assert restored_bundle.user_state.user_id == "u1"
    assert restored_bundle.adaptation_batch == ["i1", "i2"]
    assert restored_bundle.personal_mlp.user_id == "u1"

@patch("redis.Redis.from_url")
def test_redis_session_repository_lock(mock_redis_cls):
    mock_redis = MagicMock()
    mock_redis_cls.return_value = mock_redis
    repo = RedisSessionRepository("redis://localhost")
    
    mock_redis.set.return_value = True
    assert repo.try_acquire_lock("u1", 30) is True
    mock_redis.set.assert_called_with("rec:lock:u1", "1", nx=True, ex=30)
    
    repo.release_lock("u1")
    mock_redis.delete.assert_called_with("rec:lock:u1")

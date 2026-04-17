import pytest
from unittest.mock import MagicMock

from src.core.recommender.pipeline import RecommendPipeline
from src.data_access.redis.session_store import SessionSnapshot

def test_pipeline_enforces_space_separation():
    # Setup mocks
    mock_redis = MagicMock()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    
    pipeline = RecommendPipeline(
        redis_store=mock_redis,
        retriever=mock_retriever,
        reranker=mock_reranker
    )
    
    # Create fake snapshot
    mock_snap = SessionSnapshot(
        u_t_base=[0.1] * 64,       # Global space vector
        u_t_personal=[0.9] * 64,   # Personal space vector
        personal_weights="base64_fake_weights",
        model_version=1,
        state_version=1,
        updated_at=100.0,
        alpha_personal=0.7
    )
    mock_redis.read_snapshot.return_value = mock_snap
    
    # Stage 1 returns some candidates
    mock_retriever.search_ids_near_vector.return_value = ["c1", "c2", "c3"]
    
    # Stage 2 returns subset
    mock_reranker.rerank.return_value = ["c2", "c1"]
    
    res = pipeline.recommend("test_session", k=2, seed_article_id="seed1")
    
    # INVARIANT 1: Retriever must be called with u_t_base (Global Space)
    mock_retriever.search_ids_near_vector.assert_called_once_with(
        query_vec64=mock_snap.u_t_base, 
        limit=500
    )
    
    # INVARIANT 2: Reranker must be called with u_t_personal (Personal Space)
    mock_reranker.rerank.assert_called_once_with(
        candidate_ids=["c1", "c2", "c3"], # assuming seed1 wasn't in candidates
        personal_weights_b64=mock_snap.personal_weights,
        u_t_personal=mock_snap.u_t_personal,
        top_k=2
    )
    
    assert res["article_ids"] == ["c2", "c1"]
    assert res["model_version"] == 1

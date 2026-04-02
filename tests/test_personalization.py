# tests/test_personalization.py
import torch
import pytest
from src.models.mock_student_mlp import StudentMLP, create_mock_checkpoint
from src.models.personal_mlp import PersonalMLP, PersonalMLPFactory
from src.inference.user_state import UserState
from src.inference.mlp_lifecycle import (
    LocalDictBackend, LRUCache, MLPLifecycleManager
)
from src.training.adapt_personal_mlp import (
    compute_weighted_centroid, triplet_loss, adapt_personal_mlp
)


@pytest.fixture
def base_model():
    model = StudentMLP()
    model.eval()
    return model


@pytest.fixture
def factory(tmp_path):
    path = str(tmp_path / "student_mlp.pt")
    create_mock_checkpoint(path)
    return PersonalMLPFactory(path)


@pytest.fixture
def clip_embeddings():
    return {f"article_{i:04d}": torch.randn(512) for i in range(100)}


# ─── Personal MLP Tests ──────────────────────────────────────────────────────

class TestPersonalMLP:
    def test_output_shape(self, base_model):
        mlp = PersonalMLP(base_model, "user_001")
        x = torch.randn(512)
        out = mlp(x.unsqueeze(0))
        assert out.shape == (1, 64), "Output phải là 64-dim"
    
    def test_serialize_deserialize(self, base_model):
        mlp = PersonalMLP(base_model, "user_001")
        mlp.interaction_count = 5
        
        data = mlp.serialize()
        restored = PersonalMLP.deserialize(data, base_model)
        
        assert restored.user_id == "user_001"
        assert restored.interaction_count == 5
    
    def test_size_under_1mb(self, base_model):
        mlp = PersonalMLP(base_model, "user_001")
        size_kb = mlp.get_size_kb()
        assert size_kb < 1024, f"Model quá lớn: {size_kb:.1f}KB (target <1MB)"
    
    def test_independence_from_base(self, base_model):
        """Đảm bảo Personal MLP độc lập với base model."""
        mlp1 = PersonalMLP(base_model, "user_001")
        mlp2 = PersonalMLP(base_model, "user_002")
        
        # Modify mlp1
        for p in mlp1.parameters():
            p.data += 1.0
        
        # mlp2 không bị ảnh hưởng
        x = torch.randn(1, 512)
        out1 = mlp1(x)
        out2 = mlp2(x)
        assert not torch.allclose(out1, out2)


# ─── EMA Tests ───────────────────────────────────────────────────────────────

class TestUserState:
    def test_ema_first_interaction(self):
        state = UserState(user_id="u1", alpha=0.3)
        emb = torch.randn(64)
        result = state.update_ema(emb)
        assert torch.allclose(result, emb), "Lần đầu: EMA = embedding"
    
    def test_ema_update_formula(self):
        state = UserState(user_id="u1", alpha=0.3)
        emb1 = torch.ones(64)
        emb2 = torch.zeros(64)
        
        state.update_ema(emb1)
        result = state.update_ema(emb2)
        
        expected = 0.7 * emb1 + 0.3 * emb2
        assert torch.allclose(result, expected)
    
    def test_interaction_weights(self):
        state = UserState(user_id="u1")
        state.record_interaction("a1", "purchase", torch.randn(512))
        state.record_interaction("a2", "click", torch.randn(512))
        
        assert state.interaction_history[0]["weight"] == 4
        assert state.interaction_history[1]["weight"] == 1


# ─── Triplet Loss Tests ──────────────────────────────────────────────────────

class TestTripletLoss:
    def test_zero_loss_when_positive_closer(self):
        anchor = torch.zeros(64)
        positive = torch.ones(64) * 0.1   # Gần anchor
        negative = torch.ones(64) * 10.0  # Xa anchor
        loss = triplet_loss(anchor, positive, negative, margin=1.0)
        assert loss.item() == 0.0
    
    def test_positive_loss_when_negative_closer(self):
        anchor = torch.zeros(64)
        positive = torch.ones(64) * 10.0  # Xa anchor
        negative = torch.ones(64) * 0.1   # Gần anchor
        loss = triplet_loss(anchor, positive, negative, margin=1.0)
        assert loss.item() > 0.0
    
    def test_weighted_centroid(self):
        interactions = [
            {"clip_embedding": torch.ones(512), "weight": 4},
            {"clip_embedding": torch.zeros(512), "weight": 0},
        ]
        centroid = compute_weighted_centroid(interactions)
        assert centroid.shape == (512,)


# ─── LRU Cache Tests ─────────────────────────────────────────────────────────

class TestLRUCache:
    def test_eviction(self, base_model):
        cache = LRUCache(max_size=2)
        mlp1 = PersonalMLP(base_model, "u1")
        mlp2 = PersonalMLP(base_model, "u2")
        mlp3 = PersonalMLP(base_model, "u3")
        
        cache.put("u1", mlp1)
        cache.put("u2", mlp2)
        evicted = cache.put("u3", mlp3)  # Phải evict u1
        
        assert evicted == "u1"
        assert "u1" not in cache
        assert "u3" in cache
    
    def test_lru_order(self, base_model):
        cache = LRUCache(max_size=2)
        mlp1 = PersonalMLP(base_model, "u1")
        mlp2 = PersonalMLP(base_model, "u2")
        mlp3 = PersonalMLP(base_model, "u3")
        
        cache.put("u1", mlp1)
        cache.put("u2", mlp2)
        cache.get("u1")  # Access u1 → u1 trở thành MRU
        evicted = cache.put("u3", mlp3)  # Phải evict u2 (LRU)
        
        assert evicted == "u2"


# ─── Integration Test ────────────────────────────────────────────────────────

class TestPersonalizationEngine:
    def test_end_to_end(self, factory, clip_embeddings):
        from src.inference.mlp_lifecycle import LocalDictBackend, MLPLifecycleManager
        from src.inference.recommender import PersonalizationEngine
        
        storage = LocalDictBackend()
        lifecycle = MLPLifecycleManager(factory, storage)
        engine = PersonalizationEngine(lifecycle, clip_embeddings)
        
        article_ids = list(clip_embeddings.keys())
        
        # Handle interactions
        for i in range(6):
            result = engine.handle_interaction(
                user_id="user_test",
                article_id=article_ids[i],
                interaction_type="purchase",
                shown_articles=article_ids[i+1:i+6]
            )
        
        # Get recommendations
        recs = engine.get_recommendations(
            user_id="user_test",
            seed_article_id=article_ids[0],
            k=10
        )
        
        assert len(recs) == 10
        assert all(aid in clip_embeddings for aid in recs)
        assert article_ids[0] not in recs  # Không gợi ý lại seed

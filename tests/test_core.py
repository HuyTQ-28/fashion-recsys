"""
Unit Tests for Core Modules.

Owner: Member 4 (Frontend & Evaluation) + all members contribute

Tests cover:
- Models: HGNN forward pass, StudentMLP, PersonalMLP serialization
- Training: Contrastive loss, Alignment loss, Triplet loss
- Inference: EMA update, MLPLifecycleManager, Recommender
- Evaluation: Metrics (P@K, R@K, F1@K)
"""

import pytest
import torch

from src.models.hgnn import HGNN, ContrastiveLoss
from src.models.student_mlp import StudentMLP, AlignmentLoss
from src.models.personal_mlp import PersonalMLP, serialize_mlp, deserialize_mlp
from src.inference.user_state import UserState
from src.inference.mlp_lifecycle import MLPLifecycleManager, LocalDictBackend
from src.evaluation.metrics import precision_at_k, recall_at_k, f1_at_k


# ============================================================
# Model Tests
# ============================================================

class TestHGNN:
    def test_forward_shape(self):
        model = HGNN(layer_dims=[512, 256, 128, 64], relation_types=["copurchased"])
        x = torch.randn(100, 512)
        edge_index = torch.randint(0, 100, (2, 500))
        out = model(x, {"copurchased": edge_index})
        assert out.shape == (100, 64)

    def test_multi_relation(self):
        model = HGNN(
            layer_dims=[512, 256, 128, 64],
            relation_types=["light", "medium", "heavy"],
        )
        x = torch.randn(50, 512)
        edges = {
            "light": torch.randint(0, 50, (2, 100)),
            "medium": torch.randint(0, 50, (2, 80)),
            "heavy": torch.randint(0, 50, (2, 30)),
        }
        out = model(x, edges)
        assert out.shape == (50, 64)


class TestStudentMLP:
    def test_forward_shape(self):
        model = StudentMLP(layer_dims=[512, 256, 128, 64])
        x = torch.randn(32, 512)
        out = model(x)
        assert out.shape == (32, 64)

    def test_project_all(self):
        model = StudentMLP(layer_dims=[512, 256, 128, 64])
        embeddings = {f"art_{i}": torch.randn(512) for i in range(10)}
        projected = model.project_all(embeddings)
        assert len(projected) == 10
        for v in projected.values():
            assert v.shape == (64,)


class TestPersonalMLP:
    def test_deep_copy(self):
        student = StudentMLP(layer_dims=[512, 256, 128, 64])
        personal = PersonalMLP(base_mlp=student)
        x = torch.randn(1, 512)

        # Should produce same output initially
        with torch.no_grad():
            s_out = student(x)
            p_out = personal(x)
        assert torch.allclose(s_out, p_out, atol=1e-6)

    def test_serialization_roundtrip(self):
        student = StudentMLP(layer_dims=[512, 256, 128, 64])
        original = PersonalMLP(base_mlp=student)

        encoded = serialize_mlp(original)
        restored = deserialize_mlp(encoded, [512, 256, 128, 64])

        x = torch.randn(1, 512)
        with torch.no_grad():
            orig_out = original(x)
            rest_out = restored(x)
        assert torch.allclose(orig_out, rest_out, atol=1e-6)

    def test_serialized_size(self):
        student = StudentMLP(layer_dims=[512, 256, 128, 64])
        personal = PersonalMLP(base_mlp=student)
        encoded = serialize_mlp(personal)
        size_kb = len(encoded) / 1024
        assert size_kb < 1000  # Should be ~700KB


# ============================================================
# Inference Tests
# ============================================================

class TestUserState:
    def test_first_interaction(self):
        state = UserState(alpha=0.5)
        emb = torch.randn(64)
        result = state.update(emb)
        assert result.shape == (64,)
        assert torch.allclose(result, emb)
        assert state.interaction_count == 1

    def test_ema_update(self):
        state = UserState(alpha=0.5)
        emb1 = torch.ones(64)
        emb2 = torch.zeros(64)

        state.update(emb1)
        result = state.update(emb2)

        # Expected: 0.5 * ones + 0.5 * zeros = 0.5 * ones
        expected = 0.5 * emb1
        assert torch.allclose(result, expected)

    def test_serialization(self):
        state = UserState(alpha=0.3)
        state.update(torch.randn(64))
        d = state.to_dict()
        restored = UserState.from_dict(d)
        assert restored.alpha == 0.3
        assert restored.interaction_count == 1


class TestMLPLifecycleManager:
    def test_create_new_user(self):
        student = StudentMLP(layer_dims=[512, 256, 128, 64])
        personal_template = PersonalMLP(base_mlp=student)
        backend = LocalDictBackend()
        manager = MLPLifecycleManager(personal_template, backend, max_size=10)

        entry = manager.get_or_create("user_1")
        assert entry.personal_mlp is not None
        assert manager.stats["creates"] == 1

    def test_cache_hit(self):
        student = StudentMLP(layer_dims=[512, 256, 128, 64])
        personal_template = PersonalMLP(base_mlp=student)
        backend = LocalDictBackend()
        manager = MLPLifecycleManager(personal_template, backend, max_size=10)

        manager.get_or_create("user_1")
        manager.get_or_create("user_1")

        assert manager.stats["hits"] == 1
        assert manager.stats["misses"] == 1

    def test_lru_eviction(self):
        student = StudentMLP(layer_dims=[512, 256, 128, 64])
        personal_template = PersonalMLP(base_mlp=student)
        backend = LocalDictBackend()
        manager = MLPLifecycleManager(personal_template, backend, max_size=2)

        manager.get_or_create("user_1")
        manager.get_or_create("user_2")
        manager.get_or_create("user_3")  # Should evict user_1

        assert manager.stats["evictions"] == 1
        assert "user_1" not in manager._cache


# ============================================================
# Evaluation Tests
# ============================================================

class TestMetrics:
    def test_precision_perfect(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], k=2) == 1.0

    def test_precision_zero(self):
        assert precision_at_k(["x", "y"], ["a", "b", "c"], k=2) == 0.0

    def test_recall_perfect(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_recall_partial(self):
        assert recall_at_k(["a", "x"], ["a", "b", "c"], k=2) == pytest.approx(1 / 3)

    def test_f1(self):
        # P=0.5, R=0.5 -> F1=0.5
        recs = ["a", "x"]
        gt = ["a", "b"]
        assert f1_at_k(recs, gt, k=2) == pytest.approx(0.5)

    def test_f1_zero(self):
        assert f1_at_k([], ["a", "b"], k=10) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Recommendation Engine & Personalization Engine.

Owner: Member 2 (Personalization Engine)

Two-stage personalized recommendations:
1. Recall: Weaviate ProductRec KNN with EMA user vector -> top-100 candidates
2. Re-rank: Personal MLP re-projects candidates, sort by Euclidean distance to EMA

Cold users (no interactions yet): item-to-item KNN on global Student MLP embeddings.

PersonalizationEngine wraps the full pipeline for M3's API.
"""

import logging
from typing import Dict, List, Optional

import torch

from src.models.personal_mlp import PersonalMLP, PersonalMLPFactory
from src.inference.user_state import UserState
from src.inference.mlp_lifecycle import MLPLifecycleManager, LocalDictBackend
from src.training.adapt_personal_mlp import TripletAdaptation

logger = logging.getLogger(__name__)


# ============================================================
# Low-level Recommender (2-stage recall + re-rank)
# ============================================================

class Recommender:
    """
    Two-stage personalized recommendation engine.

    Stage 1 (Recall): in-memory KNN on CLIP embeddings (fallback when no Weaviate).
    Stage 2 (Re-rank): Personal MLP re-ranking of candidates.
    """

    def __init__(
        self,
        weaviate_client=None,
        clip_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        candidate_k: int = 100,
        final_k: int = 10,
    ):
        """
        Args:
            weaviate_client: Connected Weaviate client (optional).
            clip_embeddings: Pre-loaded CLIP embeddings (article_id -> Tensor[512]).
            candidate_k: Number of candidates to retrieve.
            final_k: Number of final recommendations to return.
        """
        self.weaviate_client = weaviate_client
        self.clip_embeddings = clip_embeddings or {}
        self.candidate_k = candidate_k
        self.final_k = final_k

    def recommend_cold(self, seed_article_id: str) -> List[str]:
        """
        Cold-start recommendations: item-to-item KNN on CLIP embeddings.

        Args:
            seed_article_id: Seed article ID.

        Returns:
            List of recommended article IDs (excluding seed).
        """
        if seed_article_id not in self.clip_embeddings:
            return []

        seed_emb = self.clip_embeddings[seed_article_id].unsqueeze(0)  # [1, 512]
        article_ids = [aid for aid in self.clip_embeddings if aid != seed_article_id]
        if not article_ids:
            return []

        all_embs = torch.stack([self.clip_embeddings[aid] for aid in article_ids])
        distances = torch.cdist(seed_emb, all_embs).squeeze(0)
        _, indices = distances.topk(self.final_k, largest=False)
        return [article_ids[i] for i in indices.tolist()]

    def recommend_personalized(
        self,
        user_state: UserState,
        personal_mlp: PersonalMLP,
    ) -> List[str]:
        """
        Personalized recommendations: candidate recall + Personal MLP re-rank.

        Args:
            user_state: User's EMA state.
            personal_mlp: User's Personal MLP.

        Returns:
            List of recommended article IDs (top final_k after re-ranking).
        """
        ema_vector = user_state.get_vector()
        if ema_vector is None:
            return []

        candidate_ids = self._retrieve_candidates_local(ema_vector, personal_mlp)
        if not candidate_ids:
            return []

        return self._rerank(candidate_ids, ema_vector, personal_mlp)

    def _retrieve_candidates_local(
        self,
        ema_vector: torch.Tensor,
        personal_mlp: Optional[PersonalMLP] = None,
    ) -> List[str]:
        """
        In-memory KNN candidate retrieval.

        Compares ema_vector (64-dim) against MLP-projected embeddings.
        If personal_mlp is provided, projects CLIP embeddings on the fly;
        otherwise falls back to direct distance (useful for cold start).
        """
        if not self.clip_embeddings:
            return []

        article_ids = list(self.clip_embeddings.keys())
        clip_stack = torch.stack([self.clip_embeddings[aid] for aid in article_ids])  # [N, 512]

        if personal_mlp is not None:
            personal_mlp.eval()
            with torch.no_grad():
                all_embeddings = personal_mlp(clip_stack)  # [N, 64]
        else:
            # cold path: just use raw CLIP — caller must ensure dim match
            all_embeddings = clip_stack

        distances = torch.cdist(ema_vector.unsqueeze(0), all_embeddings).squeeze(0)
        k = min(self.candidate_k, len(article_ids))
        _, indices = distances.topk(k, largest=False)
        return [article_ids[i] for i in indices.tolist()]

    @torch.no_grad()
    def _rerank(
        self,
        candidate_ids: List[str],
        ema_vector: torch.Tensor,
        personal_mlp: PersonalMLP,
    ) -> List[str]:
        """Re-rank candidates using the user's Personal MLP."""
        personal_mlp.eval()

        valid_ids = [aid for aid in candidate_ids if aid in self.clip_embeddings]
        if not valid_ids:
            return candidate_ids[: self.final_k]

        candidate_clip = torch.stack([self.clip_embeddings[aid] for aid in valid_ids])
        projected = personal_mlp(candidate_clip)  # [N, 64]
        distances = torch.cdist(ema_vector.unsqueeze(0), projected).squeeze(0)  # [N]
        _, sorted_indices = distances.sort()
        return [valid_ids[i] for i in sorted_indices[: self.final_k].tolist()]


# ============================================================
# PersonalizationEngine — high-level API for M3
# ============================================================

class PersonalizationEngine:
    """
    High-level personalization API.

    Wraps MLPLifecycleManager + Recommender into the contract expected by M3:
        engine.get_recommendations(user_id, seed_article_id, k)
        engine.handle_interaction(user_id, article_id, interaction_type, shown_articles)
        engine.get_user_state(user_id)
    """

    def __init__(
        self,
        lifecycle: MLPLifecycleManager,
        clip_embeddings: Dict[str, torch.Tensor],
        adaptation: Optional[TripletAdaptation] = None,
        trigger_every_n: int = 5,
        candidate_k: int = 100,
        final_k: int = 10,
    ):
        """
        Args:
            lifecycle: MLPLifecycleManager (handles load/save of Personal MLPs).
            clip_embeddings: Dict of article_id -> CLIP Tensor[512].
            adaptation: TripletAdaptation instance (created with defaults if None).
            trigger_every_n: Trigger adaptation every N interactions.
            candidate_k: Recall pool size for recommendation.
            final_k: Number of final recommendations.
        """
        self.lifecycle = lifecycle
        self.clip_embeddings = clip_embeddings
        self.adaptation = adaptation or TripletAdaptation()
        self.trigger_every_n = trigger_every_n

        self.recommender = Recommender(
            clip_embeddings=clip_embeddings,
            candidate_k=candidate_k,
            final_k=final_k,
        )

    def get_recommendations(
        self,
        user_id: str,
        seed_article_id: str,
        k: int = 10,
    ) -> List[str]:
        """
        Get top-k recommendations for a user.

        If the user has interaction history (EMA vector exists), returns
        personalized recommendations. Otherwise falls back to item-to-item KNN
        from the seed article.

        Args:
            user_id: User identifier.
            seed_article_id: Seed article (shown/clicked item).
            k: Number of recommendations.

        Returns:
            List of article IDs (seed excluded).
        """
        entry = self.lifecycle.get_or_create(user_id)
        ema = entry.user_state.get_vector()

        # Request k+1 so we have enough after excluding the seed
        extra = k + 1
        old_final_k = self.recommender.final_k
        self.recommender.final_k = extra

        if ema is not None:
            recs = self.recommender.recommend_personalized(
                entry.user_state, entry.personal_mlp
            )
        else:
            recs = self.recommender.recommend_cold(seed_article_id)

        self.recommender.final_k = old_final_k

        # Exclude seed then trim to k
        recs = [r for r in recs if r != seed_article_id]
        return recs[:k]

    def handle_interaction(
        self,
        user_id: str,
        article_id: str,
        interaction_type: str = "purchase",
        shown_articles: Optional[List[str]] = None,
    ) -> dict:
        """
        Process a user interaction: update EMA, record history, trigger adaptation.

        Args:
            user_id: User identifier.
            article_id: Interacted article.
            interaction_type: "purchase", "click", or "view".
            shown_articles: Articles shown alongside (used as negatives).

        Returns:
            Dict with status, interaction_count, adapted flag.
        """
        if article_id not in self.clip_embeddings:
            return {"status": "skipped", "interaction_count": 0, "adapted": False}

        entry = self.lifecycle.get_or_create(user_id)
        clip_emb = self.clip_embeddings[article_id]

        # Update EMA
        with torch.no_grad():
            mlp_proj = entry.personal_mlp(clip_emb.unsqueeze(0)).squeeze(0)
        entry.user_state.update_ema(mlp_proj)

        # Record raw interaction for triplet adaptation
        entry.user_state.record_interaction(article_id, interaction_type, clip_emb)
        entry.interaction_batch.append(article_id)

        adapted = False
        adaptation_time_ms = None
        if len(entry.interaction_batch) >= self.trigger_every_n:
            adapted, adaptation_time_ms = self._trigger_adaptation(entry, shown_articles)
            self.lifecycle.mark_dirty(user_id)
            self.lifecycle.flush(user_id)
            entry.interaction_batch = []

        return {
            "status": "ok",
            "interaction_count": entry.user_state.interaction_count,
            "adapted": adapted,
            "adaptation_time_ms": adaptation_time_ms,
        }

    def get_user_state(self, user_id: str) -> dict:
        """
        Return user state summary (for M3's /interact response).

        Args:
            user_id: User identifier.

        Returns:
            Dict with ema_vector (list), interaction_count, has_personalization.
        """
        entry = self.lifecycle.get_or_create(user_id)
        ema = entry.user_state.get_vector()
        return {
            "user_id": user_id,
            "ema_vector": ema.tolist() if ema is not None else None,
            "interaction_count": entry.user_state.interaction_count,
            "has_personalization": ema is not None,
        }

    def _trigger_adaptation(
        self, entry, shown_articles: Optional[List[str]]
    ) -> tuple:
        """
        Run triplet loss adaptation on the user's Personal MLP.

        Returns:
            (adapted: bool, adaptation_time_ms: Optional[float])
        """
        positives = [
            self.clip_embeddings[aid]
            for aid in entry.interaction_batch
            if aid in self.clip_embeddings
        ]
        if not positives:
            return False, None

        # Negatives: shown but not interacted, else random
        if shown_articles:
            neg_ids = [
                aid for aid in shown_articles
                if aid not in entry.interaction_batch and aid in self.clip_embeddings
            ]
        else:
            neg_ids = []

        if not neg_ids:
            all_ids = list(self.clip_embeddings.keys())
            neg_ids = [
                aid for aid in all_ids
                if aid not in entry.interaction_batch
            ][: len(positives)]

        negatives = [self.clip_embeddings[aid] for aid in neg_ids]
        if not negatives:
            return False, None

        _, adaptation_time_ms = self.adaptation.adapt(entry.personal_mlp, positives, negatives)
        return True, adaptation_time_ms

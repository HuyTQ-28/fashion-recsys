"""
Recommendation Engine.

Owner: Member 2 (Personalization Engine)

Two-stage personalized recommendations:
1. Recall: Weaviate ProductRec KNN with EMA user vector -> top-100 candidates
2. Re-rank: Personal MLP re-projects candidates, sort by Euclidean distance to EMA

Cold users (no Personal MLP): use global Student MLP embeddings from Weaviate.
"""

import logging
from typing import Dict, List, Optional

import torch
import weaviate
from weaviate.classes.query import MetadataQuery

from src.models.personal_mlp import PersonalMLP
from src.inference.user_state import UserState

logger = logging.getLogger(__name__)


class Recommender:
    """
    Two-stage personalized recommendation engine.

    Stage 1 (Recall): Weaviate KNN on ProductRec collection.
    Stage 2 (Re-rank): Personal MLP re-ranking of candidates.
    """

    def __init__(
        self,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        clip_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        candidate_k: int = 100,
        final_k: int = 10,
    ):
        """
        Args:
            weaviate_client: Connected Weaviate client (for ProductRec collection).
            clip_embeddings: Pre-loaded CLIP embeddings for re-ranking (article_id -> Tensor[512]).
            candidate_k: Number of candidates to retrieve from Weaviate.
            final_k: Number of final recommendations to return.
        """
        self.weaviate_client = weaviate_client
        self.clip_embeddings = clip_embeddings or {}
        self.candidate_k = candidate_k
        self.final_k = final_k

    def recommend_cold(self, seed_article_id: str) -> List[str]:
        """
        Cold-start recommendations: item-to-item KNN via Weaviate ProductRec.

        Args:
            seed_article_id: Article ID to get recommendations for.

        Returns:
            List of recommended article IDs.
        """
        if self.weaviate_client is None:
            return []

        collection = self.weaviate_client.collections.get("ProductRec")

        # Get seed article's vector from Weaviate
        result = collection.query.fetch_object_by_id(
            # We need to find by article_id property
        )

        # Query by near_vector with seed article's embedding
        # (In practice, look up the seed's MLP embedding from Weaviate)
        # Simplified: use the collection's near_object query
        results = collection.query.bm25(
            query=seed_article_id,
            limit=1,
        )

        if not results.objects:
            return []

        seed_vector = results.objects[0].vector
        if seed_vector is None:
            return []

        # KNN search
        recommendations = collection.query.near_vector(
            near_vector=seed_vector,
            limit=self.final_k + 1,  # +1 to exclude seed
            return_metadata=MetadataQuery(distance=True),
        )

        return [
            obj.properties["article_id"]
            for obj in recommendations.objects
            if obj.properties["article_id"] != seed_article_id
        ][: self.final_k]

    def recommend_personalized(
        self,
        user_state: UserState,
        personal_mlp: PersonalMLP,
    ) -> List[str]:
        """
        Personalized recommendations: Weaviate recall + Personal MLP re-rank.

        Stage 1: Retrieve top-K candidates from Weaviate ProductRec using EMA vector.
        Stage 2: Re-project candidates through Personal MLP, re-rank by distance.

        Args:
            user_state: User's EMA state (contains u_t vector).
            personal_mlp: User's Personal MLP.

        Returns:
            List of recommended article IDs (top final_k after re-ranking).
        """
        ema_vector = user_state.get_vector()
        if ema_vector is None:
            return []

        # Stage 1: Weaviate candidate retrieval
        candidate_ids = self._retrieve_candidates(ema_vector)
        if not candidate_ids:
            return []

        # Stage 2: Personal MLP re-ranking
        return self._rerank(candidate_ids, ema_vector, personal_mlp)

    def _retrieve_candidates(self, ema_vector: torch.Tensor) -> List[str]:
        """Retrieve top-K candidates from Weaviate ProductRec."""
        if self.weaviate_client is None:
            # Fallback: simple KNN on in-memory embeddings
            return self._retrieve_candidates_local(ema_vector)

        collection = self.weaviate_client.collections.get("ProductRec")

        results = collection.query.near_vector(
            near_vector=ema_vector.tolist(),
            limit=self.candidate_k,
        )

        return [obj.properties["article_id"] for obj in results.objects]

    def _retrieve_candidates_local(self, ema_vector: torch.Tensor) -> List[str]:
        """
        Fallback: in-memory KNN when Weaviate is not available.
        Used during development with mock data.
        """
        if not self.clip_embeddings:
            return []

        article_ids = list(self.clip_embeddings.keys())
        all_embeddings = torch.stack([self.clip_embeddings[aid] for aid in article_ids])

        # Simple Euclidean distance KNN
        distances = torch.cdist(ema_vector.unsqueeze(0), all_embeddings).squeeze(0)
        _, indices = distances.topk(self.candidate_k, largest=False)

        return [article_ids[i] for i in indices.tolist()]

    @torch.no_grad()
    def _rerank(
        self,
        candidate_ids: List[str],
        ema_vector: torch.Tensor,
        personal_mlp: PersonalMLP,
    ) -> List[str]:
        """
        Re-rank candidates using the user's Personal MLP.

        Projects each candidate's CLIP embedding through the Personal MLP,
        computes Euclidean distance to the user's EMA vector.
        """
        personal_mlp.eval()

        # Get CLIP embeddings for candidates
        valid_ids = [aid for aid in candidate_ids if aid in self.clip_embeddings]
        if not valid_ids:
            return candidate_ids[: self.final_k]

        candidate_clip = torch.stack([self.clip_embeddings[aid] for aid in valid_ids])

        # Project through Personal MLP
        projected = personal_mlp(candidate_clip)  # [N, 64]

        # Euclidean distance to EMA vector
        distances = torch.cdist(ema_vector.unsqueeze(0), projected).squeeze(0)  # [N]

        # Sort by distance (ascending = closest first)
        _, sorted_indices = distances.sort()

        return [valid_ids[i] for i in sorted_indices[: self.final_k].tolist()]

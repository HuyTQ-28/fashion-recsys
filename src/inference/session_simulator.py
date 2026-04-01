"""
Interaction Simulation Engine.

Owner: Member 2 (Personalization Engine)

Replays historical user sessions from test data to evaluate
the recommendation system with continual personalization.

Paper reference: Section 3
- Day-by-day cold start: delete user state at end of each day
- Multi-week personalization: keep state across days (1, 2, 3 weeks)
- Metrics: P@K, R@K, F1@K (K=10, ground truth T=12 next purchases)
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from src.models.personal_mlp import PersonalMLP
from src.inference.mlp_lifecycle import MLPLifecycleManager, LocalDictBackend
from src.inference.user_state import UserState
from src.training.adapt_personal_mlp import TripletAdaptation

logger = logging.getLogger(__name__)


class SessionSimulator:
    """
    Simulates user sessions by replaying historical transactions.

    For each user interaction:
    1. Update EMA
    2. Trigger triplet adaptation (every N interactions)
    3. Generate recommendations
    4. Compare against ground truth
    """

    def __init__(
        self,
        lifecycle_manager: MLPLifecycleManager,
        clip_embeddings: Dict[str, torch.Tensor],
        adaptation: TripletAdaptation,
        trigger_every_n: int = 5,
        recommendation_k: int = 10,
        ground_truth_t: int = 12,
    ):
        """
        Args:
            lifecycle_manager: MLP lifecycle manager (with LocalDictBackend for simulation).
            clip_embeddings: CLIP embeddings for all articles.
            adaptation: TripletAdaptation instance.
            trigger_every_n: Trigger adaptation every N interactions.
            recommendation_k: Number of recommendations to generate.
            ground_truth_t: Number of future purchases as ground truth.
        """
        self.lifecycle = lifecycle_manager
        self.clip_embeddings = clip_embeddings
        self.adaptation = adaptation
        self.trigger_every_n = trigger_every_n
        self.recommendation_k = recommendation_k
        self.ground_truth_t = ground_truth_t

    def simulate_sessions(
        self,
        test_transactions: pd.DataFrame,
        cold_start_daily: bool = False,
    ) -> Dict[str, List[dict]]:
        """
        Simulate all user sessions from test transaction data.

        Args:
            test_transactions: DataFrame with [customer_id, article_id, t_dat].
            cold_start_daily: If True, reset user state daily.

        Returns:
            Dict of user_id -> list of per-interaction evaluation records.
        """
        results = defaultdict(list)

        # Group by user
        user_groups = test_transactions.sort_values("t_dat").groupby("customer_id")

        for user_id, user_txns in user_groups:
            user_results = self._simulate_user_session(
                str(user_id), user_txns, cold_start_daily
            )
            results[str(user_id)] = user_results

        return dict(results)

    def _simulate_user_session(
        self,
        user_id: str,
        transactions: pd.DataFrame,
        cold_start_daily: bool,
    ) -> List[dict]:
        """Simulate a single user's session."""
        records = []
        articles = transactions["article_id"].tolist()
        dates = transactions["t_dat"].tolist() if "t_dat" in transactions.columns else [None] * len(articles)

        current_date = None
        interaction_batch = []

        for i, (article_id, date) in enumerate(zip(articles, dates)):
            # Day-by-day cold start: reset at day boundary
            if cold_start_daily and date is not None:
                if current_date is not None and date != current_date:
                    self.lifecycle.delete_user(user_id)
                current_date = date

            if str(article_id) not in self.clip_embeddings:
                continue

            # Get or create user's Personal MLP
            entry = self.lifecycle.get_or_create(user_id)

            # Update EMA
            clip_emb = self.clip_embeddings[str(article_id)]
            with torch.no_grad():
                mlp_proj = entry.personal_mlp(clip_emb.unsqueeze(0)).squeeze(0)
            entry.user_state.update(mlp_proj)

            # Add to interaction batch
            interaction_batch.append(str(article_id))

            # Trigger adaptation every N interactions
            adapted = False
            if len(interaction_batch) >= self.trigger_every_n:
                adapted = self._trigger_adaptation(entry, interaction_batch)
                self.lifecycle.mark_dirty(user_id)
                self.lifecycle.flush(user_id)
                interaction_batch = []

            # Generate ground truth (next T articles)
            future_articles = [
                str(a) for a in articles[i + 1 : i + 1 + self.ground_truth_t]
            ]

            # Record for evaluation
            records.append({
                "interaction_idx": i,
                "article_id": str(article_id),
                "interaction_count": entry.user_state.interaction_count,
                "adapted": adapted,
                "ground_truth": future_articles,
            })

        return records

    def _trigger_adaptation(self, entry, interaction_batch: List[str]) -> bool:
        """Run triplet loss adaptation."""
        positives = [
            self.clip_embeddings[aid]
            for aid in interaction_batch
            if aid in self.clip_embeddings
        ]

        if not positives:
            return False

        # Random negatives (H&M has no click data for hard negatives)
        all_ids = list(self.clip_embeddings.keys())
        neg_ids = [
            aid for aid in all_ids
            if aid not in interaction_batch
        ][:len(positives)]

        negatives = [self.clip_embeddings[aid] for aid in neg_ids]

        if not negatives:
            return False

        self.adaptation.adapt(entry.personal_mlp, positives, negatives)
        return True

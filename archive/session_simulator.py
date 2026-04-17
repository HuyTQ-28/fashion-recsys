"""
Interaction Simulation Engine — ARCHIVED.

Moved from src/inference/session_simulator.py.
Used only by archive/evaluation/sensitivity.py for offline hyperparameter sweeps.
Not part of the live demo API.

Paper reference: Section 3
- Day-by-day cold start: delete user state at end of each day
- Multi-week personalization: keep state across days (1, 2, 3 weeks)
- Metrics: P@K, R@K, F1@K (K=10, ground truth T=12 next purchases)
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

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
        use_mlp_projection: bool = None,
    ):
        self.lifecycle = lifecycle_manager
        self.clip_embeddings = clip_embeddings
        self.adaptation = adaptation
        self.trigger_every_n = trigger_every_n
        self.recommendation_k = recommendation_k
        self.ground_truth_t = ground_truth_t

        if use_mlp_projection is None:
            sample = next(iter(clip_embeddings.values())) if clip_embeddings else None
            self.use_mlp_projection = (sample is not None and sample.shape[-1] == 512)
        else:
            self.use_mlp_projection = use_mlp_projection

    def simulate_sessions(
        self,
        test_transactions: pd.DataFrame,
        cold_start_daily: bool = False,
    ) -> Dict[str, List[dict]]:
        results = defaultdict(list)
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
        records = []
        articles = transactions["article_id"].tolist()
        dates = (
            transactions["t_dat"].tolist()
            if "t_dat" in transactions.columns
            else [None] * len(articles)
        )

        current_date = None
        interaction_batch = []

        for i, (article_id, date) in enumerate(zip(articles, dates)):
            if cold_start_daily and date is not None:
                if current_date is not None and date != current_date:
                    self.lifecycle.delete_user(user_id)
                current_date = date

            if str(article_id) not in self.clip_embeddings:
                continue

            entry = self.lifecycle.get_or_create(user_id)
            clip_emb = self.clip_embeddings[str(article_id)]

            with torch.no_grad():
                if self.use_mlp_projection:
                    mlp_proj = entry.personal_mlp(clip_emb.unsqueeze(0)).squeeze(0)
                else:
                    mlp_proj = clip_emb
            entry.user_state.update(mlp_proj)

            interaction_batch.append(str(article_id))

            adapted = False
            if len(interaction_batch) >= self.trigger_every_n:
                adapted = self._trigger_adaptation(entry, interaction_batch)
                self.lifecycle.mark_dirty(user_id)
                self.lifecycle.flush(user_id)
                interaction_batch = []

            future_articles = [
                str(a) for a in articles[i + 1: i + 1 + self.ground_truth_t]
            ]
            records.append({
                "interaction_idx": i,
                "article_id": str(article_id),
                "interaction_count": entry.user_state.interaction_count,
                "adapted": adapted,
                "ground_truth": future_articles,
            })

        return records

    def _trigger_adaptation(self, entry, interaction_batch: List[str]) -> bool:
        positives = [
            self.clip_embeddings[aid]
            for aid in interaction_batch
            if aid in self.clip_embeddings
        ]
        if not positives:
            return False

        all_ids = list(self.clip_embeddings.keys())
        neg_ids = [
            aid for aid in all_ids if aid not in interaction_batch
        ][: len(positives)]

        negatives = [self.clip_embeddings[aid] for aid in neg_ids]
        if not negatives:
            return False

        self.adaptation.adapt(entry.personal_mlp, positives, negatives)
        return True

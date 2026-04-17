import logging
from typing import List, Tuple
import torch

from src.data_access.redis.session_store import RedisSessionStore, InteractionEvent
from src.data_access.catalog.clip_ram_index import ClipRamIndex
from src.models.personal_mlp import PersonalMLP
from src.training.adapt_personal_mlp import TripletAdaptation
from src.core.personalization.state_rebuilder import recompute_u_personal_from_history

logger = logging.getLogger(__name__)

class AdaptationWorker:
    """
    Handles asynchronous SGD triplet updates and mandatory u_t_personal rebuilds.
    """
    def __init__(self, redis_store: RedisSessionStore, clip_index: ClipRamIndex, ttl_seconds: int = 14*24*3600):
        self.redis = redis_store
        self.clip_index = clip_index
        self.ttl_seconds = ttl_seconds
        self.adaptation = TripletAdaptation()

    def run_adaptation(self, session_id: str, shown_articles: List[str] = None):
        # A) Acquire per-session lock
        if not self.redis.try_lock(session_id, ttl_seconds=30):
            logger.info(f"Worker: Adaptation for {session_id} locked, skipping.")
            return

        try:
            # B) Load latest snapshot + bounded history
            snap = self.redis.read_snapshot(session_id)
            hist = self.redis.read_history(session_id)
            
            if snap is None or len(hist) == 0:
                logger.info(f"Worker: No snapshot or history for {session_id}.")
                return

            # C) Materialize training tensors from RAM clip index
            x_pos, x_neg = self._build_triplets_from_history(hist, shown_articles)
            if not x_pos or not x_neg:
                logger.info(f"Worker: Missing pos/neg samples for {session_id}.")
                return

            # D) Load personal model and run SGD triplet updates
            model = PersonalMLP.deserialize(snap.personal_weights)
            
            # adapt() returns (model, time_ms)
            _, adapt_time = self.adaptation.adapt(model, x_pos, x_neg)
            
            # E) CRUCIAL INVALIDATION RULE
            # old u_t_personal is invalid after weight change
            # recompute from scratch using recent history and NEW model
            new_u_personal = recompute_u_personal_from_history(
                model=model,
                history=hist,
                alpha=snap.alpha_personal,
                clip_lookup=self.clip_index,
            )

            # F) Atomically commit new weights + new u_t_personal + increment model_version
            self.redis.commit_model_and_state(
                session_id=session_id,
                new_weights_b64=model.serialize(),
                new_u_t_personal=new_u_personal,
                model_version=snap.model_version + 1,
                ttl_seconds=self.ttl_seconds
            )
            logger.info(f"Worker: Adaptation finished for {session_id} in {adapt_time:.2f}ms.")

        except Exception as e:
            logger.error(f"Worker: Error during adaptation for {session_id}: {e}")
        finally:
            self.redis.unlock(session_id)

    def _build_triplets_from_history(self, hist: List[InteractionEvent], shown_articles: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Take recent interactions as positives
        # Simple heuristic: last 10 interactions
        pos_events = hist[-10:]
        positives = [self.clip_index.get(e.article_id) for e in pos_events]
        
        # Negatives: shown but not interacted
        interacted_ids = {e.article_id for e in hist}
        
        if shown_articles:
            neg_ids = [aid for aid in shown_articles if aid not in interacted_ids]
        else:
            neg_ids = []
            
        # Fallback to catalog negatives if needed
        if not neg_ids:
            # We could sample randomly from clip_index, but for now we take first N keys not in history
            catalog_keys = list(self.clip_index.clip_dict.keys())
            neg_ids = [k for k in catalog_keys if k not in interacted_ids][:len(positives)]
            
        negatives = [self.clip_index.get(aid) for aid in neg_ids]
        
        return positives, negatives

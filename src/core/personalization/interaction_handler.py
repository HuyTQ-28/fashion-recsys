import time
import logging
from typing import Dict, List, Optional
import torch

from src.data_access.redis.session_store import RedisSessionStore, InteractionEvent
from src.data_access.catalog.clip_ram_index import ClipRamIndex
from src.models.student_mlp import StudentMLP
from src.models.personal_mlp import PersonalMLP, PersonalMLPFactory
from src.core.personalization.ema import update_ema_vector

logger = logging.getLogger(__name__)

class InteractionHandler:
    def __init__(
        self, 
        redis_store: RedisSessionStore, 
        clip_index: ClipRamIndex, 
        base_mlp: StudentMLP,
        personal_mlp_factory: PersonalMLPFactory,
        trigger_every_n: int = 9,
        ttl_seconds: int = 14*24*3600
    ):
        self.redis = redis_store
        self.clip_index = clip_index
        self.base_mlp = base_mlp
        self.factory = personal_mlp_factory
        self.trigger_every_n = trigger_every_n
        self.ttl_seconds = ttl_seconds

    def handle(self, session_id: str, article_id: str, interaction_type: str) -> dict:
        x512 = self.clip_index.get(article_id)
        if x512 is None or torch.all(x512 == 0):
            return {"status": "skipped", "interaction_count": 0, "needs_adaptation": False}
            
        snap = self.redis.read_snapshot(session_id)
        
        # If new session, initialize
        if snap is None:
            new_model = self.factory.create(session_id)
            self.redis.initialize_session(
                session_id=session_id,
                initial_weights_b64=new_model.serialize(),
                alpha=0.7,
                ttl_seconds=self.ttl_seconds
            )
            snap = self.redis.read_snapshot(session_id)
            
        alpha = snap.alpha_personal

        # 2) Update u_t_base in global space
        with torch.no_grad():
            x64_base = self.base_mlp(x512.unsqueeze(0)).squeeze(0)
        u_t_base = update_ema_vector(snap.u_t_base, x64_base, alpha)

        # 3) Update u_t_personal in personal space
        personal_model = PersonalMLP.deserialize(snap.personal_weights)
        personal_model.eval()
        with torch.no_grad():
            x64_personal = personal_model(x512.unsqueeze(0)).squeeze(0)
        u_t_personal = update_ema_vector(snap.u_t_personal, x64_personal, alpha)
        
        # 1) Append event
        weight = 4 if interaction_type == "purchase" else 1
        event = InteractionEvent(
            article_id=article_id,
            interaction_type=interaction_type,
            weight=weight,
            ts=time.time()
        )
        
        # Persist state
        self.redis.append_interaction_and_update_state(
            session_id=session_id,
            event=event,
            u_t_base=u_t_base,
            u_t_personal=u_t_personal,
            ttl_seconds=self.ttl_seconds
        )
        
        hist = self.redis.read_history(session_id)
        # A simple condition: if history length % trigger_every_n == 0
        needs_adaptation = len(hist) > 0 and len(hist) % self.trigger_every_n == 0
        
        return {
            "status": "ok",
            "interaction_count": len(hist),
            "needs_adaptation": needs_adaptation
        }
        
    def get_user_state(self, session_id: str) -> dict:
        snap = self.redis.read_snapshot(session_id)
        if not snap:
            return {
                "user_id": session_id,
                "ema_vector": None,
                "interaction_count": 0,
                "has_personalization": False
            }
            
        hist = self.redis.read_history(session_id)
        return {
            "user_id": session_id,
            "ema_vector": snap.u_t_personal,
            "interaction_count": len(hist),
            "has_personalization": snap.u_t_personal is not None
        }

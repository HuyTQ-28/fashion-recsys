import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import redis

logger = logging.getLogger(__name__)

@dataclass
class SessionSnapshot:
    u_t_base: List[float]
    u_t_personal: List[float]
    personal_weights: str  # base64 encoded state_dict
    model_version: int
    state_version: int
    updated_at: float
    alpha_personal: float

@dataclass
class InteractionEvent:
    article_id: str
    interaction_type: str
    weight: int
    ts: float

class RedisSessionStore:
    def __init__(self, redis_url: str, prefix: str = "rec"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        logger.info(f"Connected to RedisSessionStore at {redis_url} with prefix {prefix}")

    def _state_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:state"

    def _model_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:model"

    def _history_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:history"
        
    def _lock_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:lock"

    def read_snapshot(self, session_id: str) -> Optional[SessionSnapshot]:
        try:
            with self.redis.pipeline(transaction=False) as pipe:
                pipe.get(self._state_key(session_id))
                pipe.get(self._model_key(session_id))
                raw_state, raw_model = pipe.execute()
                
            if raw_state is None or raw_model is None:
                return None
                
            state_doc = json.loads(raw_state)
            
            return SessionSnapshot(
                u_t_base=state_doc.get("u_t_base"),
                u_t_personal=state_doc.get("u_t_personal"),
                personal_weights=raw_model,
                model_version=state_doc.get("model_version", 0),
                state_version=state_doc.get("state_version", 0),
                updated_at=state_doc.get("updated_at", time.time()),
                alpha_personal=state_doc.get("alpha_personal", 0.7),
            )
        except Exception as e:
            logger.error(f"Failed to read snapshot for session {session_id}: {e}")
            return None

    def read_history(self, session_id: str) -> List[InteractionEvent]:
        try:
            raw_history = self.redis.lrange(self._history_key(session_id), 0, -1)
            events = []
            for item in raw_history:
                data = json.loads(item)
                events.append(InteractionEvent(
                    article_id=data["article_id"],
                    interaction_type=data["interaction_type"],
                    weight=data.get("weight", 1),
                    ts=data.get("ts", time.time())
                ))
            return events
        except Exception as e:
            logger.error(f"Failed to read history for session {session_id}: {e}")
            return []

    def append_interaction_and_update_state(
        self, 
        session_id: str, 
        event: InteractionEvent, 
        u_t_base: List[float], 
        u_t_personal: List[float],
        ttl_seconds: int,
        max_history: int = 200
    ) -> None:
        """Atomically append to history and update state vectors."""
        try:
            # We must read current state to get version and other fields, 
            # but for simplicity we will just hset or set the state document.
            # A more robust approach uses Lua script to ensure atomicity and increment state_version.
            
            # For simplicity, we just use a pipeline
            raw_state = self.redis.get(self._state_key(session_id))
            if raw_state:
                state_doc = json.loads(raw_state)
            else:
                state_doc = {"model_version": 0, "state_version": 0, "alpha_personal": 0.7}
            
            state_doc["u_t_base"] = u_t_base
            state_doc["u_t_personal"] = u_t_personal
            state_doc["state_version"] += 1
            state_doc["updated_at"] = time.time()
            
            event_json = json.dumps({
                "article_id": event.article_id,
                "interaction_type": event.interaction_type,
                "weight": event.weight,
                "ts": event.ts
            })
            
            with self.redis.pipeline(transaction=True) as pipe:
                pipe.rpush(self._history_key(session_id), event_json)
                pipe.ltrim(self._history_key(session_id), -max_history, -1)
                pipe.expire(self._history_key(session_id), ttl_seconds)
                
                pipe.set(self._state_key(session_id), json.dumps(state_doc), ex=ttl_seconds)
                # Model TTL is also refreshed
                pipe.expire(self._model_key(session_id), ttl_seconds)
                pipe.execute()
        except Exception as e:
            logger.error(f"Failed to append interaction for session {session_id}: {e}")

    def commit_model_and_state(
        self, 
        session_id: str, 
        new_weights_b64: str, 
        new_u_t_personal: List[float], 
        model_version: int,
        ttl_seconds: int
    ) -> None:
        """Atomically commit new personal model weights and its recalculated personal space vector."""
        try:
            raw_state = self.redis.get(self._state_key(session_id))
            if raw_state:
                state_doc = json.loads(raw_state)
            else:
                state_doc = {"state_version": 0, "alpha_personal": 0.7, "u_t_base": None}
                
            state_doc["u_t_personal"] = new_u_t_personal
            state_doc["model_version"] = model_version
            state_doc["updated_at"] = time.time()
            
            with self.redis.pipeline(transaction=True) as pipe:
                pipe.set(self._model_key(session_id), new_weights_b64, ex=ttl_seconds)
                pipe.set(self._state_key(session_id), json.dumps(state_doc), ex=ttl_seconds)
                pipe.expire(self._history_key(session_id), ttl_seconds)
                pipe.execute()
        except Exception as e:
            logger.error(f"Failed to commit model for session {session_id}: {e}")

    def initialize_session(self, session_id: str, initial_weights_b64: str, alpha: float, ttl_seconds: int) -> None:
        """Initialize a new session if it doesn't exist."""
        try:
            if not self.redis.exists(self._state_key(session_id)):
                state_doc = {
                    "u_t_base": None,
                    "u_t_personal": None,
                    "model_version": 0,
                    "state_version": 0,
                    "updated_at": time.time(),
                    "alpha_personal": alpha
                }
                with self.redis.pipeline(transaction=True) as pipe:
                    pipe.set(self._state_key(session_id), json.dumps(state_doc), ex=ttl_seconds)
                    pipe.set(self._model_key(session_id), initial_weights_b64, ex=ttl_seconds)
                    pipe.execute()
        except Exception as e:
            logger.error(f"Failed to initialize session {session_id}: {e}")

    def try_lock(self, session_id: str, ttl_seconds: int = 30) -> bool:
        try:
            return bool(self.redis.set(self._lock_key(session_id), "1", nx=True, ex=ttl_seconds))
        except Exception as e:
            logger.error(f"Failed to acquire lock for session {session_id}: {e}")
            return False

    def unlock(self, session_id: str) -> None:
        try:
            self.redis.delete(self._lock_key(session_id))
        except Exception as e:
            logger.error(f"Failed to release lock for session {session_id}: {e}")

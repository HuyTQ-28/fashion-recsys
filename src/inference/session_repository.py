import json
import logging
from dataclasses import dataclass
from typing import Optional, Protocol
import time

import redis
from src.inference.user_state import UserState
from src.models.personal_mlp import PersonalMLP

logger = logging.getLogger(__name__)

@dataclass
class SessionBundle:
    user_state: UserState
    personal_mlp: PersonalMLP
    adaptation_batch: list[str]

class SessionRepository(Protocol):
    def load(self, session_id: str) -> Optional[SessionBundle]: ...
    def save(self, session_id: str, bundle: SessionBundle, ttl_seconds: int) -> None: ...
    def refresh(self, session_id: str, ttl_seconds: int) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def try_acquire_lock(self, session_id: str, ttl_seconds: int) -> bool: ...
    def release_lock(self, session_id: str) -> None: ...


class RedisSessionRepository:
    def __init__(self, redis_url: str, prefix: str = "rec"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        logger.info(f"Connected to RedisSessionRepository at {redis_url} with prefix {prefix}")

    def _state_key(self, session_id: str) -> str:
        return f"{self.prefix}:state:{session_id}"

    def _model_key(self, session_id: str) -> str:
        return f"{self.prefix}:model:{session_id}"
        
    def _lock_key(self, session_id: str) -> str:
        return f"{self.prefix}:lock:{session_id}"

    def save(self, session_id: str, bundle: SessionBundle, ttl_seconds: int) -> None:
        state_payload = bundle.user_state.to_dict()
        state_payload["adaptation_batch"] = bundle.adaptation_batch
        
        # We need to save model without decoding responses since it's base64/bytes-like string from PersonalMLP
        # but decode_responses=True is on. serialize() returns a string.
        model_payload = bundle.personal_mlp.serialize()

        try:
            with self.redis.pipeline(transaction=True) as pipe:
                pipe.set(self._state_key(session_id), json.dumps(state_payload), ex=ttl_seconds)
                pipe.set(self._model_key(session_id), model_payload, ex=ttl_seconds)
                pipe.execute()
        except Exception as e:
            logger.error(f"Failed to save session {session_id} to Redis: {e}")

    def load(self, session_id: str) -> Optional[SessionBundle]:
        try:
            with self.redis.pipeline(transaction=False) as pipe:
                pipe.get(self._state_key(session_id))
                pipe.get(self._model_key(session_id))
                raw_state, raw_model = pipe.execute()
                
            if raw_state is None or raw_model is None:
                return None
                
            state_doc = json.loads(raw_state)
            user_state = UserState.from_dict(state_doc)
            personal_mlp = PersonalMLP.deserialize(raw_model)
            
            return SessionBundle(
                user_state=user_state,
                personal_mlp=personal_mlp,
                adaptation_batch=state_doc.get("adaptation_batch", []),
            )
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from Redis: {e}")
            return None

    def refresh(self, session_id: str, ttl_seconds: int) -> None:
        try:
            with self.redis.pipeline(transaction=False) as pipe:
                pipe.expire(self._state_key(session_id), ttl_seconds)
                pipe.expire(self._model_key(session_id), ttl_seconds)
                pipe.execute()
        except Exception as e:
            logger.error(f"Failed to refresh TTL for session {session_id}: {e}")

    def delete(self, session_id: str) -> None:
        try:
            self.redis.delete(self._state_key(session_id), self._model_key(session_id))
        except Exception as e:
            logger.error(f"Failed to delete session {session_id} from Redis: {e}")

    def try_acquire_lock(self, session_id: str, ttl_seconds: int = 30) -> bool:
        """Acquire a lock for async adaptation to prevent concurrent updates."""
        try:
            return bool(self.redis.set(self._lock_key(session_id), "1", nx=True, ex=ttl_seconds))
        except Exception as e:
            logger.error(f"Failed to acquire lock for session {session_id}: {e}")
            return False

    def release_lock(self, session_id: str) -> None:
        """Release the async adaptation lock."""
        try:
            self.redis.delete(self._lock_key(session_id))
        except Exception as e:
            logger.error(f"Failed to release lock for session {session_id}: {e}")

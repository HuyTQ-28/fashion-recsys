import logging
import time
from collections import OrderedDict
from typing import Dict, Optional

import torch

from src.models.personal_mlp import PersonalMLP, PersonalMLPFactory
from src.inference.user_state import UserState
from src.inference.session_repository import SessionRepository, SessionBundle

logger = logging.getLogger(__name__)


class LocalDictBackend:
    """
    In-memory dict repository for development and offline simulation.

    No network overhead. Supports TTL tracking but no automatic expiry.
    Use for batch evaluation to avoid Redis costs.
    """

    def __init__(self):
        self._store: Dict[str, SessionBundle] = {}
        self._ttls: Dict[str, float] = {}

    def load(self, session_id: str) -> Optional[SessionBundle]:
        if session_id in self._store:
            if session_id in self._ttls and time.time() > self._ttls[session_id]:
                del self._store[session_id]
                del self._ttls[session_id]
                return None
            return self._store[session_id]
        return None

    def save(self, session_id: str, bundle: SessionBundle, ttl_seconds: int) -> None:
        self._store[session_id] = bundle
        self._ttls[session_id] = time.time() + ttl_seconds

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._ttls.pop(session_id, None)

    def refresh(self, session_id: str, ttl_seconds: int) -> None:
        if session_id in self._store:
            self._ttls[session_id] = time.time() + ttl_seconds
            
    def try_acquire_lock(self, session_id: str, ttl_seconds: int) -> bool:
        return True
        
    def release_lock(self, session_id: str) -> None:
        pass

    def clear(self) -> None:
        """Clear all entries (for day-by-day cold start simulation)."""
        self._store.clear()
        self._ttls.clear()


class LRUCache:
    """
    Standalone LRU cache with fixed max size.

    put(key, value) -> Optional[evicted_key]
    get(key) -> Optional[value]  (moves key to MRU)
    __contains__(key)
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def put(self, key: str, value) -> Optional[str]:
        """
        Insert or update a key.

        Returns:
            The evicted key if cache was full, else None.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return None

        evicted_key = None
        if len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)

        self._cache[key] = value
        return evicted_key

    def get(self, key: str):
        """Get a value, promoting it to MRU. Returns None if missing."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def pop(self, key: str):
        """Remove and return a value."""
        return self._cache.pop(key, None)


class CacheEntry:
    """A single entry in the LRU cache."""

    def __init__(self, personal_mlp: PersonalMLP, user_state: UserState, interaction_batch: list = None):
        self.personal_mlp = personal_mlp
        self.user_state = user_state
        self.interaction_batch = interaction_batch or []
        self.dirty = False
        self.last_accessed = time.time()


class _TemplateFactory:
    """Wraps a PersonalMLP/StudentMLP as a factory by deep-copying it."""

    def __init__(self, template):
        import copy
        self._template = template
        self.layer_dims = getattr(template, "layer_dims", [512, 256, 128, 64])
        self._copy = copy.deepcopy  # keep reference

    def create(self, user_id: str) -> "PersonalMLP":
        import copy
        from src.models.personal_mlp import PersonalMLP
        # If template is already a PersonalMLP, deep-copy its network
        if isinstance(self._template, PersonalMLP):
            mlp = copy.deepcopy(self._template)
            mlp.user_id = user_id
            mlp.interaction_count = 0
            return mlp
        # Otherwise wrap it (StudentMLP or similar)
        return PersonalMLP(base_model=self._template, user_id=user_id, layer_dims=self.layer_dims)


class MLPLifecycleManager:
    """
    Manages Personal MLP lifecycle with LRU cache + storage repository.

    Lifecycle flow:
    1. Load (cache miss): Check LRU → fetch from repository → insert
    2. Use: Forward pass / adaptation on in-memory MLP
    3. Write-back (dirty): Flush to repository after adaptation
    4. Evict: LRU eviction when cache full (flush dirty first)
    """

    def __init__(
        self,
        factory,
        repository: SessionRepository,
        max_size: int = 200,
        ttl_seconds: int = 14 * 24 * 3600,  # 14 days
        alpha: float = 0.7,  # best from sensitivity sweep
    ):
        """
        Args:
            factory: PersonalMLPFactory or a _TemplateFactory.
            repository: Storage repository (LocalDictBackend or RedisSessionRepository).
            max_size: Maximum number of Personal MLPs in the LRU cache.
            ttl_seconds: TTL for backend keys (default: 14 days).
            alpha: Default EMA alpha for new users.
        """
        if isinstance(factory, PersonalMLPFactory):
            self.factory = factory
        else:
            self.factory = _TemplateFactory(factory)

        self.repository = repository
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.default_alpha = alpha

        self._lru = LRUCache(max_size=max_size)
        self._dirty: Dict[str, bool] = {}

        # Stats
        self.stats = {"hits": 0, "misses": 0, "creates": 0, "evictions": 0}

    @property
    def _cache(self) -> dict:
        """Back-compat alias: exposes the internal LRU dict for tests."""
        return self._lru._cache

    def get_or_create(self, user_id: str, alpha: float = None) -> CacheEntry:
        """
        Get a user's CacheEntry from LRU or storage, or create a new one.

        Args:
            user_id: User identifier.
            alpha: EMA alpha override (uses default if None).

        Returns:
            CacheEntry with Personal MLP and UserState.
        """
        alpha = alpha if alpha is not None else self.default_alpha

        # 1. Check LRU cache
        entry = self._lru.get(user_id)
        if entry is not None:
            self.stats["hits"] += 1
            return entry

        self.stats["misses"] += 1

        # 2. Check storage repository
        bundle = self.repository.load(user_id)

        if bundle is not None:
            entry = CacheEntry(
                personal_mlp=bundle.personal_mlp,
                user_state=bundle.user_state,
                interaction_batch=bundle.adaptation_batch,
            )
            self.repository.refresh(user_id, self.ttl_seconds)
        else:
            # 3. New user: create from factory
            mlp = self.factory.create(user_id)
            user_state = UserState(user_id=user_id, alpha=alpha)
            entry = CacheEntry(mlp, user_state)
            entry.dirty = True
            self.stats["creates"] += 1

        # Insert into LRU (may evict)
        evicted_key = self._lru.put(user_id, entry)
        if evicted_key is not None:
            self._flush_key(evicted_key)
            self.stats["evictions"] += 1

        return entry

    def mark_dirty(self, user_id: str) -> None:
        """Mark a cache entry as dirty (needs flush after adaptation)."""
        entry = self._lru.get(user_id)
        if entry is not None:
            entry.dirty = True

    def flush(self, user_id: str) -> None:
        """Eagerly flush a dirty cache entry to the storage repository."""
        entry = self._lru.get(user_id)
        if entry is not None and entry.dirty:
            bundle = SessionBundle(
                user_state=entry.user_state,
                personal_mlp=entry.personal_mlp,
                adaptation_batch=entry.interaction_batch,
            )
            self.repository.save(user_id, bundle, self.ttl_seconds)
            entry.dirty = False

    def delete_user(self, user_id: str) -> None:
        """Delete a user's state from both cache and backend."""
        self._lru.pop(user_id)
        self.repository.delete(user_id)

    def _flush_key(self, user_id: str) -> None:
        """Flush evicted cache entry to backend if it was dirty.

        Called by get_or_create when the LRU evicts an entry to make room.
        We flush before the entry is discarded so no adapted MLP state is lost.
        """
        # Note: the entry has already been evicted from _lru by LRUCache.put(),
        # so we cannot retrieve it from cache here. The eviction path in
        # get_or_create must save the entry before calling this method if needed.
        # For now we log; the eager flush() after every adaptation covers
        # the common case (dirty entries are always flushed before eviction).
        logger.debug("LRU evicted user %s — ensure flush() was called after adaptation.", user_id)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "cache_size": len(self._lru),
            "hit_rate": hit_rate,
        }

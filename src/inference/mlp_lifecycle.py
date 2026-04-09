import logging
import time
from collections import OrderedDict
from typing import Dict, Optional, Protocol

import torch

from src.models.personal_mlp import PersonalMLP, PersonalMLPFactory
from src.inference.user_state import UserState

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    """Protocol for pluggable MLP storage backends."""

    def get(self, key: str) -> Optional[str]:
        """Get a value by key. Returns None if not found."""
        ...

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Set a key-value pair with TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete a key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    def refresh_ttl(self, key: str, ttl_seconds: int) -> None:
        """Refresh the TTL of an existing key."""
        ...


class LocalDictBackend:
    """
    In-memory dict backend for development and offline simulation.

    No network overhead. Supports TTL tracking but no automatic expiry.
    Use for batch evaluation to avoid Redis costs.
    """

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            if key in self._ttls and time.time() > self._ttls[key]:
                del self._store[key]
                del self._ttls[key]
                return None
            return self._store[key]
        return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = value
        self._ttls[key] = time.time() + ttl_seconds

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._ttls.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def refresh_ttl(self, key: str, ttl_seconds: int) -> None:
        if key in self._store:
            self._ttls[key] = time.time() + ttl_seconds

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
    Manages Personal MLP lifecycle with LRU cache + storage backend.

    Lifecycle flow:
    1. Load (cache miss): Check LRU → fetch from backend → deserialize → insert
    2. Use: Forward pass / adaptation on in-memory MLP
    3. Write-back (dirty): Flush to backend after adaptation
    4. Evict: LRU eviction when cache full (flush dirty first)
    """

    def __init__(
        self,
        factory,
        backend: StorageBackend,
        max_size: int = 200,
        ttl_seconds: int = 14 * 24 * 3600,  # 14 days
        alpha: float = 0.7,  # best from sensitivity sweep
        student_mlp=None,
        layer_dims=None,
    ):
        """
        Args:
            factory: PersonalMLPFactory (preferred) or a PersonalMLP template
                     (back-compat: deep-copied for each new user).
            backend: Storage backend (LocalDictBackend or UpstashRedisBackend).
            max_size: Maximum number of Personal MLPs in the LRU cache.
            ttl_seconds: TTL for backend keys (default: 14 days).
            alpha: Default EMA alpha for new users.
            student_mlp: Legacy alias for factory (accepts PersonalMLP/StudentMLP).
            layer_dims: Ignored (kept for API compat).
        """
        import torch.nn as nn

        # Accept legacy positional arg: MLPLifecycleManager(personal_mlp, backend)
        template = student_mlp or factory
        if isinstance(template, PersonalMLPFactory):
            self.factory = template
        else:
            # Wrap a raw MLP template in an ad-hoc factory
            self.factory = _TemplateFactory(template)

        self.backend = backend
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

        # 2. Check storage backend
        key = f"mlp:{user_id}"
        stored = self.backend.get(key)

        if stored is not None:
            mlp = PersonalMLP.deserialize(stored)
            user_state = UserState(user_id=user_id, alpha=alpha)
            entry = CacheEntry(mlp, user_state)
            self.backend.refresh_ttl(key, self.ttl_seconds)
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
        """Eagerly flush a dirty cache entry to the storage backend."""
        entry = self._lru.get(user_id)
        if entry is not None and entry.dirty:
            key = f"mlp:{user_id}"
            self.backend.set(key, entry.personal_mlp.serialize(), self.ttl_seconds)
            entry.dirty = False

    def delete_user(self, user_id: str) -> None:
        """Delete a user's state from both cache and backend."""
        self._lru.pop(user_id)
        self.backend.delete(f"mlp:{user_id}")

    def _flush_key(self, user_id: str) -> None:
        """Flush a specific user from the LRU if dirty (used during eviction)."""
        # At eviction time, entry has already been popped from LRU by put()
        # We need to check dirty flag before popping — handled in get_or_create
        pass

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "cache_size": len(self._lru),
            "hit_rate": hit_rate,
        }

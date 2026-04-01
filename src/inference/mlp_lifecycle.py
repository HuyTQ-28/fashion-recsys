"""
MLP Lifecycle Manager with Pluggable Storage Backends.

Owner: Member 2 (Personalization Engine)
Redis Backend: Member 3 (Search & Infrastructure)

Manages Personal MLP weights with a two-tier caching strategy:
- Hot tier: LRU in-memory cache (per-container, ~200 users, ~140MB)
- Cold tier: Pluggable StorageBackend (Redis for prod, LocalDict for test)
"""

import logging
import time
from collections import OrderedDict
from typing import Dict, Optional, Protocol

import torch

from src.models.personal_mlp import PersonalMLP, serialize_mlp, deserialize_mlp
from src.inference.user_state import UserState

logger = logging.getLogger(__name__)


# ============================================================
# Storage Backend Interface (M2 defines, M3 implements Redis)
# ============================================================

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
    Use for batch evaluation (Step 2.5) to avoid Redis costs.
    """

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            # Check TTL
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


# ============================================================
# LRU Cache Entry
# ============================================================

class CacheEntry:
    """A single entry in the LRU cache."""

    def __init__(self, personal_mlp: PersonalMLP, user_state: UserState, interaction_batch: list = None):
        self.personal_mlp = personal_mlp
        self.user_state = user_state
        self.interaction_batch = interaction_batch or []
        self.dirty = False
        self.last_accessed = time.time()


# ============================================================
# MLP Lifecycle Manager
# ============================================================

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
        student_mlp: PersonalMLP,
        backend: StorageBackend,
        max_size: int = 200,
        ttl_seconds: int = 14 * 24 * 3600,  # 14 days
        layer_dims: list = None,
    ):
        """
        Args:
            student_mlp: Trained Student MLP as template for new users.
            backend: Storage backend (LocalDictBackend or UpstashRedisBackend).
            max_size: Maximum number of Personal MLPs in the LRU cache.
            ttl_seconds: TTL for Redis keys (default: 14 days).
            layer_dims: MLP layer dimensions for deserialization.
        """
        self.student_mlp = student_mlp
        self.backend = backend
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.layer_dims = layer_dims or [512, 256, 128, 64]

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Stats
        self.stats = {"hits": 0, "misses": 0, "creates": 0, "evictions": 0}

    def get_or_create(self, user_id: str, alpha: float = 0.5) -> CacheEntry:
        """
        Get a user's Personal MLP from cache or storage, or create a new one.

        Args:
            user_id: User identifier.
            alpha: EMA alpha for new user states.

        Returns:
            CacheEntry with Personal MLP and UserState.
        """
        # 1. Check LRU cache
        if user_id in self._cache:
            self._cache.move_to_end(user_id)
            self.stats["hits"] += 1
            return self._cache[user_id]

        self.stats["misses"] += 1

        # 2. Check storage backend
        key = f"mlp:{user_id}"
        stored = self.backend.get(key)

        if stored is not None:
            # Deserialize
            mlp = deserialize_mlp(stored, self.layer_dims)
            user_state = UserState(alpha=alpha)
            entry = CacheEntry(mlp, user_state)

            # Refresh TTL (touch-on-access)
            self.backend.refresh_ttl(key, self.ttl_seconds)
        else:
            # 3. New user: deep copy from Student MLP template
            mlp = PersonalMLP(base_mlp=self.student_mlp)
            user_state = UserState(alpha=alpha)
            entry = CacheEntry(mlp, user_state)
            entry.dirty = True  # Will be flushed to backend
            self.stats["creates"] += 1

        # Insert into cache (may trigger eviction)
        self._insert(user_id, entry)

        return entry

    def mark_dirty(self, user_id: str) -> None:
        """Mark a cache entry as dirty (needs flush after adaptation)."""
        if user_id in self._cache:
            self._cache[user_id].dirty = True

    def flush(self, user_id: str) -> None:
        """
        Eagerly flush a dirty cache entry to the storage backend.

        Sets the key with TTL refresh.
        """
        if user_id in self._cache and self._cache[user_id].dirty:
            entry = self._cache[user_id]
            key = f"mlp:{user_id}"
            serialized = serialize_mlp(entry.personal_mlp)
            self.backend.set(key, serialized, self.ttl_seconds)
            entry.dirty = False

    def delete_user(self, user_id: str) -> None:
        """Delete a user's state from both cache and backend."""
        self._cache.pop(user_id, None)
        self.backend.delete(f"mlp:{user_id}")

    def _insert(self, user_id: str, entry: CacheEntry) -> None:
        """Insert into cache with LRU eviction if needed."""
        if len(self._cache) >= self.max_size:
            self._evict()

        self._cache[user_id] = entry
        self._cache.move_to_end(user_id)

    def _evict(self) -> None:
        """Evict the least-recently-used entry."""
        if not self._cache:
            return

        evicted_user, evicted_entry = self._cache.popitem(last=False)

        # Flush dirty entries before eviction
        if evicted_entry.dirty:
            key = f"mlp:{evicted_user}"
            serialized = serialize_mlp(evicted_entry.personal_mlp)
            self.backend.set(key, serialized, self.ttl_seconds)

        self.stats["evictions"] += 1

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "hit_rate": hit_rate,
        }

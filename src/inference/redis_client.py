"""
Upstash Redis Backend for MLP Lifecycle Manager.

Owner: Member 3 (Search & Infrastructure)

Implements the StorageBackend protocol defined in mlp_lifecycle.py.
Uses the upstash-redis Python package (REST-based, no persistent TCP connections).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class UpstashRedisBackend:
    """
    Upstash Redis storage backend for Personal MLP weights.

    REST-based access (no persistent TCP connections) — ideal for serverless.
    Keys pattern: mlp:{user_id} -> base64-encoded serialized MLP state_dict.
    TTL: configurable, default 14 days.

    Free tier: 1MB max value (~700KB MLP fits), 10K commands/day.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize Upstash Redis client.

        Args:
            url: Upstash Redis REST URL. Falls back to UPSTASH_REDIS_REST_URL env var.
            token: Upstash Redis REST token. Falls back to UPSTASH_REDIS_REST_TOKEN env var.
        """
        from upstash_redis import Redis

        self.url = url or os.environ.get("UPSTASH_REDIS_REST_URL")
        self.token = token or os.environ.get("UPSTASH_REDIS_REST_TOKEN")

        if not self.url or not self.token:
            raise ValueError(
                "Upstash Redis credentials required. Set UPSTASH_REDIS_REST_URL "
                "and UPSTASH_REDIS_REST_TOKEN environment variables."
            )

        self.redis = Redis(url=self.url, token=self.token)
        logger.info("Connected to Upstash Redis")

    def get(self, key: str) -> Optional[str]:
        """Get value by key. Returns None if not found."""
        value = self.redis.get(key)
        return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Set key with value and TTL."""
        self.redis.set(key, value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        """Delete a key."""
        self.redis.delete(key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(self.redis.exists(key))

    def refresh_ttl(self, key: str, ttl_seconds: int) -> None:
        """Refresh TTL on existing key (touch-on-access)."""
        self.redis.expire(key, ttl_seconds)

    def get_ttl(self, key: str) -> int:
        """Get remaining TTL in seconds. Returns -2 if key doesn't exist."""
        return self.redis.ttl(key)

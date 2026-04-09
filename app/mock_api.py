"""Stub API implementation that mirrors the backend contracts.

This module lets Member 4 build and demo the full UI before backend endpoints
are fully wired.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _seed_from_text(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _tokenize(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [token.strip().lower() for token in value.split() if token.strip()]


class StubRecommendationAPI:
    """In-memory recommendation API used for UI development and demos."""

    def __init__(
        self,
        catalog_size: Optional[int] = None,
        alpha: float = 0.7,
        adapt_every: int = 3,
    ):
        self._alpha = alpha
        self._adapt_every = adapt_every
        self._ttl_seconds = 14 * 24 * 3600
        self._project_root = Path(__file__).resolve().parents[1]
        self._data_root = self._project_root / "data"
        self._articles_csv = self._data_root / "articles.csv"
        self._images_root = self._data_root / "images"
        self._fallback_image_path = self._discover_fallback_image_path()
        self._catalog = self._build_catalog(catalog_size)
        self._index = {item["article_id"]: item for item in self._catalog}
        self._users: Dict[str, Dict] = {}

    def search(
        self,
        query: Optional[str] = None,
        image_b64: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        alpha: float = 0.7,
        mode: str = "hybrid",
        limit: int = 20,
    ) -> Dict:
        """Return contract-compatible search results."""
        terms = _tokenize(query)
        use_image = bool(image_b64)
        limit = max(1, min(limit, 100))
        filters = filters or {}

        scored = []
        for item in self._catalog:
            if not self._passes_filters(item, filters):
                continue

            keyword_score = self._keyword_score(item, terms)
            semantic_score = self._semantic_score(item["article_id"], query, use_image)

            if mode == "keyword":
                score = keyword_score
            elif mode == "semantic":
                score = semantic_score
            else:
                score = alpha * semantic_score + (1.0 - alpha) * keyword_score

            score += self._personal_boost(user_id, item)
            score += self._stable_noise(f"search:{item['article_id']}", 0.0, 0.02)
            scored.append((score, item))

        scored.sort(key=lambda row: row[0], reverse=True)
        top = scored[:limit]

        results = [self._to_search_item(item, score) for score, item in top]
        personalized = bool(
            user_id
            and user_id in self._users
            and self._users[user_id]["interaction_count"] > 0
        )

        return {
            "results": results,
            "query_mode": mode,
            "total_results": len(scored),
            "personalized": personalized,
        }

    def recommend(self, article_id: str, user_id: Optional[str] = None, k: int = 10) -> Dict:
        """Return contract-compatible recommendation results."""
        k = max(1, min(k, 50))
        seed = self._index.get(article_id)

        if seed is None:
            return {
                "recommendations": [],
                "seed_article_id": article_id,
                "personalized": False,
                "user_interaction_count": 0,
            }

        scored = []
        for item in self._catalog:
            if item["article_id"] == article_id:
                continue

            score = self._similarity(seed, item)
            score += self._personal_boost(user_id, item)
            score += self._stable_noise(f"rec:{article_id}:{item['article_id']}", 0.0, 0.015)
            scored.append((score, item))

        scored.sort(key=lambda row: row[0], reverse=True)
        top = scored[:k]

        recommendations = [
            self._to_recommendation_item(item, score) for score, item in top
        ]

        interaction_count = 0
        if user_id and user_id in self._users:
            interaction_count = self._users[user_id]["interaction_count"]

        return {
            "recommendations": recommendations,
            "seed_article_id": article_id,
            "personalized": bool(user_id and interaction_count > 0),
            "user_interaction_count": interaction_count,
        }

    def interact(
        self,
        user_id: str,
        article_id: str,
        shown_articles: Optional[List[str]] = None,
    ) -> Dict:
        """Record one interaction and update user personalization state."""
        item = self._index.get(article_id)
        if item is None:
            return {
                "status": "skipped",
                "interaction_count": 0,
                "adapted": False,
                "adaptation_time_ms": None,
            }

        state = self._ensure_user_state(user_id)
        state["interaction_count"] += 1
        state["history"].append(article_id)
        state["type_pref"][item["product_type_name"]] += 1
        state["color_pref"][item["colour_group_name"]] += 1

        emb = self._article_embedding(article_id)
        if state["ema_vector"] is None:
            state["ema_vector"] = emb
        else:
            state["ema_vector"] = (
                (1.0 - self._alpha) * state["ema_vector"] + self._alpha * emb
            )

        adapted = state["interaction_count"] % self._adapt_every == 0
        adaptation_time_ms = None
        if adapted:
            adaptation_time_ms = round(8.5 + (state["interaction_count"] % 5) * 3.7, 2)

        if shown_articles:
            state["last_shown"] = shown_articles

        return {
            "status": "ok",
            "interaction_count": state["interaction_count"],
            "adapted": adapted,
            "adaptation_time_ms": adaptation_time_ms,
        }

    def user_state(self, user_id: str) -> Dict:
        """Return user-state payload shaped like the backend contract."""
        state = self._users.get(user_id)
        if state is None:
            return {
                "user_id": user_id,
                "exists": False,
                "interaction_count": 0,
                "ema_vector": None,
                "cache_location": "none",
                "redis_ttl_remaining_seconds": None,
            }

        ttl_remaining = max(0, self._ttl_seconds - state["interaction_count"] * 180)
        cache_location = "lru" if state["interaction_count"] < 15 else "redis"
        ema_list = None
        if state["ema_vector"] is not None:
            ema_list = [float(v) for v in state["ema_vector"].tolist()]

        return {
            "user_id": user_id,
            "exists": True,
            "interaction_count": state["interaction_count"],
            "ema_vector": ema_list,
            "cache_location": cache_location,
            "redis_ttl_remaining_seconds": ttl_remaining,
        }

    def _ensure_user_state(self, user_id: str) -> Dict:
        if user_id not in self._users:
            self._users[user_id] = {
                "interaction_count": 0,
                "ema_vector": None,
                "type_pref": Counter(),
                "color_pref": Counter(),
                "history": [],
                "last_shown": [],
            }
        return self._users[user_id]

    def _passes_filters(self, item: Dict, filters: Dict[str, str]) -> bool:
        for key, value in filters.items():
            if value and item.get(key) != value:
                return False
        return True

    def _keyword_score(self, item: Dict, terms: List[str]) -> float:
        if not terms:
            return 0.22

        haystack = " ".join(
            [
                item.get("product_name", ""),
                item.get("product_type_name", ""),
                item.get("detail_desc", ""),
            ]
        ).lower()

        hits = sum(1 for term in terms if term in haystack)
        return min(0.98, 0.16 + hits * 0.22)

    def _semantic_score(self, article_id: str, query: Optional[str], use_image: bool) -> float:
        seed = _seed_from_text(f"semantic:{article_id}:{query or 'none'}:{use_image}")
        rng = random.Random(seed)
        base = 0.2 + rng.random() * 0.75
        if use_image:
            base += 0.03
        return min(base, 0.99)

    def _personal_boost(self, user_id: Optional[str], item: Dict) -> float:
        if not user_id or user_id not in self._users:
            return 0.0

        state = self._users[user_id]
        count = state["interaction_count"]
        if count <= 0:
            return 0.0

        item_type = item["product_type_name"]
        item_color = item["colour_group_name"]

        type_preference = state["type_pref"][item_type] / count
        color_preference = state["color_pref"][item_color] / count
        seen_penalty = 0.08 if item["article_id"] in state["history"][-20:] else 0.0

        return 0.32 * type_preference + 0.18 * color_preference - seen_penalty

    def _similarity(self, seed: Dict, candidate: Dict) -> float:
        score = 0.1
        if seed["product_type_name"] == candidate["product_type_name"]:
            score += 0.42
        if seed["department_name"] == candidate["department_name"]:
            score += 0.24
        if seed["garment_group_name"] == candidate["garment_group_name"]:
            score += 0.16
        if seed["colour_group_name"] == candidate["colour_group_name"]:
            score += 0.1
        return min(score, 0.98)

    def _article_embedding(self, article_id: str) -> np.ndarray:
        rng = np.random.default_rng(_seed_from_text(f"emb:{article_id}"))
        vec = rng.normal(0.0, 1.0, size=64)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _stable_noise(self, key: str, low: float, high: float) -> float:
        rng = random.Random(_seed_from_text(key))
        return rng.uniform(low, high)

    def _to_search_item(self, item: Dict, score: float) -> Dict:
        payload = dict(item)
        payload["score"] = round(float(score), 4)
        payload["image_url"] = payload.get("image_path")
        return payload

    def _to_recommendation_item(self, item: Dict, score: float) -> Dict:
        payload = dict(item)
        payload["score"] = round(float(score), 4)
        payload["image_url"] = payload.get("image_path")
        return payload

    def _build_catalog(self, size: Optional[int]) -> List[Dict]:
        if not self._articles_csv.exists():
            return []

        catalog: List[Dict] = []
        with self._articles_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                article_id = (row.get("article_id") or "").strip()
                if not article_id:
                    continue

                catalog.append(
                    {
                        "article_id": article_id,
                        "product_name": (row.get("prod_name") or "").strip()
                        or f"Article {article_id}",
                        "product_type_name": (row.get("product_type_name") or "").strip()
                        or "Unknown",
                        "colour_group_name": (row.get("colour_group_name") or "").strip()
                        or "Unknown",
                        "department_name": (row.get("department_name") or "").strip()
                        or "Unknown",
                        "index_group_name": (row.get("index_group_name") or "").strip()
                        or "Unknown",
                        "garment_group_name": (row.get("garment_group_name") or "").strip()
                        or "Unknown",
                        "detail_desc": (row.get("detail_desc") or "").strip(),
                        "image_path": self._resolve_image_path(article_id),
                        "price": self._derive_price(article_id),
                    }
                )

                if size is not None and size > 0 and len(catalog) >= size:
                    break

        return catalog

    def _derive_price(self, article_id: str) -> float:
        rng = random.Random(_seed_from_text(f"price:{article_id}"))
        return round(12.0 + rng.random() * 138.0, 2)

    def _resolve_image_path(self, article_id: str) -> str:
        if len(article_id) >= 3:
            shard = article_id[:3]
            for ext in (".jpg", ".jpeg", ".png"):
                image_file = self._images_root / shard / f"{article_id}{ext}"
                if image_file.exists():
                    return str(image_file)

        return self._fallback_image_path

    def _discover_fallback_image_path(self) -> str:
        if not self._images_root.exists():
            return ""

        for shard in sorted(self._images_root.iterdir()):
            if not shard.is_dir():
                continue
            for image_file in sorted(shard.iterdir()):
                if image_file.is_file() and image_file.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    return str(image_file)
        return ""

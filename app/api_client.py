"""API clients for Streamlit frontend.

- ModalAPIClient: talks to real backend endpoints.
- StubRecommendationAPI: local in-memory fallback for UI development.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Protocol
from urllib import error, parse, request

from app.mock_api import StubRecommendationAPI


class APIClientError(RuntimeError):
    """Raised when the frontend cannot call backend endpoints."""


class RecommendationAPI(Protocol):
    """Interface used by the Streamlit page regardless of backend mode."""

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
        ...

    def recommend(self, article_id: str, user_id: Optional[str] = None, k: int = 10) -> Dict:
        ...

    def interact(
        self,
        user_id: str,
        article_id: str,
        shown_articles: Optional[List[str]] = None,
    ) -> Dict:
        ...

    def user_state(self, user_id: str) -> Dict:
        ...


class ModalAPIClient:
    """Thin HTTP client for `/search`, `/recommend`, `/interact`, `/user_state`."""

    def __init__(self, base_url: str, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

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
        payload = {
            "query": query,
            "image_b64": image_b64,
            "filters": filters,
            "user_id": user_id,
            "alpha": alpha,
            "mode": mode,
            "limit": limit,
        }
        return self._request("POST", "/search", payload=payload)

    def recommend(self, article_id: str, user_id: Optional[str] = None, k: int = 10) -> Dict:
        payload = {"article_id": article_id, "user_id": user_id, "k": k}
        return self._request("POST", "/recommend", payload=payload)

    def interact(
        self,
        user_id: str,
        article_id: str,
        shown_articles: Optional[List[str]] = None,
    ) -> Dict:
        payload = {
            "user_id": user_id,
            "article_id": article_id,
            "shown_articles": shown_articles or [],
        }
        return self._request("POST", "/interact", payload=payload)

    def user_state(self, user_id: str) -> Dict:
        # Contract says GET /user_state with user_id.
        query = parse.urlencode({"user_id": user_id})
        path = f"/user_state?{query}"
        return self._request("GET", path)

    def _request(self, method: str, path: str, payload: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        body = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url=url, data=body, headers=headers, method=method)

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise APIClientError(
                f"HTTP {exc.code} on {path}. Detail: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise APIClientError(f"Cannot connect to backend at {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise APIClientError(f"Invalid JSON response from backend at {path}") from exc


def build_api_client(use_stub: bool, api_base_url: str):
    """Build either a stub client or a real HTTP client."""
    if use_stub:
        return StubRecommendationAPI()
    return ModalAPIClient(api_base_url)

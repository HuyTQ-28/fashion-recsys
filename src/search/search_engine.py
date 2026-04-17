import logging
import os
from typing import Dict, List, Optional

import requests as http_requests
import weaviate
from weaviate.exceptions import WeaviateQueryError
from weaviate.classes.query import Filter, MetadataQuery

from src.extractor.clip_encoder import CLIPEncoder

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Search engine backed by Weaviate Product collection.

    Supports text-to-image, image-to-image, and hybrid (BM25 + vector) search
    with native metadata filtering.
    """

    def __init__(self, client: weaviate.WeaviateClient, clip_encoder: CLIPEncoder):
        """
        Args:
            client: Connected Weaviate client.
            clip_encoder: FashionCLIP encoder for text/image queries.
        """
        self.client = client
        self.clip_encoder = clip_encoder
        self.collection = client.collections.get("Product")
        # Cache env vars once for REST fallback
        raw_url = os.environ.get("WEAVIATE_URL", "")
        self._rest_base = raw_url if raw_url.startswith("http") else f"https://{raw_url}"
        self._rest_api_key = os.environ.get("WEAVIATE_API_KEY", "")

    def search(
        self,
        query: Optional[str] = None,
        image=None,
        mode: str = "hybrid",
        alpha: float = 0.7,
        filters: Optional[Dict[str, str]] = None,
        limit: int = 20,
    ) -> List[dict]:
        """
        Perform a search query.

        Args:
            query: Text query string.
            image: PIL Image for image-to-image search.
            mode: 'semantic', 'keyword', or 'hybrid'.
            alpha: Hybrid blend weight (0 = pure BM25, 1 = pure vector).
            filters: Dict of property_name -> value for metadata filtering.
            limit: Number of results to return.

        Returns:
            List of result dicts with article metadata and scores.
        """
        # Build Weaviate filter
        weaviate_filter = self._build_filter(filters) if filters else None

        if not query and not image:
            return self._safe_fetch_objects(weaviate_filter, limit)

        if mode == "keyword" and query:
            return self._bm25_search(query, weaviate_filter, limit)
        elif mode == "semantic":
            vector = self._get_query_vector(query, image)
            return self._vector_search(vector, weaviate_filter, limit)
        elif mode == "hybrid" and query:
            vector = self._get_query_vector(query, image)
            return self._hybrid_search(query, vector, alpha, weaviate_filter, limit)
        else:
            # Default: if only image provided, do semantic search
            vector = self._get_query_vector(query, image)
            return self._vector_search(vector, weaviate_filter, limit)

    def _get_query_vector(self, query=None, image=None):
        """Get query vector from text or image."""
        if image is not None:
            return self.clip_encoder.encode_image(image).tolist()
        elif query is not None:
            return self.clip_encoder.encode_text(query).tolist()
        else:
            raise ValueError("Either query text or image must be provided")

    def _build_filter(self, filters: Dict[str, str]):
        """Build Weaviate filter from dict."""
        filter_conditions = []
        for prop, value in filters.items():
            filter_conditions.append(Filter.by_property(prop).equal(value))

        if len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            # Combine with AND
            combined = filter_conditions[0]
            for f in filter_conditions[1:]:
                combined = combined & f
            return combined

    def _vector_search(self, vector, weaviate_filter, limit) -> List[dict]:
        """Pure vector (semantic) search."""
        try:
            response = self.collection.query.near_vector(
                near_vector=vector,
                filters=weaviate_filter,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )
            return self._format_results(response.objects)
        except WeaviateQueryError as exc:
            logger.warning("Vector search unavailable (gRPC): %s — falling back to REST.", exc)
            return self._safe_fetch_objects(weaviate_filter, limit)

    def _bm25_search(self, query, weaviate_filter, limit) -> List[dict]:
        """Pure BM25 keyword search."""
        try:
            response = self.collection.query.bm25(
                query=query,
                filters=weaviate_filter,
                limit=limit,
                return_metadata=MetadataQuery(score=True),
            )
            return self._format_results(response.objects)
        except WeaviateQueryError as exc:
            logger.warning("BM25 search unavailable (gRPC): %s — falling back to REST.", exc)
            return self._safe_fetch_objects(weaviate_filter, limit)

    def _hybrid_search(self, query, vector, alpha, weaviate_filter, limit) -> List[dict]:
        """Hybrid search (BM25 + vector blend)."""
        try:
            response = self.collection.query.hybrid(
                query=query,
                vector=vector,
                alpha=alpha,
                filters=weaviate_filter,
                limit=limit,
                return_metadata=MetadataQuery(score=True),
            )
            return self._format_results(response.objects)
        except WeaviateQueryError as exc:
            logger.warning("Hybrid search unavailable (gRPC): %s — falling back to REST.", exc)
            return self._safe_fetch_objects(weaviate_filter, limit)

    def _safe_fetch_objects(self, weaviate_filter, limit: int) -> List[dict]:
        """
        Try gRPC fetch_objects first; if it fails (Deadline Exceeded on WSL2),
        fall back to Weaviate's plain REST /v1/objects endpoint which bypasses gRPC entirely.
        """
        try:
            response = self.collection.query.fetch_objects(
                filters=weaviate_filter,
                limit=limit,
            )
            return self._format_results(response.objects)
        except Exception as exc:
            logger.warning("gRPC fetch_objects failed: %s — switching to REST fallback.", exc)
            return self._rest_fetch_objects(limit)

    def _rest_fetch_objects(self, limit: int = 20) -> List[dict]:
        """
        Fetch objects via Weaviate REST API, completely bypassing gRPC.
        Used as the last-resort fallback when the gRPC channel is blocked (e.g. WSL2).
        """
        url = f"{self._rest_base}/v1/objects?class=Product&limit={limit}"
        headers = {
            "Authorization": f"Bearer {self._rest_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = http_requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for obj in data.get("objects", []):
                props = dict(obj.get("properties", {}))
                props.setdefault("score", 0.0)
                results.append(props)
            logger.info("REST fallback returned %d objects.", len(results))
            return results
        except Exception as rest_exc:
            logger.error("REST fallback also failed: %s", rest_exc)
            return []

    def _format_results(self, objects) -> List[dict]:
        """Format Weaviate response objects into result dicts."""
        results = []
        for obj in objects:
            result = {**obj.properties}
            if obj.metadata.distance is not None:
                result["score"] = 1.0 - obj.metadata.distance  # Convert distance to similarity
            elif obj.metadata.score is not None:
                result["score"] = obj.metadata.score
            else:
                result["score"] = 0.0
            results.append(result)
        return results

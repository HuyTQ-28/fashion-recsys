"""
Hybrid Search Engine backed by Weaviate.

Three search modes:
1. Semantic (pure vector): CLIP text/image embedding -> near_vector
2. Keyword (pure BM25): BM25 on product_name and detail_desc
3. Hybrid (blended): Weaviate native hybrid = vector + BM25, tunable alpha
"""

import logging
from typing import Dict, List, Optional

import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from src.search.clip_encoder import CLIPEncoder

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
        response = self.collection.query.near_vector(
            near_vector=vector,
            filters=weaviate_filter,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        return self._format_results(response.objects)

    def _bm25_search(self, query, weaviate_filter, limit) -> List[dict]:
        """Pure BM25 keyword search."""
        response = self.collection.query.bm25(
            query=query,
            filters=weaviate_filter,
            limit=limit,
            return_metadata=MetadataQuery(score=True),
        )
        return self._format_results(response.objects)

    def _hybrid_search(self, query, vector, alpha, weaviate_filter, limit) -> List[dict]:
        """Hybrid search (BM25 + vector blend)."""
        response = self.collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            filters=weaviate_filter,
            limit=limit,
            return_metadata=MetadataQuery(score=True),
        )
        return self._format_results(response.objects)

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

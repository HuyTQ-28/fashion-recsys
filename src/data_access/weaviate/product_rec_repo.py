import weaviate
from typing import List
import logging

logger = logging.getLogger(__name__)

class ProductRecRepo:
    """
    Repository for the Weaviate ProductRec collection, containing 64-dim global space embeddings.
    """
    def __init__(self, client: weaviate.WeaviateClient):
        self.collection = client.collections.get("ProductRec")

    def search_ids_near_vector(self, query_vec64: List[float], limit: int = 500) -> List[str]:
        """
        Perform a vector search in the Global Space and return ONLY article_ids
        to minimize I/O overhead.
        """
        try:
            response = self.collection.query.near_vector(
                near_vector=query_vec64,
                limit=limit,
                return_properties=["article_id"]
            )
            return [obj.properties["article_id"] for obj in response.objects]
        except Exception as e:
            logger.error(f"Failed to query ProductRec collection: {e}")
            return []

    def fetch_default_ids(self, limit: int = 50) -> List[str]:
        """Fallback retrieval for cold-start sessions when no user state exists."""
        try:
            response = self.collection.query.fetch_objects(
                limit=limit,
                return_properties=["article_id"],
            )
            return [obj.properties["article_id"] for obj in response.objects if "article_id" in obj.properties]
        except Exception as e:
            logger.error(f"Failed to fetch default ProductRec ids: {e}")
            return []

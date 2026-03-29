"""
Weaviate Schema Design & Setup.

Owner: Member 3 (Search & Infrastructure)

Two collections:
- Product: 512-dim FashionCLIP embeddings + article metadata (for hybrid search)
- ProductRec: 64-dim Student MLP embeddings (for graph-aware KNN recommendations)
"""

import logging
from typing import Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization

logger = logging.getLogger(__name__)


def get_weaviate_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedded: bool = False,
) -> weaviate.WeaviateClient:
    """
    Create a Weaviate client.

    Args:
        url: Weaviate Cloud URL. If None, uses embedded.
        api_key: Weaviate API key.
        embedded: If True, use Weaviate Embedded (local, zero-infra).

    Returns:
        Connected WeaviateClient.
    """
    if embedded:
        client = weaviate.connect_to_embedded()
    elif url and api_key:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
        )
    elif url:
        client = weaviate.connect_to_custom(
            http_host=url.replace("https://", "").replace("http://", ""),
            http_port=8080,
            grpc_port=50051,
        )
    else:
        client = weaviate.connect_to_local()

    logger.info(f"Connected to Weaviate: {client.is_ready()}")
    return client


def create_product_collection(client: weaviate.WeaviateClient, delete_existing: bool = False) -> None:
    """
    Create the Product collection for hybrid search.

    Schema:
    - Named vector: clip_embedding (512-dim, cosine)
    - Filterable metadata properties from articles.csv
    - BM25 text search on product_name and detail_desc
    """
    collection_name = "Product"

    if client.collections.exists(collection_name):
        if delete_existing:
            client.collections.delete(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists, skipping")
            return

    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=weaviate.classes.config.VectorDistances.COSINE,
        ),
        properties=[
            Property(name="article_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
            Property(name="product_name", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
            Property(name="product_type_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
            Property(name="product_group_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
            Property(name="colour_group_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
            Property(name="department_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
            Property(name="index_group_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
            Property(name="garment_group_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
            Property(name="detail_desc", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
            Property(name="image_path", data_type=DataType.TEXT),
        ],
    )
    logger.info(f"Created collection: {collection_name}")


def create_product_rec_collection(client: weaviate.WeaviateClient, delete_existing: bool = False) -> None:
    """
    Create the ProductRec collection for graph-aware KNN recommendations.

    Schema:
    - Named vector: mlp_embedding (64-dim, Euclidean)
    - Only article_id property (lightweight)
    """
    collection_name = "ProductRec"

    if client.collections.exists(collection_name):
        if delete_existing:
            client.collections.delete(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists, skipping")
            return

    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=weaviate.classes.config.VectorDistances.L2,  # Euclidean for 64-dim MLP space
        ),
        properties=[
            Property(name="article_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
        ],
    )
    logger.info(f"Created collection: {collection_name}")


def setup_all_collections(client: weaviate.WeaviateClient, delete_existing: bool = False) -> None:
    """Create both Product and ProductRec collections."""
    create_product_collection(client, delete_existing)
    create_product_rec_collection(client, delete_existing)
    logger.info("All collections created successfully")

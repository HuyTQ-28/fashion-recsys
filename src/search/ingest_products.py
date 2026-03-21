"""
Product Ingestion into Weaviate Collections.

Owner: Member 3 (Search & Infrastructure)

Two ingestion scripts:
1. ingest_products: CLIP embeddings + metadata -> Product collection
2. ingest_rec_embeddings: Student MLP 64-dim -> ProductRec collection
"""

import logging
from typing import Dict

import pandas as pd
import torch
import weaviate

logger = logging.getLogger(__name__)


def ingest_products(
    client: weaviate.WeaviateClient,
    clip_embeddings: Dict[str, torch.Tensor],
    articles_df: pd.DataFrame,
    batch_size: int = 100,
) -> int:
    """
    Batch-import articles into the Product collection with CLIP embeddings and metadata.

    Args:
        client: Connected Weaviate client.
        clip_embeddings: Dict of article_id -> CLIP embedding [512].
        articles_df: DataFrame with article metadata.
        batch_size: Batch size for import.

    Returns:
        Number of successfully imported objects.
    """
    collection = client.collections.get("Product")
    articles_df = articles_df.set_index("article_id") if "article_id" in articles_df.columns else articles_df

    count = 0

    with collection.batch.dynamic() as batch:
        for article_id, embedding in clip_embeddings.items():
            # Get metadata
            props = {"article_id": str(article_id)}

            if article_id in articles_df.index:
                row = articles_df.loc[article_id]
                for col in [
                    "product_name", "product_type_name", "product_group_name",
                    "colour_group_name", "department_name", "index_group_name",
                    "garment_group_name", "detail_desc",
                ]:
                    if col in row.index and pd.notna(row[col]):
                        props[col] = str(row[col])

                # Image path
                prefix = f"0{str(article_id)[:2]}"
                props["image_path"] = f"images/{prefix}/0{article_id}.jpg"

            batch.add_object(
                properties=props,
                vector=embedding.tolist(),
            )
            count += 1

    logger.info(f"Ingested {count} articles into Product collection")
    return count

def ingest_rec_embeddings(
    client: weaviate.WeaviateClient,
    mlp_embeddings: Dict[str, torch.Tensor],
    batch_size: int = 100,
) -> int:
    """
    Batch-import Student MLP embeddings into the ProductRec collection.

    Args:
        client: Connected Weaviate client.
        mlp_embeddings: Dict of article_id -> MLP embedding [64].
        batch_size: Batch size for import.

    Returns:
        Number of successfully imported objects.
    """
    collection = client.collections.get("ProductRec")
    count = 0

    with collection.batch.dynamic() as batch:
        for article_id, embedding in mlp_embeddings.items():
            batch.add_object(
                properties={"article_id": str(article_id)},
                vector=embedding.tolist(),
            )
            count += 1

    logger.info(f"Ingested {count} MLP embeddings into ProductRec collection")
    return count

import json
import logging
import os
from typing import Dict, Optional

import pandas as pd
import torch
import weaviate

logger = logging.getLogger(__name__)

def ingest_products(
    client: weaviate.WeaviateClient,
    clip_embeddings: Dict[str, torch.Tensor],
    articles_df: pd.DataFrame,
    cloudinary_urls: Optional[Dict[str, str]] = None,
) -> int:
    """Batch-import articles into the Product collection"""

    METADATA_COLS = [
        "prod_name", "product_type_name", "product_group_name",
        "colour_group_name", "department_name", "index_group_name",
        "garment_group_name", "detail_desc",
    ]

    collection = client.collections.get("Product")

    if "article_id" in articles_df.columns:
        articles_df = articles_df.set_index("article_id")

    cloudinary_urls = cloudinary_urls or {}
    count = 0

    with collection.batch.dynamic() as batch:
        for article_id, embedding in clip_embeddings.items():
            article_id_str = str(article_id).zfill(10)

            # --- Base properties ---
            props: dict = {"article_id": article_id_str}

            # --- Metadata ---
            if article_id_str in articles_df.index:
                row = articles_df.loc[article_id_str]
                for col in METADATA_COLS:
                    if col in row.index and pd.notna(row[col]):
                        props[col] = str(row[col])

            # --- Local image path ---
            prefix = article_id_str[:3]
            props["image_path"] = f"images/{prefix}/{article_id_str}.jpg"

            # --- Cloudinary URL ---
            if article_id_str in cloudinary_urls:
                props["image_url"] = cloudinary_urls[article_id_str]
            elif article_id in cloudinary_urls:
                props["image_url"] = cloudinary_urls[article_id]

            batch.add_object(
                properties=props,
                vector=embedding.tolist(),
                uuid=weaviate.util.generate_uuid5(article_id_str),
            )
            count += 1

    failed_objs = collection.batch.failed_objects
    if len(failed_objs) > 0:
        logger.error(f"Failed to ingest {len(failed_objs)} objects into Product collection.")
        for i, failed in enumerate(failed_objs[:5]):
            logger.error(f"Failed object {i+1}: {failed.message}")

    logger.info("Ingested %d articles into Product collection", count)
    return count


def ingest_rec_embeddings(
    client: weaviate.WeaviateClient,
    mlp_embeddings: Dict[str, torch.Tensor],
) -> int:
    """Batch-import Student MLP embeddings into the ProductRec collection"""

    collection = client.collections.get("ProductRec")
    count = 0

    with collection.batch.dynamic() as batch:
        for article_id, embedding in mlp_embeddings.items():
            article_id_str = str(article_id).zfill(10)
            prefix = article_id_str[:3]
            batch.add_object(
                properties={
                    "article_id": article_id_str,
                    "image_url": f"images/{prefix}/{article_id_str}.jpg",
                },
                vector=embedding.tolist(),
                uuid=weaviate.util.generate_uuid5(article_id_str),
            )
            count += 1

    failed_objs = collection.batch.failed_objects
    if len(failed_objs) > 0:
        logger.error(f"Failed to ingest {len(failed_objs)} objects into ProductRec collection.")
        for i, failed in enumerate(failed_objs[:5]):
            logger.error(f"Failed object {i+1}: {failed.message}")

    logger.info("Ingested %d MLP embeddings into ProductRec collection", count)
    return count

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.search.weaviate_setup import get_weaviate_client, setup_all_collections
    from src.search.cloudinary_uploader import (
        configure_cloudinary,
        upload_images_to_cloudinary,
        save_url_map,
        load_url_map,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()

    DATA_DIR = "dataset/subset_1week"
    WEIGHTS_DIR = "weights"
    CLIP_PATH = os.path.join(DATA_DIR, "clip_embeddings.pt")
    ARTICLES_CSV = os.path.join(DATA_DIR, "articles.csv")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    CLOUDINARY_CACHE = os.path.join(DATA_DIR, "cloudinary_urls.json")
    MLP_STATE = os.path.join(WEIGHTS_DIR, "student_mlp.pt")

    weaviate_url = os.environ.get("WEAVIATE_URL")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

    if not weaviate_url or not weaviate_api_key:
        logger.error("WEAVIATE_URL and WEAVIATE_API_KEY must be set in your .env")
        sys.exit(1)

    cloudinary_urls: Dict[str, str] = {}

    if os.path.exists(CLOUDINARY_CACHE):
        logger.info("Found cached Cloudinary URLs at %s, loading...", CLOUDINARY_CACHE)
        cloudinary_urls = load_url_map(CLOUDINARY_CACHE)
    elif os.path.isdir(IMAGE_DIR):
        cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
        cloud_key = os.environ.get("CLOUDINARY_API_KEY")
        cloud_secret = os.environ.get("CLOUDINARY_API_SECRET")

        if cloud_name and cloud_key and cloud_secret:
            configure_cloudinary(cloud_name, cloud_key, cloud_secret)
            cloudinary_urls = upload_images_to_cloudinary(image_dir=IMAGE_DIR)
            save_url_map(cloudinary_urls, CLOUDINARY_CACHE)
        else:
            logger.warning(
                "CLOUDINARY_* env vars not set — skipping image upload. "
                "Products will be ingested without image_url."
            )
    else:
        logger.warning("Image directory %s not found — skipping Cloudinary upload.", IMAGE_DIR)

    # ----- Connect to Weaviate & ensure schema exists -----
    client = get_weaviate_client(weaviate_url, weaviate_api_key)
    setup_all_collections(client, delete_existing=False)

    # ----- Load CLIP embeddings + metadata, then ingest Products -----
    if os.path.exists(CLIP_PATH):
        logger.info("Loading CLIP embeddings from %s...", CLIP_PATH)
        clip_embeddings: Dict[str, torch.Tensor] = torch.load(
            CLIP_PATH, map_location="cpu", weights_only=True
        )

        logger.info("Loading article metadata from %s...", ARTICLES_CSV)
        articles_df = pd.read_csv(ARTICLES_CSV)
        articles_df["article_id"] = articles_df["article_id"].astype(str).str.zfill(10)

        logger.info("Starting Product ingestion (n=%d)...", len(clip_embeddings))
        ingest_products(client, clip_embeddings, articles_df, cloudinary_urls)
    else:
        logger.warning("%s not found — skipping Product ingestion.", CLIP_PATH)
        clip_embeddings = {}

    # ----- Compute & ingest 64-dim MLP rec embeddings -----
    if os.path.exists(MLP_STATE) and clip_embeddings:
        from src.models.student_mlp import StudentMLP

        logger.info("Loading Student MLP from %s...", MLP_STATE)
        sample_dim = next(iter(clip_embeddings.values())).shape[0]
        model = StudentMLP(layer_dims=[int(sample_dim), 256, 128, 64])
        model.load_state_dict(torch.load(MLP_STATE, map_location="cpu", weights_only=True))
        model.eval()

        logger.info("Computing 64-dim projection vectors for %d articles...", len(clip_embeddings))
        rec_embeddings: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for aid, vec in clip_embeddings.items():
                proj = model(vec.unsqueeze(0)).squeeze(0) 
                rec_embeddings[aid] = proj

        logger.info("Starting ProductRec ingestion...")
        ingest_rec_embeddings(client, rec_embeddings)
    else:
        logger.warning("Skipping ProductRec ingestion (MLP weights or CLIP embeddings not available).")

    client.close()
    logger.info("All ingestion pipelines complete.")
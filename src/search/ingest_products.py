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
                    "prod_name", "product_type_name", "product_group_name",
                    "colour_group_name", "department_name", "index_group_name",
                    "garment_group_name", "detail_desc",
                ]:
                    if col in row.index and pd.notna(row[col]):
                        props[col] = str(row[col])

                # Image path
                prefix = f"{str(article_id)[:3]}"
                props["image_path"] = f"images/{prefix}/{article_id}.jpg"

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
            props = {"article_id": str(article_id)}
            
            # Image path
            prefix = f"{str(article_id)[:3]}"
            props["image_path"] = f"images/{prefix}/{article_id}.jpg"
            batch.add_object(
                properties=props,
                vector=embedding.tolist(),
            )
            count += 1

    logger.info(f"Ingested {count} MLP embeddings into ProductRec collection")
    return count

if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv
    from weaviate_setup import get_weaviate_client
    
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    
    url = os.environ.get("WEAVIATE_URL")
    api_key = os.environ.get("WEAVIATE_API_KEY")
    
    if not url or not api_key:
        logger.error("Error: WEAVIATE_URL and WEAVIATE_API_KEY must be in your .env")
        exit(1)
        
    client = get_weaviate_client(url, api_key)
    data_dir = "dataset/subset_1week"
    
    # 1. Ingest Base Products (CLIP text-image search + metadata)
    clip_path = os.path.join(data_dir, "clip_embeddings.pt")
    if os.path.exists(clip_path):
        logger.info(f"Loading {clip_path}...")
        clip_embeddings = torch.load(clip_path, map_location="cpu", weights_only=True)
        articles_df = pd.read_csv(os.path.join(data_dir, "articles.csv"))
        articles_df["article_id"] = articles_df["article_id"].astype(str).str.zfill(10)
        
        logger.info("Starting base Product ingestion...")
        ingest_products(client, clip_embeddings, articles_df)
    else:
        logger.warning(f"{clip_path} not found. Skipping Product catalog.")

    # 2. Compute & Ingest Rec Vectors (64-dim MLP space)
    mlp_state = os.path.join(data_dir, "student_mlp_full.pt")
    if os.path.exists(mlp_state) and os.path.exists(clip_path):
        import torch.nn.functional as F
        
        # Ensure 'src' is in path to load model
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
        from MLP_student.model import StudentMLP
        
        logger.info(f"Loading Student MLP from {mlp_state}...")
        sample_dim = next(iter(clip_embeddings.values())).shape[0]
        model = StudentMLP(in_dim=sample_dim, out_dim=64)
        model.load_state_dict(torch.load(mlp_state, map_location="cpu", weights_only=True))
        model.eval()
        
        logger.info("Computing catalog 64-dim projection vectors...")
        rec_embeddings = {}
        with torch.no_grad():
            for aid, vec in clip_embeddings.items():
                # Forward pass + normalize correctly before storing
                proj = F.normalize(model(vec.unsqueeze(0)), dim=1).squeeze(0)
                rec_embeddings[aid] = proj
        
        logger.info("Starting ProductRec ingestion...")
        ingest_rec_embeddings(client, rec_embeddings)
    else:
        logger.warning(f"Could not load either MLP weights or CLIP file to compute rec vector catalog!")

    client.close()
    logger.info("All ingestion pipelines complete.")
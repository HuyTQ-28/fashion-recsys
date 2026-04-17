import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import cloudinary
import cloudinary.uploader
from tqdm import tqdm
logger = logging.getLogger(__name__)

# -------- Configuration ----------

def configure_cloudinary(
    cloud_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> None:
    """
    Configure the Cloudinary SDK from explicit args or environment variables.
    """
    cloudinary.config(
        cloud_name=cloud_name or os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=api_key or os.environ["CLOUDINARY_API_KEY"],
        api_secret=api_secret or os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )
    logger.info("Cloudinary configured (cloud=%s)", cloudinary.config().cloud_name)


# ------- Upload function ---------

def upload_images_to_cloudinary(
    image_dir: str,
    output_path: str,
    article_ids: Optional[list] = None,
    folder: str = "fashion-recsys/products",
    batch_size: int = 50,
) -> Dict[str, str]:

    image_dir = Path(image_dir)
    url_map: Dict[str, str] = {}
    failed: list = []

    # Load existing URLs if the file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                url_map = json.load(f)
            logger.info("Found existing JSON file. Loaded %d already uploaded images.", len(url_map))
        except json.JSONDecodeError:
            logger.warning("Existing JSON file is corrupted or empty. Starting fresh.")
    else:
        logger.info("No existing JSON file found. Starting fresh upload.")

    # Collect image paths
    if article_ids is not None:
        image_paths = []
        for aid in article_ids:
            prefix = str(aid)[:3]
            img_path = image_dir / prefix / f"{aid}.jpg"
            if img_path.exists():
                image_paths.append((str(aid), img_path))
            else:
                logger.warning("Image not found, skipping: %s", img_path)
    else:
        image_paths = [
            (img.stem, img)
            for img in sorted(image_dir.rglob("*.jpg"))
        ]

    # Filter out already uploaded images
    pending_uploads = []
    for article_id, img_path in image_paths:
        if article_id not in url_map:
            pending_uploads.append((article_id, img_path))
            
    logger.info("Total images: %d | Skipped: %d | Pending upload: %d", 
                len(image_paths), len(image_paths) - len(pending_uploads), len(pending_uploads))

    # Upload loop
    if not pending_uploads:
        logger.info("Everything is up to date! No new images to upload.")
        return url_map

    for i, (article_id, img_path) in enumerate(
        tqdm(pending_uploads, desc="Uploading to Cloudinary"), start=1
    ):
        try:
            result = cloudinary.uploader.upload(
                str(img_path),
                public_id=article_id,
                asset_folder=folder, 
                overwrite=False,
                resource_type="image",
            )
            url_map[article_id] = result["secure_url"]
        except Exception as exc:
            logger.warning("Failed to upload %s: %s", article_id, exc)
            failed.append(article_id)

        # Save JSON file every batch_size to prevent data loss
        if i % batch_size == 0:
            save_url_map(url_map, output_path)

    logger.info("Cloudinary upload complete — new success: %d, failed: %d", 
                len(pending_uploads) - len(failed), len(failed))
    
    return url_map

# -------- Persistence helpers ---------

def save_url_map(url_map: Dict[str, str], output_path: str) -> None:
    """Persist the article_id -> url mapping to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(url_map, f, indent=2)
    logger.info("Saved %d URLs to %s", len(url_map), output_path)


def load_url_map(path: str) -> Dict[str, str]:
    """Load a previously saved article_id -> url mapping from JSON."""
    with open(path) as f:
        url_map = json.load(f)
    logger.info("Loaded %d Cloudinary URLs from %s", len(url_map), path)
    return url_map


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Batch-upload product images to Cloudinary.")
    parser.add_argument("--image_dir", default="dataset/subset_1week/images", help="Root directory of product images")
    parser.add_argument(
        "--output",
        default="dataset/subset_1week/cloudinary_urls.json",
        help="Path to save the article_id -> url JSON mapping",
    )
    parser.add_argument(
        "--folder",
        default="fashion-recsys/products",
        help="Cloudinary folder to upload images into",
    )
    args = parser.parse_args()

    configure_cloudinary()

    url_map = upload_images_to_cloudinary(
        image_dir=args.image_dir,
        output_path=args.output,
        folder=args.folder,
    )

    save_url_map(url_map, args.output)

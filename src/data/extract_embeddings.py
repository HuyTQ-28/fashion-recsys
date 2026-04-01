"""
FashionCLIP Embedding Extraction.

Owner: Member 3 (Search & Infrastructure)

Extracts 512-dim image embeddings using FashionCLIP vision encoder.
All embeddings are L2-normalized for cosine similarity in Weaviate.

Usage:
    python -m src.data.extract_embeddings --image_dir data/raw/images --output data/processed/clip_embeddings.pt
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FashionCLIPExtractor:
    """Extract image embeddings using FashionCLIP (patrickjohncyh/fashion-clip)."""

    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip", device: Optional[str] = None):
        """
        Initialize the FashionCLIP extractor.

        Args:
            model_name: HuggingFace model name for FashionCLIP.
            device: Device to run on ('cuda', 'cpu', or None for auto-detect).
        """
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FashionCLIP model: {model_name} on {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.embedding_dim = self.model.config.projection_dim  # 512

    @torch.no_grad()
    def extract_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract a single image embedding.

        Args:
            image: PIL Image.

        Returns:
            L2-normalized embedding tensor of shape [512].
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embedding = self.model.get_image_features(**inputs)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """
        Extract embeddings for a batch of images.

        Args:
            images: List of PIL Images.
            batch_size: Processing batch size.

        Returns:
            L2-normalized embeddings tensor of shape [N, 512].
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def extract_from_directory(
        self,
        image_dir: str,
        article_ids: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for all images in a directory.

        H&M image structure: images/{0-prefix}/{article_id}.jpg
        e.g., images/010/0108775015.jpg

        Args:
            image_dir: Root directory containing product images.
            article_ids: Optional list of article IDs to process (if None, process all).
            batch_size: Processing batch size.

        Returns:
            Dict mapping article_id -> L2-normalized embedding tensor [512].
        """
        image_dir = Path(image_dir)
        embeddings = {}

        # Collect image paths
        if article_ids is not None:
            image_paths = []
            for aid in article_ids:
                # H&M image naming: 0{article_id[:3]}/{article_id}.jpg
                prefix = f"0{aid[:2]}"
                img_path = image_dir / prefix / f"0{aid}.jpg"
                if img_path.exists():
                    image_paths.append((aid, img_path))
                else:
                    logger.warning(f"Image not found: {img_path}")
        else:
            image_paths = []
            for img_file in image_dir.rglob("*.jpg"):
                aid = img_file.stem.lstrip("0")
                image_paths.append((aid, img_file))

        logger.info(f"Processing {len(image_paths)} images from {image_dir}")

        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP embeddings"):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            batch_ids = []

            for aid, img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    batch_ids.append(aid)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")

            if batch_images:
                batch_embeddings = self.extract_batch(batch_images, batch_size=len(batch_images))
                for aid, emb in zip(batch_ids, batch_embeddings):
                    embeddings[aid] = emb

        logger.info(f"Extracted {len(embeddings)} embeddings (dim={self.embedding_dim})")
        return embeddings


def save_embeddings(embeddings: Dict[str, torch.Tensor], output_path: str) -> None:
    """Save embeddings dict to a .pt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)
    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")


def load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """Load embeddings dict from a .pt file."""
    embeddings = torch.load(path, weights_only=False)
    logger.info(f"Loaded {len(embeddings)} embeddings from {path}")
    return embeddings


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract FashionCLIP image embeddings")
    parser.add_argument("--image_dir", required=True, help="Directory containing product images")
    parser.add_argument("--output", default="data/processed/clip_embeddings.pt", help="Output path")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    extractor = FashionCLIPExtractor()
    embeddings = extractor.extract_from_directory(args.image_dir, batch_size=args.batch_size)
    save_embeddings(embeddings, args.output)

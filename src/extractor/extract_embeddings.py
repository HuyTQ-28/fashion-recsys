import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FashionCLIPExtractor:

    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip", device: Optional[str] = None):
        """
        Initialize the FashionCLIP extractor.
        """
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FashionCLIP model: {model_name} on {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.embedding_dim = self.model.config.projection_dim

    @torch.no_grad()
    def extract_text_embedding(self, text: str) -> torch.Tensor:
        """
        Extract a single text embedding.
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        output = self.model.get_text_features(**inputs)
        embedding = output.pooler_output if not isinstance(output, torch.Tensor) else output
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def extract_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract a single image embedding.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.get_image_features(**inputs)
        embedding = output.pooler_output if not isinstance(output, torch.Tensor) else output
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """
        Extract embeddings for a batch of images.
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            output = self.model.get_image_features(**inputs)
            embeddings = output.pooler_output if not isinstance(output, torch.Tensor) else output
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
        """
        image_dir = Path(image_dir)
        embeddings = {}

        # Collect image paths
        if article_ids is not None:
            image_paths = []
            for aid in article_ids:
                prefix = f"{aid[:3]}"
                img_path = image_dir / prefix / f"{aid}.jpg"
                if img_path.exists():
                    image_paths.append((aid, img_path))
                else:
                    logger.warning(f"Image not found: {img_path}")
        else:
            image_paths = []
            for img_file in image_dir.rglob("*.jpg"):
                aid = img_file.stem
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


# Inference on Modal
import modal
app = modal.App("fashion-clip-extractor")

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "transformers", "Pillow", "tqdm")
)

MOUNT_DIR = "/data"
vol = modal.Volume.from_name("fa_volume")

@app.cls(image=modal_image, gpu="A100", volumes={MOUNT_DIR: vol}, timeout=7200)
class ModalCLIPExtractor:
    """
    Wrapper class to run FashionCLIPExtractor on Modal.
    """
    @modal.enter()
    def setup(self):
        self.extractor = FashionCLIPExtractor(device="cuda")

    @modal.method()
    def process_directory(self, input_dir_rel: str, output_file_rel: str, batch_size: int = 64):
        full_image_dir = f"{MOUNT_DIR}/{input_dir_rel}"
        full_output_path = f"{MOUNT_DIR}/{output_file_rel}"
        
        logger.info(f"Modal is reading images from: {full_image_dir}")
        
        embeddings = self.extractor.extract_from_directory(
            image_dir=full_image_dir,
            batch_size=batch_size
        )
        
        save_embeddings(embeddings, full_output_path)
        
        vol.commit()
        return f"Successfully saved to {full_output_path} on Modal Volume"

@app.local_entrypoint()
def main(
    image_dir: str = "subset_1week/images",
    output: str = "processed/clip_embeddings.pt",
    batch_size: int = 64
):
    logger.info("Initializing connection to Modal GPU...")
    
    modal_extractor = ModalCLIPExtractor()
    
    result_msg = modal_extractor.process_directory.remote(
        input_dir_rel=image_dir,
        output_file_rel=output,
        batch_size=batch_size
    )
    
    logger.info(f"☁️ MODAL RESPONSE: {result_msg}")

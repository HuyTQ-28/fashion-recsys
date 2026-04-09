import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

class CLIPEncoder:
    """Runtime encoder for search queries using FashionCLIP."""

    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip", device: Optional[str] = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info(f"CLIPEncoder loaded on {self.device}")

    @torch.no_grad()
    def encode_text(self, query: str) -> torch.Tensor:
        """
        Encode a text query into a 512-dim CLIP embedding.

        Args:
            query: Natural language search query (e.g., "red summer dress").
        Returns:
            L2-normalized embedding tensor [512].
        """
        inputs = self.processor(text=query, return_tensors="pt").to(self.device)
        embedding = self.model.get_text_features(**inputs)
        if not isinstance(embedding, torch.Tensor):
            embedding = embedding.pooler_output
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.cpu()

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode an image into a 512-dim CLIP embedding.

        Args:
            image: PIL Image.

        Returns:
            L2-normalized embedding tensor [512].
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embedding = self.model.get_image_features(**inputs)
        if not isinstance(embedding, torch.Tensor):
            embedding = embedding.pooler_output
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze(0).cpu()

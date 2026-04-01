"""
FashionCLIP Text/Image Encoder for Search Queries.

Owner: Member 3 (Search & Infrastructure)

Reusable encoder for encoding text queries and images through FashionCLIP.
Used at runtime for hybrid search (text-to-image, image-to-image).
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Fashion-specific prompt templates to improve retrieval quality
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a fashion product: {}",
    "a clothing item: {}",
    "{}",
]


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
    def encode_text(self, query: str, use_templates: bool = True) -> torch.Tensor:
        """
        Encode a text query into a 512-dim CLIP embedding.

        Args:
            query: Natural language search query (e.g., "red summer dress").
            use_templates: If True, encode with fashion-specific prompt templates
                          and average the results for better retrieval.

        Returns:
            L2-normalized embedding tensor [512].
        """
        if use_templates:
            texts = [template.format(query) for template in PROMPT_TEMPLATES]
        else:
            texts = [query]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        embeddings = self.model.get_text_features(**inputs)
        embedding = embeddings.mean(dim=0)  # Average across templates
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
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze(0).cpu()

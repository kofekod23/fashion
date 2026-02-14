"""OpenCLIP model implementation (supports Marqo FashionCLIP)."""

import logging
from typing import Union

import numpy as np
import torch
from PIL import Image

from .base import BaseModel
from src.config import config

logger = logging.getLogger(__name__)


class OpenCLIPModel(BaseModel):
    """OpenCLIP model wrapper for Marqo/marqo-fashionCLIP."""

    def __init__(self, model_name: str = "Marqo/marqo-fashionCLIP", device: str | None = None):
        self._model_name = model_name
        self._device = device or config.device

        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self._device = "cpu"

        logger.info(f"Loading OpenCLIP model {model_name} on {self._device}")

        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:{model_name}", device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_name}")
        self._model.eval()

        # Marqo FashionCLIP is 512d
        self._dimension = 512

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        img = self._load_image(image)
        with torch.no_grad():
            image_tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            features = self._model.encode_image(image_tensor)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().float().numpy().flatten()

    def encode_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text query cannot be empty")
        with torch.no_grad():
            tokens = self._tokenizer([text]).to(self._device)
            features = self._model.encode_text(tokens)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().float().numpy().flatten()

    def encode_images_batch(self, images: list[Union[Image.Image, str]]) -> np.ndarray:
        imgs = [self._load_image(img) for img in images]
        with torch.no_grad():
            tensors = torch.stack([self._preprocess(img) for img in imgs]).to(self._device)
            features = self._model.encode_image(tensors)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def get_dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

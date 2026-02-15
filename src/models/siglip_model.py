"""Marqo FashionSigLIP model implementation (OpenCLIP model + HuggingFace tokenizer).

open_clip.get_tokenizer() fails on T5Tokenizer (batch_encode_plus error).
AutoModel.from_pretrained() fails on meta tensors (.to(device) error).
Hybrid approach: OpenCLIP for model/images, HF AutoProcessor for text tokenization.
"""

import logging
from typing import Union

import numpy as np
import torch
from PIL import Image

from .base import BaseModel
from src.config import config

logger = logging.getLogger(__name__)


class SigLIPModel(BaseModel):
    """Marqo FashionSigLIP model wrapper (OpenCLIP + HF tokenizer)."""

    MAX_TEXT_TOKENS = 64

    def __init__(self, model_name: str = "Marqo/marqo-fashionSigLIP", device: str | None = None):
        self._model_name = model_name
        self._device = device or config.device

        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self._device = "cpu"

        logger.info(f"Loading FashionSigLIP model {model_name} on {self._device}")

        import open_clip
        from transformers import AutoProcessor

        # Model + image preprocess via OpenCLIP
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:{model_name}", device=self._device
        )
        self._model.eval()

        # Tokenizer via HF (open_clip.get_tokenizer breaks on T5)
        hf_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._tokenizer = hf_processor.tokenizer

        # Marqo FashionSigLIP is 768d
        self._dimension = 768

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
            inputs = self._tokenizer(
                [text], return_tensors="pt", padding=True,
                truncation=True, max_length=self.MAX_TEXT_TOKENS,
            )
            tokens = inputs["input_ids"].to(self._device)
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

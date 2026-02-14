"""SigLIP2 model implementation."""

import logging
from typing import Union

import numpy as np
import torch
from PIL import Image

from .base import BaseModel
from src.config import config

logger = logging.getLogger(__name__)


class SigLIPModel(BaseModel):
    """SigLIP2-SO400M model wrapper using HuggingFace transformers."""

    MAX_TEXT_TOKENS = 64

    def __init__(self, model_name: str = "google/siglip2-so400m-patch14-384", device: str | None = None):
        self._model_name = model_name
        self._device = device or config.device

        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self._device = "cpu"

        logger.info(f"Loading SigLIP2 model {model_name} on {self._device}")

        from transformers import AutoModel, AutoProcessor

        self._model = AutoModel.from_pretrained(model_name)
        self._processor = AutoProcessor.from_pretrained(model_name)

        self._model.to(self._device)
        self._model.eval()

        # SigLIP2-SO400M is 1152d
        self._dimension = 1152

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def _to_tensor(self, feat):
        if isinstance(feat, torch.Tensor):
            return feat
        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            return feat.pooler_output
        if hasattr(feat, "last_hidden_state"):
            return feat.last_hidden_state[:, 0, :]
        raise ValueError(f"Cannot extract tensor from {type(feat)}")

    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        img = self._load_image(image)
        with torch.no_grad():
            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            features = self._model.get_image_features(**inputs)
            features = self._to_tensor(features)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().float().numpy().flatten()

    def encode_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text query cannot be empty")
        with torch.no_grad():
            tok = getattr(self._processor, "tokenizer", self._processor)
            inputs = tok(
                [text], return_tensors="pt", padding=True,
                truncation=True, max_length=self.MAX_TEXT_TOKENS,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            features = self._model.get_text_features(**inputs)
            features = self._to_tensor(features)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().float().numpy().flatten()

    def encode_images_batch(self, images: list[Union[Image.Image, str]]) -> np.ndarray:
        imgs = [self._load_image(img) for img in images]
        with torch.no_grad():
            inputs = self._processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            features = self._model.get_image_features(**inputs)
            features = self._to_tensor(features)
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

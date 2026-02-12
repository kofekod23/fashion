"""CLIP model implementation (supports Fashion CLIP)."""

import logging
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel as HFCLIPModel
from transformers import CLIPProcessor

from .base import BaseModel
from src.config import config

logger = logging.getLogger(__name__)


class CLIPModel(BaseModel):
    """CLIP model wrapper using HuggingFace transformers."""

    MODEL_DIMENSIONS = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-base-patch16": 512,
        "openai/clip-vit-large-patch14": 768,
        "openai/clip-vit-large-patch14-336": 768,
        "patrickjohncyh/fashion-clip": 512,
    }

    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip", device: str | None = None):
        """Initialize the CLIP model."""
        self._model_name = model_name
        self._device = device or config.device

        if self._device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            self._device = "cpu"

        print(f"Loading CLIP model on device: {self._device}")

        self._model = HFCLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)

        if self._device == "cuda" and config.use_half_precision:
            self._model = self._model.half()
            self._use_half = True
            print("Using half precision (FP16) for GPU inference")
        else:
            self._use_half = False

        self._model.to(self._device)
        self._model.eval()

        self._dimension = self.MODEL_DIMENSIONS.get(
            model_name, self._model.config.projection_dim
        )

        if self._device == "cuda":
            gpu_info = config.get_cuda_info()
            print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']})")

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """Load image from path or return as-is if already PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """Encode a single image into a vector embedding."""
        try:
            img = self._load_image(image)

            with torch.no_grad():
                inputs = self._processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                features = self._model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)

            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.exception(f"Failed to encode image: {e}")
            raise RuntimeError(f"Image encoding failed: {e}")

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into a vector embedding."""
        try:
            if not text or not text.strip():
                raise ValueError("Text query cannot be empty")

            with torch.no_grad():
                inputs = self._processor(
                    text=[text], return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                features = self._model.get_text_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)

            return features.cpu().numpy().flatten()
        except ValueError:
            raise
        except Exception as e:
            logger.exception(f"Failed to encode text '{text}': {e}")
            raise RuntimeError(f"Text encoding failed: {e}")

    def encode_images_batch(
        self, images: list[Union[Image.Image, str]]
    ) -> np.ndarray:
        """Encode a batch of images into vector embeddings."""
        imgs = [self._load_image(img) for img in images]

        with torch.no_grad():
            inputs = self._processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.cpu().numpy()

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return self._model_name

    @property
    def device(self) -> str:
        """Get the current device."""
        return self._device

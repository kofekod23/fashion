"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from PIL import Image


class BaseModel(ABC):
    """Abstract base class for image/text embedding models."""

    @abstractmethod
    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """Encode an image into a vector embedding."""
        pass

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into a vector embedding."""
        pass

    @abstractmethod
    def encode_images_batch(
        self, images: list[Union[Image.Image, str]]
    ) -> np.ndarray:
        """Encode a batch of images into vector embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name/identifier of the model."""
        pass

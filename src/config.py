"""Configuration management for the fashion search engine."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_default_device() -> str:
    """Determine the default device (cuda if available, else cpu)."""
    device_env = os.getenv("DEVICE", "auto")
    if device_env.lower() == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_env.lower()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Weaviate settings
    weaviate_url: str = field(
        default_factory=lambda: os.getenv(
            "WEAVIATE_URL", "http://localhost:8080"
        )
    )
    weaviate_api_key: str = field(
        default_factory=lambda: os.getenv("WEAVIATE_API_KEY", "")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "FashionCollection")
    )

    # Model settings
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME", "patrickjohncyh/fashion-clip"
        )
    )
    model_type: str = field(
        default_factory=lambda: os.getenv("MODEL_TYPE", "clip")
    )

    # Device settings (cuda, cpu, or auto)
    device: str = field(default_factory=_get_default_device)
    use_half_precision: bool = field(
        default_factory=lambda: os.getenv("USE_HALF_PRECISION", "true").lower() == "true"
    )

    # Indexer settings
    index_workers: int = field(
        default_factory=lambda: int(
            os.getenv("INDEX_WORKERS", str(os.cpu_count() or 4))
        )
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "100"))
    )

    # Image processing settings
    image_size: int = field(
        default_factory=lambda: int(os.getenv("IMAGE_SIZE", "224"))
    )
    thumbnail_size: int = field(
        default_factory=lambda: int(os.getenv("THUMBNAIL_SIZE", "150"))
    )

    # Web server settings
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", "8000"))
    )

    # Supported image extensions
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

    def get_model_class(self):
        """Get the appropriate model class based on configuration."""
        from src.models import CLIPModel, OpenCLIPModel, SigLIPModel

        model_classes = {
            "clip": CLIPModel,
            "openclip": OpenCLIPModel,
            "siglip": SigLIPModel,
        }
        return model_classes.get(self.model_type, CLIPModel)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.weaviate_url:
            errors.append("WEAVIATE_URL is required")
        is_local = (
            "localhost" in self.weaviate_url
            or "127.0.0.1" in self.weaviate_url
            or self.weaviate_url.startswith("http://weaviate")
            or self.weaviate_url.startswith("http://")
        )
        if not is_local and not self.weaviate_api_key:
            errors.append("WEAVIATE_API_KEY is required for Weaviate Cloud")
        return errors

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_cuda_info(self) -> dict:
        """Get CUDA device information."""
        info = {
            "cuda_available": False,
            "device": self.device,
            "gpu_name": None,
            "gpu_memory": None,
        }
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
        except ImportError:
            pass
        return info

    def set_device(self, device: str) -> None:
        """Set the compute device (cuda or cpu)."""
        if device.lower() == "cuda" and not self.is_cuda_available():
            raise ValueError("CUDA is not available on this system")
        self.device = device.lower()


# Global config instance
config = Config()

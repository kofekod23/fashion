"""Multithreaded image indexer for batch processing."""

import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image
from tqdm import tqdm

from src.config import Config
from src.database.weaviate_client import ImageDocument, WeaviateClient
from src.models.base import BaseModel


@dataclass
class IndexingStats:
    """Statistics from an indexing run."""

    total_files: int
    indexed: int
    failed: int
    skipped: int
    elapsed_seconds: float


@dataclass
class ProcessedImage:
    """Result of processing a single image."""

    success: bool
    document: ImageDocument | None = None
    error: str | None = None
    path: str = ""


class ImageIndexer:
    """Multithreaded image indexer for batch processing."""

    def __init__(
        self,
        model: BaseModel,
        db_client: WeaviateClient,
        config: Config,
    ):
        self.model = model
        self.db_client = db_client
        self.config = config
        self.workers = config.index_workers
        self.batch_size = config.batch_size
        self.thumbnail_size = config.thumbnail_size
        self.image_extensions = config.image_extensions

    def _find_images(self, directory: Path) -> list[Path]:
        """Find all image files in a directory recursively."""
        images = []
        for ext in self.image_extensions:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _create_thumbnail_base64(self, image: Image.Image) -> str:
        """Create a base64-encoded thumbnail."""
        img_copy = image.copy()
        img_copy.thumbnail((self.thumbnail_size, self.thumbnail_size))

        if img_copy.mode in ("RGBA", "P"):
            img_copy = img_copy.convert("RGB")

        buffer = io.BytesIO()
        img_copy.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _process_single_image(self, image_path: Path) -> ProcessedImage:
        """Process a single image file."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                thumbnail_b64 = self._create_thumbnail_base64(img)
                vector = self.model.encode_image(img)

            document = ImageDocument(
                filename=image_path.name,
                path=str(image_path.absolute()),
                thumbnail_base64=thumbnail_b64,
                vector=vector,
                width=width,
                height=height,
                indexed_at=datetime.utcnow().isoformat(),
            )

            return ProcessedImage(
                success=True,
                document=document,
                path=str(image_path),
            )

        except Exception as e:
            return ProcessedImage(
                success=False,
                error=str(e),
                path=str(image_path),
            )

    def _batch_iterator(
        self, items: list, batch_size: int
    ) -> Iterator[list]:
        """Yield batches from a list."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def index_directory(
        self,
        directory: str | Path,
        recreate_schema: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexingStats:
        """Index all images in a directory."""
        import time

        start_time = time.time()
        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        image_paths = self._find_images(directory)
        total_files = len(image_paths)

        if total_files == 0:
            return IndexingStats(
                total_files=0, indexed=0, failed=0, skipped=0, elapsed_seconds=0,
            )

        if recreate_schema:
            self.db_client.create_schema(self.model.get_dimension())

        indexed = 0
        failed = 0
        skipped = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_path = {
                executor.submit(self._process_single_image, path): path
                for path in image_paths
            }

            documents_batch = []
            pbar = tqdm(total=total_files, desc="Indexing images")

            for future in as_completed(future_to_path):
                result = future.result()
                pbar.update(1)

                if result.success and result.document:
                    documents_batch.append(result.document)
                else:
                    failed += 1
                    if result.error:
                        tqdm.write(f"Failed: {result.path} - {result.error}")

                if len(documents_batch) >= self.batch_size:
                    self.db_client.insert_batch(documents_batch)
                    indexed += len(documents_batch)
                    documents_batch = []

                if progress_callback:
                    progress_callback(indexed + failed + skipped, total_files)

            if documents_batch:
                self.db_client.insert_batch(documents_batch)
                indexed += len(documents_batch)

            pbar.close()

        elapsed = time.time() - start_time

        return IndexingStats(
            total_files=total_files,
            indexed=indexed,
            failed=failed,
            skipped=skipped,
            elapsed_seconds=elapsed,
        )

    def index_single(self, image_path: str | Path) -> bool:
        """Index a single image."""
        result = self._process_single_image(Path(image_path))
        if result.success and result.document:
            self.db_client.insert_batch([result.document])
            return True
        return False

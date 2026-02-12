#!/usr/bin/env python3
"""Index the ASOS e-commerce dataset from HuggingFace into Weaviate.

Usage:
    python scripts/index_asos.py --max-items 100 --max-images-per-product 1
"""

import argparse
import base64
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_price(price_str: str) -> float | None:
    """Parse ASOS price string like '£45.00' or '$29.99' to float."""
    if not price_str:
        return None
    try:
        cleaned = price_str.strip().replace("£", "").replace("$", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def parse_description(desc_str: str) -> dict:
    """Parse ASOS product description (may be JSON or plain text)."""
    if not desc_str:
        return {"brand": None, "text": None}
    try:
        data = json.loads(desc_str)
        if isinstance(data, dict):
            return {
                "brand": data.get("brand") or data.get("Brand"),
                "text": data.get("description") or data.get("Description") or str(data),
            }
        return {"brand": None, "text": str(data)}
    except (json.JSONDecodeError, TypeError):
        return {"brand": None, "text": str(desc_str)}


def parse_images(images_str: str) -> list[str]:
    """Parse ASOS images field (JSON array of URLs or single URL)."""
    if not images_str:
        return []
    try:
        data = json.loads(images_str)
        if isinstance(data, list):
            return [url for url in data if isinstance(url, str) and url.startswith("http")]
        if isinstance(data, str) and data.startswith("http"):
            return [data]
        return []
    except (json.JSONDecodeError, TypeError):
        if isinstance(images_str, str) and images_str.startswith("http"):
            return [images_str]
        return []


def detect_gender(product_name: str, category: str) -> str | None:
    """Detect gender from product name or category."""
    text = f"{product_name} {category}".lower()
    men_keywords = ["men's", "mens", "male", "homme", " man ", "for men"]
    women_keywords = ["women's", "womens", "female", "femme", " woman ", "for women", "ladies", "maternity"]

    for kw in women_keywords:
        if kw in text:
            return "women"
    for kw in men_keywords:
        if kw in text:
            return "men"
    return None


def download_image(url: str, timeout: int = 10) -> Image.Image | None:
    """Download an image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
        return None


def create_thumbnail_base64(image: Image.Image, size: int = 150) -> str:
    """Create a base64-encoded JPEG thumbnail."""
    img_copy = image.copy()
    img_copy.thumbnail((size, size))
    if img_copy.mode in ("RGBA", "P"):
        img_copy = img_copy.convert("RGB")
    buffer = io.BytesIO()
    img_copy.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_product(item: dict, model, max_images: int = 1, thumbnail_size: int = 150) -> list:
    """Process a single ASOS product and return ImageDocument dicts."""
    from src.database.weaviate_client import ImageDocument

    product_id = str(item.get("id", item.get("Unnamed: 0", "")))
    product_name = item.get("product_name", item.get("name", ""))
    category = item.get("category", item.get("product_type", ""))
    color = item.get("colour", item.get("color", ""))
    price_str = item.get("price", item.get("current_price", ""))
    price = parse_price(str(price_str)) if price_str else None
    product_url = item.get("url", item.get("product_url", ""))

    desc_data = parse_description(item.get("description", ""))
    brand = desc_data["brand"] or item.get("brand", "")
    description = desc_data["text"] or ""

    images_str = item.get("images", item.get("image", ""))
    image_urls = parse_images(str(images_str))

    gender = detect_gender(product_name or "", category or "")

    documents = []

    for idx, url in enumerate(image_urls[:max_images]):
        img = download_image(url)
        if img is None:
            continue

        try:
            width, height = img.size
            thumbnail_b64 = create_thumbnail_base64(img, thumbnail_size)
            vector = model.encode_image(img)

            doc = ImageDocument(
                filename=f"{product_id}_{idx}.jpg",
                path=url,
                thumbnail_base64=thumbnail_b64,
                vector=vector,
                width=width,
                height=height,
                indexed_at=datetime.utcnow().isoformat(),
                product_id=product_id,
                product_name=product_name,
                category=category,
                color=color,
                price=price,
                brand=brand,
                product_url=product_url,
                description=description,
                image_index=idx,
                gender=gender,
            )
            documents.append(doc)
        except Exception as e:
            logger.debug(f"Failed to process image {url}: {e}")

    return documents


def main():
    parser = argparse.ArgumentParser(description="Index ASOS dataset into Weaviate")
    parser.add_argument("--max-items", type=int, default=None, help="Max products to index (default: all)")
    parser.add_argument("--max-images-per-product", type=int, default=1, help="Max images per product (default: 1)")
    parser.add_argument("--batch-size", type=int, default=50, help="Weaviate batch insert size")
    parser.add_argument("--workers", type=int, default=4, help="Download/processing workers")
    args = parser.parse_args()

    print("=" * 60)
    print("ASOS Fashion Dataset Indexer")
    print("=" * 60)

    # Load dataset from HuggingFace
    print("\nLoading ASOS dataset from HuggingFace...")
    from datasets import load_dataset
    dataset = load_dataset("UniqueData/asos-e-commerce-dataset", split="train")
    total_available = len(dataset)
    print(f"Dataset loaded: {total_available} products")

    if args.max_items:
        dataset = dataset.select(range(min(args.max_items, total_available)))
    total_items = len(dataset)
    print(f"Will index: {total_items} products (max {args.max_images_per_product} images each)")

    # Initialize model and database
    print("\nInitializing model...")
    from src.config import config
    model_class = config.get_model_class()
    model = model_class(config.model_name)
    print(f"Model: {config.model_name} (dim={model.get_dimension()})")

    print("\nConnecting to Weaviate...")
    from src.database.weaviate_client import WeaviateClient
    db_client = WeaviateClient(config)
    db_client.connect()
    print("Connected to Weaviate")

    # Create schema
    print("Creating collection schema...")
    db_client.create_schema(model.get_dimension())
    print(f"Collection '{config.collection_name}' created")

    # Process products
    print(f"\nIndexing products...")
    start_time = time.time()
    indexed = 0
    failed = 0
    batch = []

    pbar = tqdm(total=total_items, desc="Indexing")

    for item in dataset:
        try:
            documents = process_product(
                item, model,
                max_images=args.max_images_per_product,
                thumbnail_size=config.thumbnail_size,
            )
            batch.extend(documents)

            if not documents:
                failed += 1

            # Insert batch
            if len(batch) >= args.batch_size:
                db_client.insert_batch(batch)
                indexed += len(batch)
                batch = []

        except Exception as e:
            failed += 1
            logger.debug(f"Failed to process product: {e}")

        pbar.update(1)

    # Insert remaining
    if batch:
        db_client.insert_batch(batch)
        indexed += len(batch)

    pbar.close()

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"Total products processed: {total_items}")
    print(f"Images indexed: {indexed}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.1f}s ({indexed / max(elapsed, 1):.1f} images/sec)")

    # Verify
    stats = db_client.get_stats()
    print(f"\nWeaviate collection '{config.collection_name}': {stats.get('count', 0)} documents")

    db_client.close()


if __name__ == "__main__":
    main()

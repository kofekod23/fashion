"""Weaviate database client for vector storage and search."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ImageDocument:
    """Represents a fashion image document for storage."""

    filename: str
    path: str
    thumbnail_base64: str
    vector: np.ndarray
    width: int | None = None
    height: int | None = None
    indexed_at: str | None = None
    # Fashion metadata
    product_id: str | None = None
    product_name: str | None = None
    category: str | None = None
    color: str | None = None
    size: str | None = None
    price: float | None = None
    brand: str | None = None
    product_url: str | None = None
    description: str | None = None
    image_index: int | None = None
    gender: str | None = None


@dataclass
class SearchResult:
    """Represents a search result."""

    filename: str
    path: str
    thumbnail_base64: str
    score: float
    width: int | None = None
    height: int | None = None
    # Fashion metadata
    product_id: str | None = None
    product_name: str | None = None
    category: str | None = None
    color: str | None = None
    size: str | None = None
    price: float | None = None
    brand: str | None = None
    product_url: str | None = None
    description: str | None = None
    image_index: int | None = None
    gender: str | None = None


@dataclass
class SearchFilters:
    """Filters for fashion search."""

    categories: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    brands: list[str] = field(default_factory=list)
    gender: str | None = None


class WeaviateClient:
    """Client for Weaviate vector database operations."""

    def __init__(self, config: Config):
        self.config = config
        self.collection_name = config.collection_name
        self._client: weaviate.WeaviateClient | None = None

    def connect(self) -> None:
        """Connect to Weaviate (local or cloud)."""
        url = self.config.weaviate_url
        api_key = self.config.weaviate_api_key

        is_local = (
            "localhost" in url
            or "127.0.0.1" in url
            or url.startswith("http://weaviate")
            or url.startswith("http://")
            or not api_key
        )

        try:
            if is_local:
                match = re.match(r"https?://([^:]+):(\d+)", url)
                if match:
                    host = match.group(1)
                    port = int(match.group(2))
                else:
                    host = "localhost"
                    port = 8080

                logger.info(f"Connecting to local Weaviate at {host}:{port}")
                self._client = weaviate.connect_to_local(host=host, port=port)
            else:
                logger.info(f"Connecting to Weaviate Cloud at {url}")
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=weaviate.auth.AuthApiKey(api_key),
                )

            if self._client.is_ready():
                logger.info("Weaviate connection established successfully")
            else:
                raise ConnectionError("Weaviate is not ready")

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise ConnectionError(f"Cannot connect to Weaviate at {url}: {e}")

    def close(self) -> None:
        """Close the Weaviate connection."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Get the Weaviate client, connecting if needed."""
        if self._client is None:
            self.connect()
        return self._client

    def create_schema(self, vector_dimension: int) -> None:
        """Create the fashion collection schema."""
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        self.client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="thumbnail_base64", data_type=DataType.TEXT),
                Property(name="width", data_type=DataType.INT),
                Property(name="height", data_type=DataType.INT),
                Property(name="indexed_at", data_type=DataType.TEXT),
                # Fashion metadata
                Property(name="product_id", data_type=DataType.TEXT),
                Property(name="product_name", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="color", data_type=DataType.TEXT),
                Property(name="size", data_type=DataType.TEXT),
                Property(name="price", data_type=DataType.NUMBER),
                Property(name="brand", data_type=DataType.TEXT),
                Property(name="product_url", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="image_index", data_type=DataType.INT),
                Property(name="gender", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
        )

    def collection_exists(self) -> bool:
        """Check if the fashion collection exists."""
        return self.client.collections.exists(self.collection_name)

    def insert_batch(self, documents: list[ImageDocument]) -> int:
        """Insert a batch of fashion image documents."""
        collection = self.client.collections.get(self.collection_name)
        inserted = 0

        with collection.batch.dynamic() as batch:
            for doc in documents:
                properties = {
                    "filename": doc.filename,
                    "path": doc.path,
                    "thumbnail_base64": doc.thumbnail_base64,
                    "width": doc.width,
                    "height": doc.height,
                    "indexed_at": doc.indexed_at or datetime.utcnow().isoformat(),
                }
                # Add fashion metadata if present
                if doc.product_id is not None:
                    properties["product_id"] = doc.product_id
                if doc.product_name is not None:
                    properties["product_name"] = doc.product_name
                if doc.category is not None:
                    properties["category"] = doc.category
                if doc.color is not None:
                    properties["color"] = doc.color
                if doc.size is not None:
                    properties["size"] = doc.size
                if doc.price is not None:
                    properties["price"] = doc.price
                if doc.brand is not None:
                    properties["brand"] = doc.brand
                if doc.product_url is not None:
                    properties["product_url"] = doc.product_url
                if doc.description is not None:
                    properties["description"] = doc.description
                if doc.image_index is not None:
                    properties["image_index"] = doc.image_index
                if doc.gender is not None:
                    properties["gender"] = doc.gender

                batch.add_object(
                    properties=properties,
                    vector=doc.vector.tolist(),
                )
                inserted += 1

        return inserted

    def _build_filter(self, filters: SearchFilters | None) -> Filter | None:
        """Build Weaviate filter from SearchFilters."""
        if filters is None:
            return None

        conditions = []

        # Category filter
        if filters.categories:
            cat_conditions = [
                Filter.by_property("category").equal(c) for c in filters.categories
            ]
            if len(cat_conditions) == 1:
                conditions.append(cat_conditions[0])
            else:
                combined = cat_conditions[0]
                for cc in cat_conditions[1:]:
                    combined = combined | cc
                conditions.append(combined)

        # Color filter
        if filters.colors:
            color_conditions = [
                Filter.by_property("color").equal(c) for c in filters.colors
            ]
            if len(color_conditions) == 1:
                conditions.append(color_conditions[0])
            else:
                combined = color_conditions[0]
                for cc in color_conditions[1:]:
                    combined = combined | cc
                conditions.append(combined)

        # Price filters
        if filters.min_price is not None:
            conditions.append(
                Filter.by_property("price").greater_or_equal(filters.min_price)
            )
        if filters.max_price is not None:
            conditions.append(
                Filter.by_property("price").less_or_equal(filters.max_price)
            )

        # Brand filter
        if filters.brands:
            brand_conditions = [
                Filter.by_property("brand").equal(b) for b in filters.brands
            ]
            if len(brand_conditions) == 1:
                conditions.append(brand_conditions[0])
            else:
                combined = brand_conditions[0]
                for bc in brand_conditions[1:]:
                    combined = combined | bc
                conditions.append(combined)

        # Gender filter
        if filters.gender:
            conditions.append(
                Filter.by_property("gender").equal(filters.gender)
            )

        if not conditions:
            return None

        combined_filter = conditions[0]
        for condition in conditions[1:]:
            combined_filter = combined_filter & condition

        return combined_filter

    def search_by_vector(
        self,
        vector: np.ndarray,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Search for similar images by vector with optional filters."""
        try:
            if not self.collection_exists():
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return []

            collection = self.client.collections.get(self.collection_name)
            weaviate_filter = self._build_filter(filters)

            if weaviate_filter:
                results = collection.query.near_vector(
                    near_vector=vector.tolist(),
                    limit=limit,
                    filters=weaviate_filter,
                    return_metadata=MetadataQuery(distance=True),
                )
            else:
                results = collection.query.near_vector(
                    near_vector=vector.tolist(),
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                )

            search_results = []
            for obj in results.objects:
                distance = obj.metadata.distance or 0
                score = 1 - distance

                search_results.append(
                    SearchResult(
                        filename=obj.properties.get("filename", ""),
                        path=obj.properties.get("path", ""),
                        thumbnail_base64=obj.properties.get("thumbnail_base64", ""),
                        score=score,
                        width=obj.properties.get("width"),
                        height=obj.properties.get("height"),
                        product_id=obj.properties.get("product_id"),
                        product_name=obj.properties.get("product_name"),
                        category=obj.properties.get("category"),
                        color=obj.properties.get("color"),
                        size=obj.properties.get("size"),
                        price=obj.properties.get("price"),
                        brand=obj.properties.get("brand"),
                        product_url=obj.properties.get("product_url"),
                        description=obj.properties.get("description"),
                        image_index=obj.properties.get("image_index"),
                        gender=obj.properties.get("gender"),
                    )
                )

            return search_results

        except Exception as e:
            logger.exception(f"Search error: {e}")
            raise RuntimeError(f"Search failed: {e}")

    def search_by_text(
        self,
        text_vector: np.ndarray,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Search for images by text query vector."""
        return self.search_by_vector(text_vector, limit, filters)

    def search_by_image(
        self,
        image_vector: np.ndarray,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Search for similar images by image vector."""
        return self.search_by_vector(image_vector, limit, filters)

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            if not self.collection_exists():
                return {"exists": False, "count": 0}

            collection = self.client.collections.get(self.collection_name)
            aggregate = collection.aggregate.over_all(total_count=True)

            return {
                "exists": True,
                "count": aggregate.total_count,
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.exception(f"Failed to get stats: {e}")
            return {"exists": False, "count": 0, "error": str(e)}

    def delete_all(self) -> None:
        """Delete all documents in the collection."""
        if self.collection_exists():
            self.client.collections.delete(self.collection_name)

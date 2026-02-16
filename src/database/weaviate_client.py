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
    occasion: str | None = None


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
    occasion: str | None = None


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

    VECTOR_NAMES = ["fashion_clip", "marqo_clip"]

    def create_schema(self, vector_dimension: int = 0) -> None:
        """Create the fashion collection schema with named vectors."""
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        named_vectors = [
            Configure.NamedVectors.none(
                name=name,
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
            )
            for name in self.VECTOR_NAMES
        ]

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
                Property(name="occasion", data_type=DataType.TEXT),
            ],
            vectorizer_config=named_vectors,
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
                        occasion=obj.properties.get("occasion"),
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

    @staticmethod
    def _obj_to_result(obj, score: float = 0.0) -> SearchResult:
        """Convert a Weaviate object to a SearchResult."""
        return SearchResult(
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
            occasion=obj.properties.get("occasion"),
        )

    @staticmethod
    def _rrf_fusion(
        result_lists: list[list[SearchResult]],
        limit: int,
        k: int = 60,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of multiple result lists."""
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                key = result.product_id or result.filename
                scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
                if key not in result_map:
                    result_map[key] = result

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

        fused = []
        for key in sorted_keys[:limit]:
            r = result_map[key]
            r.score = scores[key]
            fused.append(r)
        return fused

    def search_hybrid(
        self,
        query: str,
        vectors: dict[str, np.ndarray],
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: BM25 (2x weight) + 2 vector models with RRF fusion.

        4 sources in RRF (BM25 counted twice for more keyword influence):
        1-2. Pure BM25 on product_name, category, color, description, occasion
        3. near_vector Fashion CLIP (512d)
        4. near_vector Marqo CLIP (512d)
        """
        try:
            if not self.collection_exists():
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return []

            collection = self.client.collections.get(self.collection_name)
            weaviate_filter = self._build_filter(filters)
            fetch_limit = limit * 3

            result_lists: list[list[SearchResult]] = []

            # 1-2. Pure BM25 keyword search (counted twice in RRF for more weight)
            try:
                bm25_kwargs = dict(
                    query=query,
                    query_properties=[
                        "product_name^3",
                        "category^2",
                        "color^2",
                        "description",
                        "occasion",
                    ],
                    limit=fetch_limit,
                    return_metadata=MetadataQuery(score=True),
                )
                if weaviate_filter:
                    bm25_kwargs["filters"] = weaviate_filter

                bm25_results = collection.query.bm25(**bm25_kwargs)
                bm25_list = [
                    self._obj_to_result(obj, obj.metadata.score or 0)
                    for obj in bm25_results.objects
                ]
                # Add BM25 twice in RRF to give keywords more weight (2/5 vs 3/5)
                result_lists.append(bm25_list)
                result_lists.append(bm25_list)
                logger.info(f"BM25: {len(bm25_results.objects)} results (2x weight in RRF, 2/4)")
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

            # 3-5. near_vector for each model
            for vec_name, vec in vectors.items():
                try:
                    nv_kwargs = dict(
                        near_vector=vec.tolist(),
                        target_vector=vec_name,
                        limit=fetch_limit,
                        return_metadata=MetadataQuery(distance=True),
                    )
                    if weaviate_filter:
                        nv_kwargs["filters"] = weaviate_filter
                    results = collection.query.near_vector(**nv_kwargs)
                    result_lists.append([
                        self._obj_to_result(obj, 1 - (obj.metadata.distance or 0))
                        for obj in results.objects
                    ])
                    logger.info(f"near_vector {vec_name}: {len(results.objects)} results")
                except Exception as e:
                    logger.warning(f"{vec_name} search failed: {e}")

            if not result_lists:
                logger.warning("No search results from any source")
                return []

            # RRF fusion of all 4 sources (BM25 x2 + 2 vectors)
            fused = self._rrf_fusion(result_lists, limit)
            logger.info(f"RRF fusion: {len(result_lists)} sources → {len(fused)} results")
            return fused

        except Exception as e:
            logger.exception(f"Hybrid search error: {e}")
            raise RuntimeError(f"Hybrid search failed: {e}")

    def search_multi_vector(
        self,
        vectors: dict[str, np.ndarray],
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Multi-vector search without BM25 (for image search)."""
        try:
            if not self.collection_exists():
                return []

            collection = self.client.collections.get(self.collection_name)
            weaviate_filter = self._build_filter(filters)
            fetch_limit = limit * 3

            result_lists: list[list[SearchResult]] = []

            for vec_name, vec in vectors.items():
                try:
                    nv_kwargs = dict(
                        near_vector=vec.tolist(),
                        target_vector=vec_name,
                        limit=fetch_limit,
                        return_metadata=MetadataQuery(distance=True),
                    )
                    if weaviate_filter:
                        nv_kwargs["filters"] = weaviate_filter
                    results = collection.query.near_vector(**nv_kwargs)
                    result_lists.append([
                        self._obj_to_result(obj, 1 - (obj.metadata.distance or 0))
                        for obj in results.objects
                    ])
                    logger.info(f"near_vector {vec_name}: {len(results.objects)} results")
                except Exception as e:
                    logger.warning(f"{vec_name} search failed: {e}")

            if not result_lists:
                return []

            return self._rrf_fusion(result_lists, limit)

        except Exception as e:
            logger.exception(f"Multi-vector search error: {e}")
            raise RuntimeError(f"Multi-vector search failed: {e}")

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

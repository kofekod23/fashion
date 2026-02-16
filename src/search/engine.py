"""Search engine combining models and database operations for fashion."""

import logging
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from PIL import Image

from src.database.weaviate_client import SearchFilters, SearchResult, WeaviateClient
from src.models.base import BaseModel

from .query_parser import ParsedQuery, QueryParser

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    """Response from a search query."""

    results: list[SearchResult]
    query_type: str
    total_results: int
    parsed_query: ParsedQuery | None = None
    applied_filters: dict = field(default_factory=dict)


class SearchEngine:
    """Search engine for fashion image retrieval with multi-model hybrid search."""

    def __init__(self, models: dict[str, BaseModel], db_client: WeaviateClient):
        self.models = models
        self.model = models.get("fashion_clip")  # backward compat
        self.db_client = db_client
        self.query_parser = QueryParser()

    def _encode_text_all(self, text: str) -> dict[str, np.ndarray]:
        """Encode text with all available models."""
        vectors = {}
        for name, model in self.models.items():
            try:
                vectors[name] = model.encode_text(text)
            except Exception as e:
                logger.warning(f"Failed to encode text with {name}: {e}")
        return vectors

    def _encode_image_all(self, image: Union[Image.Image, str]) -> dict[str, np.ndarray]:
        """Encode image with all available models."""
        vectors = {}
        for name, model in self.models.items():
            try:
                vectors[name] = model.encode_image(image)
            except Exception as e:
                logger.warning(f"Failed to encode image with {name}: {e}")
        return vectors

    def search_by_text(self, query: str, limit: int = 10) -> SearchResponse:
        """Basic text search with hybrid multi-model."""
        vectors = self._encode_text_all(query)
        results = self.db_client.search_hybrid(
            query=query, vectors=vectors, limit=limit,
        )

        return SearchResponse(
            results=results,
            query_type="text",
            total_results=len(results),
        )

    def smart_search(
        self,
        query: str,
        limit: int = 20,
        gender: str | None = None,
        context: str | None = None,
    ) -> SearchResponse:
        """Smart search with semantic query understanding + hybrid multi-model.

        Combines gender + context + description for CLIP search.
        Also extracts metadata filters (price, color, category).
        Uses BM25 + 2 vector models with RRF fusion.
        """
        # Parse the query with chatbot context
        parsed = self.query_parser.parse_with_context(query, gender=gender, context=context)

        # Build the visual query for CLIP
        visual_parts = []

        # Add gender context
        if gender and gender.lower() not in ("mixte", "unisex"):
            gender_map = {"homme": "men's", "femme": "women's", "men": "men's", "women": "women's"}
            visual_parts.append(gender_map.get(gender.lower(), ""))

        # Add context
        if context:
            context_map = {
                "classique": "classic elegant",
                "bureau": "office formal professional",
                "ceremonie": "ceremony elegant formal evening",
                "sport": "sporty athletic activewear",
                "soiree": "evening party glamorous",
                "casual": "casual everyday relaxed",
            }
            visual_parts.append(context_map.get(context.lower(), context))

        # Add the parsed visual query
        visual_query = parsed.get_visual_query()
        visual_parts.append(visual_query)

        # Combine into final query
        combined_query = " ".join(part for part in visual_parts if part)

        # Encode with all models
        vectors = self._encode_text_all(combined_query)

        # Build filters from parsed query
        filters = None
        applied_filters = {}

        if parsed.has_filters():
            filters = SearchFilters(
                categories=parsed.categories,
                colors=parsed.colors,
                min_price=parsed.min_price,
                max_price=parsed.max_price,
                brands=parsed.brands,
                gender=parsed.gender,
            )

            if parsed.categories:
                applied_filters["categories"] = parsed.categories
            if parsed.colors:
                applied_filters["colors"] = parsed.colors
            if parsed.min_price:
                applied_filters["min_price"] = parsed.min_price
            if parsed.max_price:
                applied_filters["max_price"] = parsed.max_price
            if parsed.brands:
                applied_filters["brands"] = parsed.brands
            if parsed.gender:
                applied_filters["gender"] = parsed.gender

        # Hybrid search: BM25 on combined_query + 2 vectors + RRF
        results = self.db_client.search_hybrid(
            query=combined_query,
            vectors=vectors,
            limit=limit,
            filters=filters,
        )

        return SearchResponse(
            results=results,
            query_type="smart",
            total_results=len(results),
            parsed_query=parsed,
            applied_filters=applied_filters,
        )

    def search_by_image(
        self,
        image: Union[Image.Image, str],
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        """Search for similar fashion items using an image (multi-vector, no BM25)."""
        vectors = self._encode_image_all(image)
        results = self.db_client.search_multi_vector(
            vectors=vectors, limit=limit, filters=filters,
        )

        return SearchResponse(
            results=results,
            query_type="image",
            total_results=len(results),
        )

    def get_stats(self) -> dict:
        """Get search engine statistics."""
        db_stats = self.db_client.get_stats()

        model_names = {name: m.model_name for name, m in self.models.items()}
        primary = self.model

        return {
            **db_stats,
            "model_name": primary.model_name if primary else "multi-model",
            "vector_dimension": primary.get_dimension() if primary else 0,
            "models": model_names,
        }

"""Search engine combining model and database operations for fashion."""

from dataclasses import dataclass, field
from typing import Union

from PIL import Image

from src.database.weaviate_client import SearchFilters, SearchResult, WeaviateClient
from src.models.base import BaseModel

from .query_parser import ParsedQuery, QueryParser


@dataclass
class SearchResponse:
    """Response from a search query."""

    results: list[SearchResult]
    query_type: str
    total_results: int
    parsed_query: ParsedQuery | None = None
    applied_filters: dict = field(default_factory=dict)


class SearchEngine:
    """Search engine for fashion image retrieval with semantic query understanding."""

    def __init__(self, model: BaseModel, db_client: WeaviateClient):
        self.model = model
        self.db_client = db_client
        self.query_parser = QueryParser()

    def search_by_text(self, query: str, limit: int = 10) -> SearchResponse:
        """Basic text search without smart parsing."""
        text_vector = self.model.encode_text(query)
        results = self.db_client.search_by_text(text_vector, limit)

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
        """Smart search with semantic query understanding.

        Combines gender + context + description for Fashion CLIP search.
        Also extracts metadata filters (price, color, category).
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

        # Combine into final CLIP query
        combined_query = " ".join(part for part in visual_parts if part)

        # Encode to vector
        text_vector = self.model.encode_text(combined_query)

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

        # Search with filters
        results = self.db_client.search_by_text(text_vector, limit, filters)

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
        """Search for similar fashion items using an image."""
        image_vector = self.model.encode_image(image)
        results = self.db_client.search_by_image(image_vector, limit, filters)

        return SearchResponse(
            results=results,
            query_type="image",
            total_results=len(results),
        )

    def get_stats(self) -> dict:
        """Get search engine statistics."""
        db_stats = self.db_client.get_stats()

        return {
            **db_stats,
            "model_name": self.model.model_name,
            "vector_dimension": self.model.get_dimension(),
        }

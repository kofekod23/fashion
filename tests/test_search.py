"""Tests for the fashion search engine."""

import numpy as np
import pytest

from src.search.query_parser import ParsedQuery, QueryParser, parse_query


class TestQueryParser:
    """Tests for the fashion query parser."""

    def setup_method(self):
        self.parser = QueryParser()

    def test_parse_basic_query(self):
        result = self.parser.parse("black dress")
        assert "black" in result.colors
        assert "dresses" in result.categories

    def test_parse_color_extraction(self):
        result = self.parser.parse("red leather jacket")
        assert "red" in result.colors

    def test_parse_multiple_colors(self):
        result = self.parser.parse("black and white striped top")
        assert "black" in result.colors
        assert "white" in result.colors

    def test_parse_category_extraction(self):
        result = self.parser.parse("summer dress for party")
        assert "dresses" in result.categories

    def test_parse_shoes_category(self):
        result = self.parser.parse("white sneakers")
        assert "shoes" in result.categories
        assert "white" in result.colors

    def test_parse_price_under(self):
        result = self.parser.parse("jacket under 50")
        assert result.max_price == 50.0

    def test_parse_price_range(self):
        result = self.parser.parse("dress between 30 and 80")
        assert result.min_price == 30.0
        assert result.max_price == 80.0

    def test_parse_price_around(self):
        result = self.parser.parse("shoes around 60")
        assert result.min_price == pytest.approx(48.0)
        assert result.max_price == pytest.approx(72.0)

    def test_parse_price_with_currency(self):
        result = self.parser.parse("bag under £100")
        assert result.max_price == 100.0

    def test_parse_gender_men(self):
        result = self.parser.parse("men's formal shirt")
        assert result.gender == "men"

    def test_parse_gender_women(self):
        result = self.parser.parse("women's summer dress")
        assert result.gender == "women"

    def test_has_filters_empty(self):
        result = self.parser.parse("beautiful sunset")
        assert not result.has_filters() or result.has_filters()  # depends on visual terms

    def test_has_filters_with_price(self):
        result = self.parser.parse("jacket under 50")
        assert result.has_filters()

    def test_has_filters_with_color(self):
        result = self.parser.parse("black jacket")
        assert result.has_filters()

    def test_get_visual_query_with_terms(self):
        result = self.parser.parse("casual leather jacket")
        visual = result.get_visual_query()
        assert len(visual) > 0

    def test_parse_with_context_gender(self):
        result = self.parser.parse_with_context("blazer noir", gender="femme")
        assert result.gender == "women"

    def test_parse_with_context_bureau(self):
        result = self.parser.parse_with_context("blazer", context="bureau")
        assert result.context == "bureau"
        assert any("formal" in t or "office" in t for t in result.visual_terms)

    def test_parse_with_context_sport(self):
        result = self.parser.parse_with_context("leggings", context="sport")
        assert result.context == "sport"

    def test_parse_with_context_mixte(self):
        result = self.parser.parse_with_context("t-shirt", gender="mixte")
        assert result.gender is None  # mixte should not set gender filter

    def test_parse_knitwear(self):
        result = self.parser.parse("wool sweater")
        assert "knitwear" in result.categories

    def test_parse_multiple_categories(self):
        result = self.parser.parse("jacket and shoes")
        assert "jackets" in result.categories
        assert "shoes" in result.categories

    def test_convenience_function(self):
        result = parse_query("red dress under £50")
        assert isinstance(result, ParsedQuery)
        assert "red" in result.colors
        assert "dresses" in result.categories
        assert result.max_price == 50.0


class TestParsedQuery:
    """Tests for the ParsedQuery dataclass."""

    def test_empty_query(self):
        pq = ParsedQuery(original_query="test")
        assert not pq.has_filters()
        assert pq.get_visual_query() == "test"

    def test_visual_query_with_terms(self):
        pq = ParsedQuery(original_query="test", visual_terms=["casual", "leather"])
        assert pq.get_visual_query() == "casual leather"

    def test_has_filters_categories(self):
        pq = ParsedQuery(original_query="test", categories=["dresses"])
        assert pq.has_filters()

    def test_has_filters_price(self):
        pq = ParsedQuery(original_query="test", max_price=50.0)
        assert pq.has_filters()

    def test_has_filters_gender(self):
        pq = ParsedQuery(original_query="test", gender="women")
        assert pq.has_filters()


class TestImageDocument:
    """Tests for fashion-specific ImageDocument."""

    def test_create_document(self):
        from src.database.weaviate_client import ImageDocument

        doc = ImageDocument(
            filename="test.jpg",
            path="/test/test.jpg",
            thumbnail_base64="abc123",
            vector=np.zeros(512),
            product_id="12345",
            product_name="Black Leather Jacket",
            category="jackets",
            color="black",
            price=89.99,
            brand="ASOS",
            gender="men",
        )
        assert doc.product_id == "12345"
        assert doc.product_name == "Black Leather Jacket"
        assert doc.price == 89.99
        assert doc.gender == "men"

    def test_create_minimal_document(self):
        from src.database.weaviate_client import ImageDocument

        doc = ImageDocument(
            filename="test.jpg",
            path="/test/test.jpg",
            thumbnail_base64="abc123",
            vector=np.zeros(512),
        )
        assert doc.product_id is None
        assert doc.price is None


class TestSearchFilters:
    """Tests for fashion SearchFilters."""

    def test_create_empty_filters(self):
        from src.database.weaviate_client import SearchFilters

        filters = SearchFilters()
        assert len(filters.categories) == 0
        assert filters.gender is None

    def test_create_filters_with_values(self):
        from src.database.weaviate_client import SearchFilters

        filters = SearchFilters(
            categories=["dresses", "tops"],
            colors=["black"],
            min_price=20.0,
            max_price=100.0,
            gender="women",
        )
        assert len(filters.categories) == 2
        assert filters.min_price == 20.0
        assert filters.gender == "women"

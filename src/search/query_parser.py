"""Query parser for fashion search queries.

Translates queries like:
- "black leather jacket" -> visual search + color filter
- "dress under £50" -> visual search + price filter
- "casual summer outfit" -> visual search with fashion context
"""

import re
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    """Parsed fashion search query with extracted requirements."""

    original_query: str

    # Visual search terms (for CLIP embedding)
    visual_terms: list[str] = field(default_factory=list)

    # Metadata filters
    categories: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    brands: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    gender: str | None = None
    context: str | None = None

    def has_filters(self) -> bool:
        """Check if any metadata filters are set."""
        return any([
            len(self.categories) > 0,
            len(self.colors) > 0,
            self.min_price is not None,
            self.max_price is not None,
            len(self.brands) > 0,
            self.gender is not None,
        ])

    def get_visual_query(self) -> str:
        """Get the query string for visual/CLIP search."""
        if self.visual_terms:
            return " ".join(self.visual_terms)
        return self.original_query


class QueryParser:
    """Parser for natural language fashion queries."""

    # Color keywords
    COLORS = [
        "black", "white", "red", "blue", "navy", "pink", "green", "brown",
        "beige", "grey", "gray", "cream", "burgundy", "maroon", "orange",
        "yellow", "purple", "violet", "khaki", "olive", "teal", "coral",
        "gold", "silver", "tan", "ivory", "charcoal", "multicolour",
        "multi", "nude", "rose", "mint", "lilac", "turquoise", "camel",
    ]

    # Category keywords
    CATEGORIES = {
        "dresses": ["dress", "dresses", "gown", "maxi", "midi"],
        "tops": ["top", "tops", "blouse", "shirt", "t-shirt", "tee", "camisole", "crop top", "vest"],
        "trousers": ["trousers", "pants", "jeans", "chinos", "leggings", "joggers"],
        "shoes": ["shoes", "boots", "trainers", "sneakers", "heels", "sandals", "loafers", "flats", "pumps"],
        "jackets": ["jacket", "jackets", "coat", "coats", "blazer", "parka", "bomber", "puffer", "windbreaker"],
        "bags": ["bag", "bags", "handbag", "backpack", "clutch", "tote", "crossbody", "purse"],
        "knitwear": ["knitwear", "jumper", "sweater", "cardigan", "pullover", "hoodie"],
        "shorts": ["shorts"],
        "suits": ["suit", "suits", "tuxedo"],
        "skirts": ["skirt", "skirts"],
        "swimwear": ["swimwear", "bikini", "swimsuit", "swimming"],
        "accessories": ["scarf", "hat", "belt", "gloves", "sunglasses", "watch", "jewellery", "jewelry", "necklace", "bracelet", "earrings"],
        "underwear": ["underwear", "lingerie", "bra", "briefs", "boxers", "socks"],
        "activewear": ["activewear", "sportswear", "gym", "running", "yoga", "leotard"],
    }

    # Context -> visual style keywords
    CONTEXT_KEYWORDS = {
        "classique": ["classic", "timeless", "elegant", "refined", "tailored"],
        "bureau": ["formal", "office", "professional", "blazer", "tailored", "smart"],
        "ceremonie": ["elegant", "formal", "dress", "suit", "glamorous", "evening", "gown"],
        "sport": ["activewear", "sporty", "athletic", "gym", "running", "performance"],
        "soiree": ["evening", "party", "glamorous", "cocktail", "sequin", "satin", "sparkle"],
        "casual": ["casual", "relaxed", "everyday", "comfortable", "streetwear", "denim"],
    }

    # Gender keywords
    GENDER_KEYWORDS = {
        "men": ["men", "man", "male", "homme", "mens", "men's", "masculine", "guy"],
        "women": ["women", "woman", "female", "femme", "womens", "women's", "feminine", "lady", "ladies"],
    }

    # Price patterns (£ currency)
    PRICE_PATTERNS = [
        (r"(?:under|less\s+than|max(?:imum)?|budget(?:\s+of)?)\s*[£$]?\s*([\d,.]+)", "max_price"),
        (r"(?:over|more\s+than|min(?:imum)?|at\s+least)\s*[£$]?\s*([\d,.]+)", "min_price"),
        (r"between\s*[£$]?\s*([\d,.]+)\s*(?:and|-)\s*[£$]?\s*([\d,.]+)", "price_range"),
        (r"(?:around|about|approximately)\s*[£$]?\s*([\d,.]+)", "price_around"),
        (r"[£$]\s*([\d,.]+)", "max_price_hint"),
    ]

    # Visual/descriptive keywords for fashion
    VISUAL_KEYWORDS = [
        # Styles
        "casual", "formal", "elegant", "sporty", "bohemian", "boho", "vintage",
        "retro", "minimalist", "classic", "modern", "trendy", "chic", "edgy",
        "preppy", "grunge", "punk", "romantic", "gothic", "streetwear", "luxury",
        # Materials
        "leather", "silk", "satin", "cotton", "denim", "linen", "wool", "velvet",
        "suede", "lace", "chiffon", "cashmere", "nylon", "polyester", "tweed",
        "corduroy", "mesh", "knit", "faux fur", "fur",
        # Patterns
        "floral", "striped", "stripes", "plaid", "checkered", "polka dot",
        "animal print", "leopard", "camouflage", "camo", "paisley", "geometric",
        "abstract", "tie dye", "herringbone", "houndstooth", "tartan", "gingham",
        # Fits & cuts
        "oversized", "slim", "slim fit", "fitted", "loose", "baggy", "cropped",
        "high waist", "low rise", "wide leg", "skinny", "straight", "bootcut",
        "flared", "tapered", "a-line", "wrap", "bodycon", "pleated",
        # Details
        "embroidered", "sequin", "beaded", "ruffled", "fringe", "buttoned",
        "zip", "hooded", "collar", "v-neck", "crew neck", "off shoulder",
        "strapless", "sleeveless", "long sleeve", "short sleeve", "pockets",
        # Seasons / occasions
        "summer", "winter", "spring", "autumn", "festival", "beach", "wedding",
        "party", "office", "work", "date night", "holiday", "travel",
    ]

    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language fashion query."""
        query_lower = query.lower()
        result = ParsedQuery(original_query=query)

        self._extract_colors(query_lower, result)
        self._extract_categories(query_lower, result)
        self._extract_price_constraints(query_lower, result)
        self._extract_gender(query_lower, result)
        self._extract_visual_terms(query_lower, result)

        return result

    def parse_with_context(self, query: str, gender: str | None = None, context: str | None = None) -> ParsedQuery:
        """Parse a query with chatbot-provided gender and context."""
        result = self.parse(query)

        # Apply gender from chatbot
        if gender and gender.lower() not in ("mixte", "unisex"):
            gender_map = {"homme": "men", "femme": "women", "men": "men", "women": "women"}
            result.gender = gender_map.get(gender.lower(), gender.lower())

        # Apply context
        if context:
            result.context = context.lower()
            # Add context visual terms
            context_terms = self.CONTEXT_KEYWORDS.get(context.lower(), [])
            if context_terms:
                result.visual_terms = context_terms + result.visual_terms

        return result

    def _extract_colors(self, query: str, result: ParsedQuery) -> None:
        """Extract color keywords."""
        for color in self.COLORS:
            if re.search(r"\b" + re.escape(color) + r"\b", query):
                result.colors.append(color)

    def _extract_categories(self, query: str, result: ParsedQuery) -> None:
        """Extract product categories."""
        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query):
                    if category not in result.categories:
                        result.categories.append(category)
                    break

    def _extract_price_constraints(self, query: str, result: ParsedQuery) -> None:
        """Extract price constraints."""
        for pattern, constraint_type in self.PRICE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if constraint_type == "max_price":
                    result.max_price = self._parse_price(match.group(1))
                elif constraint_type == "min_price":
                    result.min_price = self._parse_price(match.group(1))
                elif constraint_type == "price_range":
                    result.min_price = self._parse_price(match.group(1))
                    result.max_price = self._parse_price(match.group(2))
                elif constraint_type == "price_around":
                    price = self._parse_price(match.group(1))
                    result.min_price = price * 0.8
                    result.max_price = price * 1.2
                elif constraint_type == "max_price_hint":
                    # Only use if no other price was set
                    if result.max_price is None and result.min_price is None:
                        result.max_price = self._parse_price(match.group(1))

    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float."""
        price_str = price_str.replace(",", "")
        return float(price_str)

    def _extract_gender(self, query: str, result: ParsedQuery) -> None:
        """Extract gender from query."""
        for gender, keywords in self.GENDER_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", query):
                    result.gender = gender
                    return

    def _extract_visual_terms(self, query: str, result: ParsedQuery) -> None:
        """Extract visual/descriptive terms for CLIP search."""
        found_visual = []
        for keyword in self.VISUAL_KEYWORDS:
            if keyword in query:
                found_visual.append(keyword)

        if found_visual:
            result.visual_terms = found_visual + result.visual_terms
        else:
            # Use cleaned query
            clean_query = query
            # Remove price mentions
            clean_query = re.sub(r"[£$]?\s*[\d,.]+", "", clean_query)
            # Remove common filter words
            filter_words = ["under", "over", "less than", "more than", "at least",
                           "minimum", "maximum", "budget", "between", "around",
                           "for", "men", "women", "homme", "femme", "mixte"]
            for word in filter_words:
                clean_query = clean_query.replace(word, "")
            clean_query = " ".join(clean_query.split())

            if clean_query.strip():
                result.visual_terms = [clean_query.strip()] + result.visual_terms


def parse_query(query: str) -> ParsedQuery:
    """Parse a natural language fashion query."""
    parser = QueryParser()
    return parser.parse(query)

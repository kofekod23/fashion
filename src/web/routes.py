"""API routes for the fashion search engine."""

import asyncio
import base64
import io
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
import weaviate.classes.query
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from .app import get_db_client, get_model, get_search_engine

logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path("/app/data")
QUERY_LOG_FILE = DATA_DIR / "query_log.jsonl"
EXAMPLES_FILE = DATA_DIR / "examples.json"
SETTINGS_FILE = DATA_DIR / "site_settings.json"
FAQ_FILE = DATA_DIR / "faq.json"
STATIC_DIR = DATA_DIR / "static"
ADMIN_PASSWORD = "ROY"

# Default examples (fashion)
DEFAULT_EXAMPLES = [
    "black leather jacket",
    "summer floral dress",
    "casual white sneakers",
    "formal blazer for office",
    "elegant evening gown",
    "denim jeans slim fit"
]

# Default site settings
DEFAULT_SETTINGS = {
    "site_title": "AI Fashion Search",
    "site_subtitle": "Find your perfect outfit with AI-powered visual search",
    "footer_text": "Created by Julien G. with the help of AI",
    "primary_color": "#2c3e50",
    "secondary_color": "#8e44ad",
    "accent_color": "#e74c3c",
    "logo_url": "",
    "favicon_url": ""
}

# Default FAQ content
DEFAULT_FAQ = {
    "content": """<h2>Questions frequentes</h2>
<h3>Comment fonctionne la recherche ?</h3>
<p>Notre recherche IA utilise Fashion CLIP pour comprendre vos descriptions de vetements et trouver les articles les plus pertinents visuellement.</p>

<h3>Quels filtres sont disponibles ?</h3>
<p>Vous pouvez filtrer par categorie, couleur, prix et genre. Le chatbot vous guide pour affiner votre recherche.</p>

<h3>Puis-je chercher avec une image ?</h3>
<p>Oui ! Utilisez le bouton de recherche par image pour uploader une photo et trouver des articles similaires.</p>
"""
}


async def get_geolocation(ip: str) -> dict | None:
    """Get geolocation data for an IP address."""
    if ip in ("127.0.0.1", "localhost", "::1") or ip.startswith(("10.", "192.168.", "172.")):
        return None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,region,regionName,city,zip,lat,lon,isp")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "country": data.get("country"),
                        "country_code": data.get("countryCode"),
                        "region": data.get("regionName"),
                        "city": data.get("city"),
                        "zip": data.get("zip"),
                        "lat": data.get("lat"),
                        "lon": data.get("lon"),
                        "isp": data.get("isp"),
                    }
    except Exception as e:
        logger.warning(f"Failed to get geolocation for {ip}: {e}")
    return None


def log_query(query: str, results_count: int, query_type: str = "text", ip: str | None = None, geolocation: dict | None = None):
    """Log a search query to the log file."""
    try:
        QUERY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "query_type": query_type,
            "results_count": results_count,
        }
        if ip:
            log_entry["ip"] = ip
        if geolocation:
            log_entry["geolocation"] = geolocation
        with open(QUERY_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")


def get_examples() -> list[str]:
    """Get the current example queries."""
    try:
        if EXAMPLES_FILE.exists():
            with open(EXAMPLES_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load examples: {e}")
    return DEFAULT_EXAMPLES


def save_examples(examples: list[str]):
    """Save example queries to file."""
    EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXAMPLES_FILE, "w") as f:
        json.dump(examples, f, indent=2)


def get_site_settings() -> dict:
    """Get the current site settings."""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                return {**DEFAULT_SETTINGS, **saved}
    except Exception as e:
        logger.warning(f"Failed to load site settings: {e}")
    return DEFAULT_SETTINGS.copy()


def save_site_settings(settings: dict):
    """Save site settings to file."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def get_faq() -> dict:
    """Get the current FAQ content."""
    try:
        if FAQ_FILE.exists():
            with open(FAQ_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load FAQ: {e}")
    return DEFAULT_FAQ.copy()


def save_faq(faq: dict):
    """Save FAQ content to file."""
    FAQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FAQ_FILE, "w") as f:
        json.dump(faq, f, indent=2)


# Thread pool for running blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


async def run_in_executor(func, *args, **kwargs):
    """Run a blocking function in a thread pool executor."""
    loop = asyncio.get_running_loop()
    if kwargs:
        func_to_call = partial(func, *args, **kwargs)
    elif args:
        func_to_call = partial(func, *args)
    else:
        func_to_call = func
    return await loop.run_in_executor(_executor, func_to_call)


# Reindex status tracking
_reindex_status = {
    "running": False,
    "complete": False,
    "progress": 0,
    "message": "Not started",
    "indexed": 0,
}

router = APIRouter()


class TextSearchRequest(BaseModel):
    """Request body for text search."""

    query: str
    limit: int = 20
    smart: bool = True
    gender: str | None = None
    context: str | None = None


class SearchResultResponse(BaseModel):
    """Single search result in response."""

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
    gender: str | None = None


class AppliedFiltersResponse(BaseModel):
    """Applied filters in smart search."""

    categories: list[str] | None = None
    colors: list[str] | None = None
    min_price: float | None = None
    max_price: float | None = None
    brands: list[str] | None = None
    gender: str | None = None


class SearchResponse(BaseModel):
    """Response for search endpoints."""

    results: list[SearchResultResponse]
    query_type: str
    total_results: int
    visual_query: str | None = None
    applied_filters: AppliedFiltersResponse | None = None


class DeviceInfo(BaseModel):
    """Device/GPU information."""

    device: str
    cuda_available: bool
    gpu_name: str | None = None
    gpu_memory: str | None = None


class StatsResponse(BaseModel):
    """Response for stats endpoint."""

    exists: bool
    count: int
    collection_name: str | None = None
    model_name: str | None = None
    vector_dimension: int | None = None
    device_info: DeviceInfo | None = None


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the search page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/search", response_model=SearchResponse)
async def search_by_text(search_request: TextSearchRequest, request: Request):
    """Search for fashion items using text."""
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.client.host
    logger.info(f"Search: query='{search_request.query}' gender={search_request.gender} context={search_request.context} from {client_ip}")
    try:
        engine = get_search_engine()

        if search_request.smart:
            response = await run_in_executor(
                engine.smart_search,
                search_request.query,
                search_request.limit,
                search_request.gender,
                search_request.context,
            )
        else:
            response = await run_in_executor(engine.search_by_text, search_request.query, search_request.limit)

        geolocation = await get_geolocation(client_ip)
        log_query(search_request.query, response.total_results, "smart" if search_request.smart else "text", ip=client_ip, geolocation=geolocation)

        applied_filters = None
        visual_query = None

        if response.parsed_query:
            visual_query = response.parsed_query.get_visual_query()

        if response.applied_filters:
            applied_filters = AppliedFiltersResponse(
                categories=response.applied_filters.get("categories"),
                colors=response.applied_filters.get("colors"),
                min_price=response.applied_filters.get("min_price"),
                max_price=response.applied_filters.get("max_price"),
                brands=response.applied_filters.get("brands"),
                gender=response.applied_filters.get("gender"),
            )

        return SearchResponse(
            results=[
                SearchResultResponse(
                    filename=r.filename,
                    path=r.path,
                    thumbnail_base64=r.thumbnail_base64,
                    score=r.score,
                    width=r.width,
                    height=r.height,
                    product_id=r.product_id,
                    product_name=r.product_name,
                    category=r.category,
                    color=r.color,
                    size=r.size,
                    price=r.price,
                    brand=r.brand,
                    product_url=r.product_url,
                    description=r.description,
                    gender=r.gender,
                )
                for r in response.results
            ],
            query_type=response.query_type,
            total_results=response.total_results,
            visual_query=visual_query,
            applied_filters=applied_filters,
        )

    except Exception as e:
        logger.exception(f"Search error for query '{search_request.query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}. Please check that Weaviate is running and the collection is indexed.",
        )


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    image: Annotated[UploadFile, File()],
    limit: Annotated[int, Form()] = 10,
):
    """Search for similar fashion items using an uploaded image."""
    try:
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        engine = get_search_engine()
        response = await run_in_executor(engine.search_by_image, pil_image, limit)

        return SearchResponse(
            results=[
                SearchResultResponse(
                    filename=r.filename,
                    path=r.path,
                    thumbnail_base64=r.thumbnail_base64,
                    score=r.score,
                    width=r.width,
                    height=r.height,
                    product_id=r.product_id,
                    product_name=r.product_name,
                    category=r.category,
                    color=r.color,
                    size=r.size,
                    price=r.price,
                    brand=r.brand,
                    product_url=r.product_url,
                    description=r.description,
                    gender=r.gender,
                )
                for r in response.results
            ],
            query_type=response.query_type,
            total_results=response.total_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Image search error: {e}")
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get index statistics."""
    try:
        from src.config import config

        engine = get_search_engine()
        stats = await run_in_executor(engine.get_stats)

        cuda_info = config.get_cuda_info()
        device_info = DeviceInfo(
            device=cuda_info["device"],
            cuda_available=cuda_info["cuda_available"],
            gpu_name=cuda_info.get("gpu_name"),
            gpu_memory=cuda_info.get("gpu_memory"),
        )

        return StatsResponse(
            exists=stats.get("exists", False),
            count=stats.get("count", 0),
            collection_name=stats.get("collection_name"),
            model_name=stats.get("model_name"),
            vector_dimension=stats.get("vector_dimension"),
            device_info=device_info,
        )

    except Exception as e:
        logger.exception(f"Stats error: {e}")
        from src.config import config
        cuda_info = config.get_cuda_info()
        return StatsResponse(
            exists=False, count=0,
            device_info=DeviceInfo(
                device=cuda_info.get("device", "cpu"),
                cuda_available=cuda_info.get("cuda_available", False),
            ),
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ==================== PRODUCT DETAIL ====================

class ProductPhotoResponse(BaseModel):
    """Single photo in product response."""
    filename: str
    thumbnail_base64: str
    image_index: int | None = None


class ProductDetailResponse(BaseModel):
    """Response for product detail endpoint."""
    product_id: str
    photos: list[ProductPhotoResponse]
    product_name: str | None = None
    category: str | None = None
    color: str | None = None
    size: str | None = None
    price: float | None = None
    brand: str | None = None
    product_url: str | None = None
    description: str | None = None
    gender: str | None = None


@router.get("/product/{product_id}", response_model=ProductDetailResponse)
async def get_product_photos(product_id: str):
    """Get all photos and details for a specific product."""
    try:
        db_client = get_db_client()
        collection = db_client.client.collections.get(db_client.collection_name)

        result = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("product_id").equal(product_id),
            limit=10,
            return_properties=[
                "filename", "thumbnail_base64", "image_index",
                "product_name", "category", "color", "size",
                "price", "brand", "product_url", "description", "gender",
            ],
        )

        if not result.objects:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

        photos = []
        product_data = {}

        for obj in result.objects:
            props = obj.properties
            photos.append(
                ProductPhotoResponse(
                    filename=props.get("filename", ""),
                    thumbnail_base64=props.get("thumbnail_base64", ""),
                    image_index=props.get("image_index"),
                )
            )
            if not product_data:
                product_data = {
                    "product_name": props.get("product_name"),
                    "category": props.get("category"),
                    "color": props.get("color"),
                    "size": props.get("size"),
                    "price": props.get("price"),
                    "brand": props.get("brand"),
                    "product_url": props.get("product_url"),
                    "description": props.get("description"),
                    "gender": props.get("gender"),
                }

        photos.sort(key=lambda p: p.image_index or 0)

        return ProductDetailResponse(
            product_id=product_id,
            photos=photos,
            **product_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get product photos: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get product: {str(e)}")


# ==================== FEATURED PRODUCTS ====================

class FeaturedProductResponse(BaseModel):
    """Single featured product in response."""
    product_id: str
    thumbnail_base64: str
    product_name: str | None = None
    price: float | None = None
    brand: str | None = None
    category: str | None = None


@router.get("/featured", response_model=list[FeaturedProductResponse])
async def get_featured_products(limit: int = 20):
    """Get random products for the homepage (first image only)."""
    try:
        import random
        db_client = get_db_client()
        collection = db_client.client.collections.get(db_client.collection_name)

        # Fetch products where image_index=0 (main image)
        result = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("image_index").equal(0),
            limit=100,
            return_properties=["product_id", "thumbnail_base64", "product_name", "price", "brand", "category"],
        )

        if not result.objects:
            return []

        products = list(result.objects)
        random.shuffle(products)
        selected = products[:limit]

        return [
            FeaturedProductResponse(
                product_id=obj.properties.get("product_id", ""),
                thumbnail_base64=obj.properties.get("thumbnail_base64", ""),
                product_name=obj.properties.get("product_name"),
                price=obj.properties.get("price"),
                brand=obj.properties.get("brand"),
                category=obj.properties.get("category"),
            )
            for obj in selected
        ]

    except Exception as e:
        logger.exception(f"Failed to get featured products: {e}")
        return []


# ==================== ADMIN ENDPOINTS ====================

class ExamplesRequest(BaseModel):
    """Request body for updating examples."""
    password: str
    examples: list[str]


@router.get("/admin/examples")
async def get_example_queries():
    """Get the current example search queries."""
    return {"examples": get_examples()}


@router.post("/admin/examples")
async def update_example_queries(request: ExamplesRequest):
    """Update the example search queries (password protected)."""
    if request.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    if not request.examples or len(request.examples) == 0:
        raise HTTPException(status_code=400, detail="At least one example is required")
    save_examples(request.examples)
    return {"status": "success", "examples": request.examples}


@router.get("/admin/logs")
async def get_query_logs(password: str, limit: int = 100):
    """Get recent query logs (password protected)."""
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    logs = []
    try:
        if QUERY_LOG_FILE.exists():
            with open(QUERY_LOG_FILE, "r") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to read query logs: {e}")
    logs.reverse()
    return {"logs": logs, "total": len(logs)}


# ==================== SITE SETTINGS ====================

class SiteSettingsRequest(BaseModel):
    """Request body for updating site settings."""
    password: str
    settings: dict


@router.get("/admin/settings")
async def get_settings():
    """Get the current site settings."""
    return {"settings": get_site_settings()}


@router.post("/admin/settings")
async def update_settings(request: SiteSettingsRequest):
    """Update site settings (password protected)."""
    if request.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    current = get_site_settings()
    current.update(request.settings)
    save_site_settings(current)
    return {"status": "success", "settings": current}


@router.post("/admin/upload/{file_type}")
async def upload_file(
    file_type: str,
    password: Annotated[str, Form()],
    file: Annotated[UploadFile, File()]
):
    """Upload logo or favicon (password protected)."""
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    if file_type not in ["logo", "favicon"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use 'logo' or 'favicon'")

    allowed_types = ["image/png", "image/jpeg", "image/gif", "image/svg+xml", "image/x-icon", "image/vnd.microsoft.icon"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ext_map = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
        "image/svg+xml": ".svg", "image/x-icon": ".ico", "image/vnd.microsoft.icon": ".ico"
    }
    ext = ext_map.get(file.content_type, ".png")
    filename = f"{file_type}{ext}"
    filepath = STATIC_DIR / filename

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    url = f"/static/{filename}"
    settings = get_site_settings()
    settings[f"{file_type}_url"] = url
    save_site_settings(settings)

    return {"status": "success", "url": url, "filename": filename}


@router.get("/static/{filename}")
async def serve_static(filename: str):
    """Serve uploaded static files."""
    from fastapi.responses import FileResponse
    filepath = STATIC_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)


# ==================== FAQ ====================

class FAQRequest(BaseModel):
    """Request body for updating FAQ."""
    password: str
    content: str


@router.get("/faq")
async def get_faq_content():
    """Get the current FAQ content."""
    return get_faq()


@router.post("/admin/faq")
async def update_faq(request: FAQRequest):
    """Update FAQ content (password protected)."""
    if request.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    faq = {"content": request.content}
    save_faq(faq)
    return {"status": "success", "content": request.content}


# ==================== REINDEX ====================

def run_reindex_task():
    """Background task to re-index all images."""
    global _reindex_status
    try:
        _reindex_status["running"] = True
        _reindex_status["complete"] = False
        _reindex_status["progress"] = 5
        _reindex_status["message"] = "Loading model..."

        model = get_model()
        db_client = get_db_client()

        _reindex_status["progress"] = 15
        _reindex_status["message"] = "Counting products..."

        stats = db_client.get_stats()
        total = stats.get("count", 0)

        _reindex_status["progress"] = 100
        _reindex_status["message"] = f"Complete! {total} products indexed."
        _reindex_status["indexed"] = total
        _reindex_status["complete"] = True

    except Exception as e:
        logger.exception(f"Reindex failed: {e}")
        _reindex_status["message"] = f"Error: {str(e)}"
        _reindex_status["complete"] = True

    finally:
        _reindex_status["running"] = False


@router.post("/admin/reindex")
async def start_reindex(background_tasks: BackgroundTasks):
    """Start re-indexing (background)."""
    global _reindex_status

    if _reindex_status["running"]:
        raise HTTPException(status_code=409, detail="Reindex already in progress")

    _reindex_status = {
        "running": True, "complete": False,
        "progress": 0, "message": "Starting...", "indexed": 0,
    }

    thread = threading.Thread(target=run_reindex_task)
    thread.start()

    return {"status": "started", "message": "Reindex started in background"}


@router.get("/admin/reindex/status")
async def get_reindex_status():
    """Get the current reindex status."""
    return _reindex_status

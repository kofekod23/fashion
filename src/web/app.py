"""FastAPI application factory."""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from src.config import config
from src.database.weaviate_client import WeaviateClient
from src.models.base import BaseModel
from src.models.clip_model import CLIPModel
from src.models.openclip_model import OpenCLIPModel
from src.models.siglip_model import SigLIPModel
from src.search.engine import SearchEngine

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Global instances with reentrant lock for safety
_model: BaseModel | None = None
_models: dict[str, BaseModel] | None = None
_db_client: WeaviateClient | None = None
_search_engine: SearchEngine | None = None
_init_lock = threading.RLock()

# Model configurations: vector_name -> (class, model_name)
MODEL_CONFIGS = {
    "fashion_clip": (CLIPModel, "patrickjohncyh/fashion-clip"),
    "marqo_clip": (OpenCLIPModel, "Marqo/marqo-fashionCLIP"),
    "siglip2": (SigLIPModel, "Marqo/marqo-fashionSigLIP"),
}


def get_model() -> BaseModel:
    """Get the Fashion CLIP model (backward compatibility)."""
    return get_models()["fashion_clip"]


def get_models() -> dict[str, BaseModel]:
    """Get all 3 model instances."""
    global _models
    with _init_lock:
        if _models is None:
            _models = {}
            for name, (cls, model_name) in MODEL_CONFIGS.items():
                logger.info(f"Loading model {name}: {model_name}")
                _models[name] = cls(model_name)
                logger.info(f"Model {name} loaded ({_models[name].get_dimension()}d)")
    return _models


def get_db_client() -> WeaviateClient:
    """Get the global database client instance."""
    global _db_client
    with _init_lock:
        if _db_client is None:
            _db_client = WeaviateClient(config)
            _db_client.connect()
    return _db_client


def get_search_engine() -> SearchEngine:
    """Get the global search engine instance."""
    global _search_engine
    with _init_lock:
        if _search_engine is None:
            _search_engine = SearchEngine(get_models(), get_db_client())
    return _search_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    try:
        logger.info("Loading models...")
        models = get_models()
        logger.info(f"All {len(models)} models loaded: {list(models.keys())}")

        logger.info(f"Connecting to Weaviate at {config.weaviate_url}...")
        db_client = get_db_client()

        if db_client.collection_exists():
            stats = db_client.get_stats()
            logger.info(
                f"Connected to Weaviate. Collection '{config.collection_name}' has {stats.get('count', 0)} images."
            )
        else:
            logger.warning(
                f"Connected to Weaviate but collection '{config.collection_name}' does not exist. "
                "Run the indexer first to create and populate the collection."
            )

    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.warning("Application starting with degraded functionality. Some features may not work.")

    yield

    global _db_client
    if _db_client:
        logger.info("Closing Weaviate connection...")
        _db_client.close()
        _db_client = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Fashion Search Engine",
        description="Search fashion items using natural language with multi-model hybrid search",
        version="0.2.0",
        lifespan=lifespan,
    )

    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

    from .routes import router
    app.include_router(router)

    return app


app = create_app()

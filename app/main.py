"""
Fashion RecSys — FastAPI Application Entry Point.

Run locally:
    uvicorn app.main:app --reload --port 8000

Environment variables required (.env):
    WEAVIATE_URL           - Weaviate Cloud cluster URL
    WEAVIATE_API_KEY       - Weaviate API key
    UPSTASH_REDIS_REST_URL - Upstash Redis REST URL
    UPSTASH_REDIS_REST_TOKEN - Upstash Redis REST token

Optional:
    CLIP_EMBEDDINGS_PATH   - Path to clip_embeddings.pt (default: checkpoints/clip_embeddings.pt)
    STUDENT_MLP_PATH       - Path to student_mlp_full.pt (default: checkpoints/student_mlp_full.pt)
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import search, recommend, interact, catalog

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: eager warm-up of all singletons ─────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up all singleton dependencies on startup so the first request
    doesn't pay the cold-start penalty.
    """
    logger.info("Starting up Fashion RecSys API…")

    from app.dependencies import (
        get_clip_ram_index,
        get_clip_encoder,
        get_recommend_pipeline,
        get_interaction_handler,
        get_adaptation_worker,
        get_redis_store,
        get_search_engine,
        get_weaviate_client,
    )

    get_weaviate_client()
    get_clip_encoder()
    get_search_engine()
    get_redis_store()
    get_clip_ram_index()
    get_recommend_pipeline()
    get_interaction_handler()
    get_adaptation_worker()

    logger.info("All services ready. API is accepting requests.")
    yield

    # Shutdown: close Weaviate connection
    try:
        get_weaviate_client().close()
        logger.info("Weaviate connection closed.")
    except Exception:
        pass


# ── Application factory ───────────────────────────────────────

app = FastAPI(
    title="Fashion RecSys API",
    description=(
        "Real-Time Personalized Fashion Recommender. "
        "Hybrid search (BM25 + FashionCLIP) backed by Weaviate. "
        "Per-user Personal MLP adaptation via triplet loss, state stored in Redis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Next.js frontend (any origin in dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────

app.include_router(catalog.router)
app.include_router(search.router)
app.include_router(recommend.router)
app.include_router(interact.router)


# ── Health check ──────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health() -> dict:
    """Return API liveness status."""
    return {"status": "ok"}

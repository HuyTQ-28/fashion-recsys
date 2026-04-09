"""
Modal Serverless Deployment.

Defines Modal App with:
- CPU function: /search endpoint (stateless, FashionCLIP + Weaviate)
- CPU function: /recommend endpoint (Weaviate + Personal MLP re-rank)
- CPU function: /interact endpoint (EMA update + triplet adaptation)
- CPU function: /user_state endpoint (Get user state)
"""

import os
import io
import base64
import logging
import torch
from typing import Optional, List, Dict
import urllib.parse

from fastapi import Request, HTTPException
import modal

logger = logging.getLogger(__name__)

# ============================================================
# Modal App Configuration
# ============================================================

app = modal.App("fashion-recsys")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "transformers>=4.36.0",
        "fashion-clip>=0.2.0",
        "weaviate-client>=4.4.0",
        "upstash-redis>=1.0.0",
        "fastapi>=0.109.0",
        "pydantic>=2.5.0",
        "pandas>=2.1.0",
        "numpy<2.0.0",
        "Pillow>=10.0.0",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("src", remote_path="/root/src", ignore=["venv", "__pycache__", ".git"])
)

# Volumes and secrets
vol_checkpoints = modal.Volume.from_name("checkpoints", create_if_missing=True)

# Global variables for caching instances across requests within the same container
_clip_encoder = None
_weaviate_client = None
_search_engine = None
_redis_backend = None
_personalization_engine = None

def get_weaviate_client():
    global _weaviate_client
    if _weaviate_client is None:
        import weaviate
        url = os.environ["WEAVIATE_URL"]
        api_key = os.environ["WEAVIATE_API_KEY"]
        # Determine if we should use https based on the URL
        if url.startswith("http"):
            url = url.split("://")[1]
            
        _weaviate_client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
            skip_init_checks=True
        )
    return _weaviate_client

def get_clip_encoder():
    global _clip_encoder
    if _clip_encoder is None:
        from src.extractor.clip_encoder import CLIPEncoder
        _clip_encoder = CLIPEncoder()
    return _clip_encoder

def get_search_engine():
    global _search_engine
    if _search_engine is None:
        from src.search.search_engine import HybridSearchEngine
        _search_engine = HybridSearchEngine(get_weaviate_client(), get_clip_encoder())
    return _search_engine

def get_redis_backend():
    global _redis_backend
    if _redis_backend is None:
        from src.inference.redis_client import UpstashRedisBackend
        _redis_backend = UpstashRedisBackend()
    return _redis_backend

def get_personalization_engine():
    global _personalization_engine
    if _personalization_engine is None:
        from src.inference.recommender import PersonalizationEngine
        from src.inference.mlp_lifecycle import MLPLifecycleManager
        from src.models.personal_mlp import PersonalMLP
        
        # Load clip embeddings for in-memory re-ranking
        clip_embeddings = {}
        if os.path.exists("/checkpoints/clip_embeddings.pt"):
            clip_embeddings = torch.load("/checkpoints/clip_embeddings.pt", map_location="cpu", weights_only=True)
        
        # Create Student MLP template
        student_model = PersonalMLP(layer_dims=[512, 256, 128, 64], user_id="template")
        if os.path.exists("/checkpoints/student_mlp_full.pt"):
            state_dict = torch.load("/checkpoints/student_mlp_full.pt", map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            remapped = {("network." + k[4:] if k.startswith("net.") else k): v for k, v in state_dict.items()}
            student_model.load_state_dict(remapped)
        
        lifecycle_manager = MLPLifecycleManager(
            factory=student_model,
            backend=get_redis_backend(),
        )
        _personalization_engine = PersonalizationEngine(
            lifecycle=lifecycle_manager,
            clip_embeddings=clip_embeddings
        )
    return _personalization_engine


@app.function(
    image=image,
    memory=1024,
    secrets=[modal.Secret.from_name("weaviate-credentials")],
)
@modal.fastapi_endpoint(method="POST")
def search(request: dict):
    """Hybrid search endpoint."""
    query = request.get("query")
    image_b64 = request.get("image_b64")
    filters = request.get("filters")
    user_id = request.get("user_id")
    alpha = request.get("alpha", 0.7)
    mode = request.get("mode", "hybrid")
    limit = request.get("limit", 20)
    
    img = None
    if image_b64:
        from PIL import Image
        img_data = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
    engine = get_search_engine()
    results = engine.search(
        query=query,
        image=img,
        mode=mode,
        alpha=alpha,
        filters=filters,
        limit=limit
    )
    
    return {"results": results, "query_mode": mode, "total_results": len(results)}


@app.function(
    image=image,
    memory=1024,
    secrets=[
        modal.Secret.from_name("weaviate-credentials"),
        modal.Secret.from_name("upstash-redis"),
    ],
    volumes={"/checkpoints": vol_checkpoints},
)
@modal.fastapi_endpoint(method="POST")
def recommend(request: dict):
    """Personalized recommendation endpoint."""
    article_id = request.get("article_id")
    user_id = request.get("user_id")
    k = request.get("k", 10)
    
    if not article_id:
        return {"recommendations": []}
        
    engine = get_personalization_engine()
    rec_ids = engine.get_recommendations(
        user_id=user_id or "cold_user",
        seed_article_id=article_id,
        k=k
    )
    
    # Format IDs back into dicts to match Streamlit expectations
    recommendations = [{"article_id": r_id} for r_id in rec_ids]
    return {"recommendations": recommendations}


@app.function(
    image=image,
    memory=1024,
    secrets=[
        modal.Secret.from_name("upstash-redis"),
    ],
    volumes={"/checkpoints": vol_checkpoints},
)
@modal.fastapi_endpoint(method="POST")
def interact(request: dict):
    """User interaction endpoint (EMA + adaptation)."""
    user_id = request.get("user_id")
    article_id = request.get("article_id")
    shown_articles = request.get("shown_articles", [])
    
    if not user_id or not article_id:
        return {"status": "error", "message": "user_id and article_id required"}
        
    engine = get_personalization_engine()
    result = engine.handle_interaction(
        user_id=user_id,
        article_id=article_id,
        interaction_type="purchase",
        shown_articles=shown_articles
    )
    return result


@app.function(
    image=image,
    memory=1024,
    secrets=[
        modal.Secret.from_name("upstash-redis"),
    ],
    volumes={"/checkpoints": vol_checkpoints},
)
@modal.fastapi_endpoint(method="GET")
def user_state(user_id: str):
    """Fetch current user state."""
    if not user_id:
        return {"user_id": None, "interaction_count": 0, "has_personalization": False}
        
    engine = get_personalization_engine()
    return engine.get_user_state(user_id)

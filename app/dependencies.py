import logging
import os
from functools import lru_cache
from typing import Dict
import torch

from src.data_access.weaviate.product_rec_repo import ProductRecRepo
from src.data_access.catalog.clip_ram_index import ClipRamIndex
from src.data_access.redis.session_store import RedisSessionStore
from src.core.recommender.reranker import PersonalReranker
from src.core.recommender.pipeline import RecommendPipeline
from src.core.personalization.interaction_handler import InteractionHandler
from src.workers.adaptation_worker import AdaptationWorker
from src.models.student_mlp import StudentMLP
from src.models.personal_mlp import PersonalMLPFactory

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_weaviate_client():
    """Return a connected Weaviate v4 client (singleton)."""
    import weaviate

    url = os.environ["WEAVIATE_URL"]
    api_key = os.environ["WEAVIATE_API_KEY"]

    url = url.split("://")[-1]

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
        skip_init_checks=True,
    )
    logger.info("Weaviate client connected: %s", client.is_ready())
    return client

@lru_cache(maxsize=1)
def get_clip_encoder():
    """Return a loaded FashionCLIP encoder (singleton)."""
    from src.extractor.clip_encoder import CLIPEncoder

    encoder = CLIPEncoder()
    logger.info("CLIPEncoder loaded")
    return encoder

@lru_cache(maxsize=1)
def get_search_engine():
    """Return the HybridSearchEngine backed by Weaviate + CLIP (singleton)."""
    from src.search.search_engine import HybridSearchEngine

    engine = HybridSearchEngine(get_weaviate_client(), get_clip_encoder())
    logger.info("HybridSearchEngine ready")
    return engine

@lru_cache(maxsize=1)
def get_redis_store() -> RedisSessionStore:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    prefix = os.environ.get("REDIS_KEY_PREFIX", "rec")
    
    store = RedisSessionStore(redis_url=url, prefix=prefix)
    logger.info("RedisSessionStore connected")
    return store

@lru_cache(maxsize=1)
def get_clip_ram_index() -> ClipRamIndex:
    """
    Load pre-computed CLIP embeddings from disk into memory.
    """
    configured = os.environ.get("CLIP_EMBEDDINGS_PATH", "checkpoints/clip_embeddings.pt")
    fallback_candidates = [
        configured,
        "dataset/subset_1week/clip_embeddings.pt",
        "checkpoints/clip_embeddings.pt",
    ]

    path = next((p for p in fallback_candidates if os.path.exists(p)), None)

    if path is None:
        logger.warning(
            "clip_embeddings.pt not found in any known location (%s) — running cold-start only",
            ", ".join(fallback_candidates),
        )
        embeddings = {}
    else:
        embeddings = torch.load(path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded {len(embeddings)} CLIP embeddings from {path}")
        
    return ClipRamIndex(clip_dict=embeddings)

@lru_cache(maxsize=1)
def get_base_mlp() -> StudentMLP:
    checkpoint_path = os.environ.get("STUDENT_MLP_PATH", "checkpoints/student_mlp_full.pt")
    in_dim = get_clip_ram_index().dim
    model = StudentMLP(layer_dims=[in_dim, 256, 128, 64])
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    else:
        logger.warning(f"student_mlp_full.pt not found at {checkpoint_path} — using random weights")
    model.eval()
    return model

@lru_cache(maxsize=1)
def get_personal_mlp_factory() -> PersonalMLPFactory:
    checkpoint_path = os.environ.get("STUDENT_MLP_PATH", "checkpoints/student_mlp_full.pt")
    in_dim = get_clip_ram_index().dim
    return PersonalMLPFactory(checkpoint_path=checkpoint_path, layer_dims=[in_dim, 256, 128, 64])

@lru_cache(maxsize=1)
def get_product_rec_repo() -> ProductRecRepo:
    return ProductRecRepo(get_weaviate_client())

@lru_cache(maxsize=1)
def get_recommend_pipeline() -> RecommendPipeline:
    reranker = PersonalReranker(get_clip_ram_index())
    return RecommendPipeline(
        redis_store=get_redis_store(),
        retriever=get_product_rec_repo(),
        reranker=reranker
    )

@lru_cache(maxsize=1)
def get_interaction_handler() -> InteractionHandler:
    import yaml
    
    # Load config settings if available
    config_path = os.environ.get("PERSONALIZATION_CONFIG_PATH", "configs/personalization.yaml")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            
    p_config = config.get("personalization_mlp", config.get("personal_mlp", {}))
    
    trigger_every_n = int(os.environ.get("TRIGGER_EVERY_N_INTERACTIONS", p_config.get("trigger_every_n_interactions", 9)))
    ttl_days = p_config.get("redis_ttl_days", 14)
    
    return InteractionHandler(
        redis_store=get_redis_store(),
        clip_index=get_clip_ram_index(),
        base_mlp=get_base_mlp(),
        personal_mlp_factory=get_personal_mlp_factory(),
        trigger_every_n=trigger_every_n,
        ttl_seconds=ttl_days * 24 * 3600
    )

@lru_cache(maxsize=1)
def get_adaptation_worker() -> AdaptationWorker:
    import yaml
    
    config_path = os.environ.get("PERSONALIZATION_CONFIG_PATH", "configs/personalization.yaml")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            pass
            
    p_config = config.get("personalization_mlp", config.get("personal_mlp", {}))
    ttl_days = p_config.get("redis_ttl_days", 14)

    return AdaptationWorker(
        redis_store=get_redis_store(),
        clip_index=get_clip_ram_index(),
        ttl_seconds=ttl_days * 24 * 3600
    )

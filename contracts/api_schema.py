"""
API Contracts — Pydantic models for request/response schemas.

All team members build against these contracts:
- Member 3 (Infra) implements the API endpoints
- Member 4 (Frontend) calls the API from Streamlit
- Member 2 (Personalization) implements the engine behind the contracts
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================
# /search endpoint
# ============================================================

class SearchRequest(BaseModel):
    """Request body for POST /search."""
    query: Optional[str] = Field(None, description="Text search query (e.g., 'red summer dress')")
    image_b64: Optional[str] = Field(None, description="Base64-encoded image for image-to-image search")
    filters: Optional[Dict[str, str]] = Field(
        None,
        description="Metadata filters: colour_group_name, department_name, product_type_name, etc.",
    )
    user_id: Optional[str] = Field(None, description="User ID for personalized re-ranking (optional)")
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Hybrid blend: 0=pure BM25, 1=pure vector")
    mode: str = Field("hybrid", description="Search mode: 'semantic', 'keyword', or 'hybrid'")
    limit: int = Field(20, ge=1, le=100, description="Number of results to return")


class SearchResultItem(BaseModel):
    """A single search result."""
    article_id: str
    score: float
    product_name: str
    product_type_name: str
    colour_group_name: str
    department_name: str
    index_group_name: str
    garment_group_name: str
    detail_desc: Optional[str] = None
    image_path: str


class SearchResponse(BaseModel):
    """Response body for POST /search."""
    results: List[SearchResultItem]
    query_mode: str
    total_results: int
    personalized: bool = False


# ============================================================
# /recommend endpoint
# ============================================================

class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""
    article_id: str = Field(..., description="Seed article ID to get recommendations for")
    user_id: Optional[str] = Field(None, description="User ID for personalized re-ranking")
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")


class RecommendItem(BaseModel):
    """A single recommendation."""
    article_id: str
    score: float
    product_name: str
    image_path: str


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""
    recommendations: List[RecommendItem]
    seed_article_id: str
    personalized: bool = False
    user_interaction_count: int = 0


# ============================================================
# /interact endpoint
# ============================================================

class InteractRequest(BaseModel):
    """Request body for POST /interact."""
    user_id: str = Field(..., description="User ID")
    article_id: str = Field(..., description="Article the user clicked/interacted with")
    shown_articles: Optional[List[str]] = Field(
        None,
        description="Articles displayed but not clicked (hard negatives for triplet loss)",
    )


class InteractResponse(BaseModel):
    """Response body for POST /interact."""
    status: str = "ok"
    interaction_count: int
    adapted: bool = Field(False, description="Whether triplet adaptation was triggered")
    adaptation_time_ms: Optional[float] = None


# ============================================================
# /user_state endpoint (debug / visualization)
# ============================================================

class UserStateRequest(BaseModel):
    """Request body for GET /user_state."""
    user_id: str


class UserStateResponse(BaseModel):
    """Response body for GET /user_state."""
    user_id: str
    exists: bool
    interaction_count: int = 0
    ema_vector: Optional[List[float]] = None
    cache_location: str = "none"  # "lru", "redis", "none"
    redis_ttl_remaining_seconds: Optional[int] = None

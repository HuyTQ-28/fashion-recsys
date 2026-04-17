from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Text query (e.g. 'red summer dress')")
    image_b64: Optional[str] = Field(None, description="Base64-encoded product image")
    mode: str = Field("hybrid", description="Search mode: 'hybrid', 'semantic', or 'keyword'")
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Hybrid blend weight (0=BM25, 1=vector)")
    filters: Optional[Dict[str, str]] = Field(None, description="Metadata filters, e.g. {'colour_group_name': 'Black'}")
    limit: int = Field(20, ge=1, le=100, description="Max number of results")


class SearchResult(BaseModel):
    article_id: str
    prod_name: Optional[str] = None
    colour_group_name: Optional[str] = None
    product_group_name: Optional[str] = None
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    score: float = 0.0


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_mode: str
    total_results: int


class RecommendRequest(BaseModel):
    article_id: str = Field(..., description="Seed article ID for recommendation")
    user_id: Optional[str] = Field(None, description="User ID for personalized recommendations")
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")


class RecommendResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    user_id: str
    personalized: bool


class InteractRequest(BaseModel):
    user_id: str = Field(..., description="User ID performing the interaction")
    article_id: str = Field(..., description="Article the user interacted with")
    interaction_type: str = Field("purchase", description="Interaction type: 'purchase', 'click', or 'view'")
    shown_articles: List[str] = Field(default_factory=list, description="Articles shown alongside (used as negatives)")


class InteractResponse(BaseModel):
    status: str
    interaction_count: int
    adapted: bool
    adaptation_time_ms: Optional[float] = None


class UserStateResponse(BaseModel):
    user_id: str
    ema_vector: Optional[List[float]]
    interaction_count: int
    has_personalization: bool

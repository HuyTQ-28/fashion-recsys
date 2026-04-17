import logging

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_search_engine
from app.schemas import SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalog", tags=["Catalog"])


@router.get("", response_model=SearchResponse)
def catalog(
    limit: int = Query(default=20, ge=1, le=100, description="Max number of items"),
    engine=Depends(get_search_engine),
) -> SearchResponse:
    """
    Return a browseable page of catalog products without any search query.

    This endpoint is used for first-load / cold-start browsing. It calls
    ``fetch_objects`` via gRPC, with an automatic REST API fallback when
    gRPC is unavailable (e.g. WSL2 network restrictions).
    """
    results = engine.search(query=None, image=None, limit=limit)
    return SearchResponse(
        results=results,
        query_mode="fetch",
        total_results=len(results),
    )

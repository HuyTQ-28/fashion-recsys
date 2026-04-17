import base64
import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_search_engine
from app.schemas import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
def search(
    body: SearchRequest,
    engine=Depends(get_search_engine),
) -> SearchResponse:
    """
    Hybrid product search combining BM25 keyword matching with FashionCLIP
    semantic vectors.

    **Modes**:
    - `hybrid` (default): blended BM25 + vector, controlled by `alpha`
    - `semantic`: pure vector search from text or image query
    - `keyword`: pure BM25 text search

    **Image search**: supply a base64-encoded JPEG/PNG in `image_b64`.
    The image is encoded by FashionCLIP and compared against the product catalog.

    **Metadata filters**: narrow results by any indexed property, e.g.
    `{"colour_group_name": "Black", "product_group_name": "Tops"}`.
    """
    img = None
    if body.image_b64:
        try:
            from PIL import Image
            raw = base64.b64decode(body.image_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc

    if not body.query and img is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of 'query' or 'image_b64'.",
        )

    results = engine.search(
        query=body.query,
        image=img,
        mode=body.mode,
        alpha=body.alpha,
        filters=body.filters,
        limit=body.limit,
    )

    return SearchResponse(
        results=results,
        query_mode=body.mode,
        total_results=len(results),
    )

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks

from app.dependencies import get_interaction_handler, get_adaptation_worker
from app.schemas import InteractRequest, InteractResponse, UserStateResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Personalization"])


@router.post("/interact", response_model=InteractResponse)
def interact(
    body: InteractRequest,
    background_tasks: BackgroundTasks,
    handler=Depends(get_interaction_handler),
    worker=Depends(get_adaptation_worker),
) -> InteractResponse:
    """
    Record that a user interacted with an article and update their live state.

    **Per interaction**:
    1. Projects the article's CLIP embedding through the user's Personal MLP.
    2. Updates the EMA user vector: `u_t = (1-α)·u_{t-1} + α·MLP_u(h_CNN)`.
    3. Every `trigger_every_n` interactions, runs a few SGD steps of triplet
       loss adaptation on the Personal MLP (Equations 6-7 of the paper).
    4. Flushes the updated MLP weights to Redis.

    **Negatives for triplet loss**: items in `shown_articles` that were *not*
    interacted with. Falls back to random catalog negatives if none are provided.
    """
    result = handler.handle(
        session_id=body.user_id,
        article_id=body.article_id,
        interaction_type=body.interaction_type,
    )

    if result.get("status") == "skipped":
        raise HTTPException(
            status_code=404,
            detail=f"article_id '{body.article_id}' not found in embedding catalog.",
        )

    if result.get("needs_adaptation"):
        background_tasks.add_task(
            worker.run_adaptation,
            session_id=body.user_id,
            shown_articles=body.shown_articles,
        )

    return InteractResponse(
        status=result["status"],
        interaction_count=result["interaction_count"],
        adapted=result.get("needs_adaptation", False),
        adaptation_time_ms=None,
    )


@router.get("/user_state", response_model=UserStateResponse)
def user_state(
    user_id: str = Query(..., description="User ID to inspect"),
    handler=Depends(get_interaction_handler),
) -> UserStateResponse:
    """
    Return the current EMA representation for a user.

    Useful for debugging personalization: inspect the 64-dim user vector,
    interaction count, and whether the user has been personalized yet.
    Returns `has_personalization: false` for cold users with no interactions.
    """
    state = handler.get_user_state(user_id)
    return UserStateResponse(**state)

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RecommendPipeline:
    """
    Orchestrates the Two-Stage Retriever-Reranker pipeline.
    Ensures strict separation of Global Space and Personal Space.
    """
    def __init__(self, redis_store, retriever, reranker):
        self.redis = redis_store
        self.retriever = retriever
        self.reranker = reranker

    def recommend(self, session_id: str, k: int = 10, seed_article_id: Optional[str] = None) -> Dict[str, Any]:
        # 1) Read consistent snapshot from Redis
        snap = self.redis.read_snapshot(session_id)
        
        # Cold start logic if missing personalization
        if snap is None or not snap.u_t_base or not snap.u_t_personal:
            default_ids = self.retriever.fetch_default_ids(limit=max(k * 3, 20))
            if seed_article_id and seed_article_id in default_ids:
                default_ids.remove(seed_article_id)
            return {"article_ids": default_ids[:k], "model_version": 0}

        # 2) Stage-1 Retriever in GLOBAL space
        # Query Weaviate using u_t_base
        cand_ids = self.retriever.search_ids_near_vector(
            query_vec64=snap.u_t_base,
            limit=500
        )

        if not cand_ids:
            return {"article_ids": [], "model_version": snap.model_version}

        # Filter out seed article if provided
        if seed_article_id and seed_article_id in cand_ids:
            cand_ids.remove(seed_article_id)

        # 3) Stage-2 Reranker in PERSONAL space
        # Query RAM index using u_t_personal
        top_ids = self.reranker.rerank(
            candidate_ids=cand_ids,
            personal_weights_b64=snap.personal_weights,
            u_t_personal=snap.u_t_personal,
            top_k=k,
        )

        return {"article_ids": top_ids, "model_version": snap.model_version}

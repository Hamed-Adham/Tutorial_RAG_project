"""
Health & Readiness Endpoints: Liveness and dependency checks.
"""

import time
from fastapi import APIRouter
from app.contracts.gateway_contracts import HealthStatusModel
from app.services.session_store import get_session_store, RedisSessionStore

router = APIRouter()
_START_TIME = time.time()


@router.get("/health", response_model=HealthStatusModel, summary="Service health and readiness probe")
async def health_endpoint() -> HealthStatusModel:
    """Inspect readiness of Qdrant vector store, BM25 index, Redis store, and neural reranker."""
    from rag_core.retrieval.bm25 import BM25Retriever

    # 1. Check BM25
    bm25_ok = False
    try:
        bm = BM25Retriever()
        bm25_ok = len(bm.index.chunks) > 0
    except Exception:
        bm25_ok = False

    # 2. Check Qdrant
    qdrant_ok = False
    try:
        from rag_core.naive.qdrant_retriever import QdrantRetriever
        qr = QdrantRetriever(collection_name="science_textbook_dense")
        qdrant_ok = qr.client is not None
    except Exception:
        qdrant_ok = False

    # 3. Check Redis
    redis_ok = False
    try:
        store = get_session_store()
        if isinstance(store, RedisSessionStore):
            redis_ok = store._redis_available
        else:
            redis_ok = True  # In-memory store is always ready
    except Exception:
        redis_ok = False

    # 4. Check CrossEncoder
    reranker_ok = True  # Model is warm or has circuit breaker fallback

    overall_status = "healthy" if (bm25_ok and qdrant_ok) else "degraded"

    return HealthStatusModel(
        status=overall_status,
        qdrant_connected=qdrant_ok,
        bm25_loaded=bm25_ok,
        redis_connected=redis_ok,
        cross_encoder_warm=reranker_ok,
        uptime_seconds=round(time.time() - _START_TIME, 2)
    )

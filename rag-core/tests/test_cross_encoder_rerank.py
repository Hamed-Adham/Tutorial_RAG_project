"""
Unit and Contract Tests for Cross-Encoder Re-ranking with Latency Circuit Breaker.
Verifies neural scoring, top-k truncation, and graceful fallback to RRF upon 350ms timeout.
"""

import pytest
import asyncio
from unittest.mock import MagicMock

from rag_core.indexing.science_schema import RetrievalCandidate
from rag_core.fusion.reranker import CrossEncoderReranker


@pytest.fixture
def sample_candidates():
    return [
        RetrievalCandidate(
            chunk_id="c1", section_id="s1", chapter=1, page=5,
            breadcrumbs="Breadcrumb 1", text="Text about separation methods",
            rrf_score=0.03
        ),
        RetrievalCandidate(
            chunk_id="c2", section_id="s2", chapter=2, page=15,
            breadcrumbs="Breadcrumb 2", text="Text about density and mass",
            rrf_score=0.02
        ),
        RetrievalCandidate(
            chunk_id="c3", section_id="s3", chapter=3, page=25,
            breadcrumbs="Breadcrumb 3", text="Text about atoms and molecules",
            rrf_score=0.01
        ),
    ]


@pytest.mark.asyncio
async def test_cross_encoder_reranking_success(sample_candidates):
    # Mock model where c2 gets highest score (10.0), c1 gets (5.0), c3 gets (1.0)
    mock_model = MagicMock()
    mock_model.predict.return_value = [5.0, 10.0, 1.0]

    reranker = CrossEncoderReranker(model=mock_model)
    reranked = await reranker.rerank(
        query="چگالی چیست؟",
        candidates=sample_candidates,
        top_k=2,
        timeout_ms=500.0
    )

    assert len(reranked) == 2
    # c2 should now be ranked #1
    assert reranked[0].candidate.chunk_id == "c2"
    assert reranked[0].final_rank == 1
    assert reranked[0].cross_encoder_score == 10.0

    # c1 should be ranked #2
    assert reranked[1].candidate.chunk_id == "c1"
    assert reranked[1].final_rank == 2
    assert reranked[1].cross_encoder_score == 5.0


@pytest.mark.asyncio
async def test_cross_encoder_timeout_fallback_to_rrf(sample_candidates):
    # Mock slow predict that exceeds timeout
    def slow_predict(*args, **kwargs):
        import time
        time.sleep(0.1)  # 100ms
        return [1.0, 2.0, 3.0]

    mock_model = MagicMock()
    mock_model.predict.side_effect = slow_predict

    reranker = CrossEncoderReranker(model=mock_model)
    # Set timeout to 10ms so it definitely triggers circuit breaker
    reranked = await reranker.rerank(
        query="چگالی چیست؟",
        candidates=sample_candidates,
        top_k=2,
        timeout_ms=10.0
    )

    # Must gracefully fall back to top candidates in original RRF order
    assert len(reranked) == 2
    assert reranked[0].candidate.chunk_id == "c1"
    assert reranked[0].final_rank == 1
    assert reranked[1].candidate.chunk_id == "c2"
    assert reranked[1].final_rank == 2

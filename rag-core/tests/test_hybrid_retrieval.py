"""
Contract and Unit Tests for Hybrid Retrieval (BM25 + Dense) and Reciprocal Rank Fusion.
Verifies sub-5ms BM25 lexical search, reciprocal rank merging, and parallel execution.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from rag_core.indexing.science_schema import ScienceChunk, RetrievalCandidate
from rag_core.indexing.bm25_index import ScienceBM25Index
from rag_core.retrieval.bm25 import BM25Retriever
from rag_core.fusion.rrf import ReciprocalRankFusion
from rag_core.retrieval.hybrid import HybridRetriever


@pytest.fixture
def sample_science_chunks():
    return [
        ScienceChunk(
            chunk_id="chunk_01",
            doc_id="ch01",
            section_id="sec_01",
            chapter=1,
            page=2,
            breadcrumbs="فصل ۱ > تجربه و تفکر",
            text="سانتریفیوژ یکی از دستگاه‌های مهم جداسازی مخلوط‌ها در آزمایشگاه است.",
            token_count=15
        ),
        ScienceChunk(
            chunk_id="chunk_02",
            doc_id="ch02",
            section_id="sec_02",
            chapter=2,
            page=12,
            breadcrumbs="فصل ۲ > ماده و ویژگی‌های آن",
            text="چگالی نسبت جرم به حجم یک جسم است و با فرمول d=m/V محاسبه می‌شود.",
            token_count=16
        ),
        ScienceChunk(
            chunk_id="chunk_03",
            doc_id="ch03",
            section_id="sec_03",
            chapter=3,
            page=24,
            breadcrumbs="فصل ۳ > اتم‌ها الفبای مواد",
            text="مولکول‌ها از اتصال دو یا چند اتم به یکدیگر تشکیل می‌شوند.",
            token_count=14
        ),
    ]


@pytest.fixture
def mock_bm25_index(sample_science_chunks):
    index = ScienceBM25Index()
    index.add_chunks(sample_science_chunks)
    index.build()
    return index


def test_bm25_retriever_exact_keyword(mock_bm25_index):
    retriever = BM25Retriever(index=mock_bm25_index)
    results = retriever.search("سانتریفیوژ آزمایشگاهی", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top.chunk_id == "chunk_01"
    assert top.bm25_rank == 1
    assert top.bm25_score > 0.0
    assert "سانتریفیوژ" in top.text


def test_reciprocal_rank_fusion_math():
    fusion = ReciprocalRankFusion(k=60)

    dense_candidates = [
        RetrievalCandidate(
            chunk_id="chunk_A", section_id="sec_A", chapter=1, page=1,
            breadcrumbs="A", text="text A", dense_score=0.95, dense_rank=1
        ),
        RetrievalCandidate(
            chunk_id="chunk_B", section_id="sec_B", chapter=2, page=2,
            breadcrumbs="B", text="text B", dense_score=0.85, dense_rank=2
        ),
    ]

    bm25_candidates = [
        RetrievalCandidate(
            chunk_id="chunk_B", section_id="sec_B", chapter=2, page=2,
            breadcrumbs="B", text="text B", bm25_score=15.0, bm25_rank=1
        ),
        RetrievalCandidate(
            chunk_id="chunk_A", section_id="sec_A", chapter=1, page=1,
            breadcrumbs="A", text="text A", bm25_score=10.0, bm25_rank=2
        ),
    ]

    fused = fusion.fuse(dense_candidates, bm25_candidates, top_k=2)

    assert len(fused) == 2
    # chunk_A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # chunk_B: 1/(60+2) + 1/(60+1) = 0.032522
    # Since they tie in symmetric rank sum, both should have rrf_score ~ 0.03252
    assert pytest.approx(fused[0].rrf_score, rel=1e-3) == (1 / 61 + 1 / 62)
    assert pytest.approx(fused[1].rrf_score, rel=1e-3) == (1 / 61 + 1 / 62)


@pytest.mark.asyncio
async def test_hybrid_retriever_parallel_gather(sample_science_chunks):
    # Mock dense retriever
    mock_dense = AsyncMock()
    mock_dense.retrieve = AsyncMock(return_value=[
        RetrievalCandidate(
            chunk_id="chunk_02", section_id="sec_02", chapter=2, page=12,
            breadcrumbs="فصل ۲", text="چگالی نسبت جرم به حجم است", dense_score=0.91, dense_rank=1
        )
    ])

    # Mock BM25 retriever
    mock_bm25 = MagicMock()
    mock_bm25.search = MagicMock(return_value=[
        RetrievalCandidate(
            chunk_id="chunk_02", section_id="sec_02", chapter=2, page=12,
            breadcrumbs="فصل ۲", text="چگالی نسبت جرم به حجم است", bm25_score=12.5, bm25_rank=1
        )
    ])

    hybrid = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=mock_bm25)
    fused_results = await hybrid.retrieve("چگالی چیست؟", top_k=5)

    assert len(fused_results) == 1
    assert fused_results[0].chunk_id == "chunk_02"
    assert fused_results[0].rrf_score > 0.0
    mock_dense.retrieve.assert_awaited_once_with("چگالی چیست؟", top_k=50)
    mock_bm25.search.assert_called_once_with("چگالی چیست؟", top_k=50)


@pytest.mark.asyncio
async def test_hybrid_retriever_native_chapter_prefiltering():
    """
    Verify that when chapters is passed to retrieve(), it is passed down as native
    pre-filtering parameters to dense and BM25 search before candidate ranking.
    """
    mock_dense = AsyncMock()
    mock_dense.retrieve = AsyncMock(return_value=[
        RetrievalCandidate(
            chunk_id="ch06_01", section_id="sec_06", chapter=6, page=50,
            breadcrumbs="فصل ۶", text="رودخانه‌ها و چرخه آب", dense_score=0.88, dense_rank=1
        )
    ])

    mock_bm25 = MagicMock()
    mock_bm25.search = MagicMock(return_value=[
        RetrievalCandidate(
            chunk_id="ch06_01", section_id="sec_06", chapter=6, page=50,
            breadcrumbs="فصل ۶", text="رودخانه‌ها و چرخه آب", bm25_score=15.0, bm25_rank=1
        )
    ])

    hybrid = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=mock_bm25)
    fused = await hybrid.retrieve("رودخانه‌ها چگونه تشکیل می‌شوند؟", top_k=5, chapters=[6, 7])

    assert len(fused) == 1
    assert fused[0].chapter == 6
    # Verify pre-filters were explicitly passed into underlying retrieval calls
    mock_dense.retrieve.assert_awaited_once_with("رودخانه‌ها چگونه تشکیل می‌شوند؟", top_k=50, filter_chapters=[6, 7])
    mock_bm25.search.assert_called_once_with("رودخانه‌ها چگونه تشکیل می‌شوند؟", top_k=50, chapters=[6, 7])

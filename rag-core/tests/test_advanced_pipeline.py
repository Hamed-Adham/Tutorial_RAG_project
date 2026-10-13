#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end integration tests for AdvancedRAGPipeline.
Asserts latency budgets (<2.0s p95), strict Markdown citations,
and proper pipeline stage orchestration.
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock

from rag_core.indexing.science_schema import (
    RetrievalCandidate,
    RerankedCandidate,
    AdvancedRAGResult,
    RewriteDecision,
    ExpandedContext,
    ParentSection
)
from rag_core.pipeline import AdvancedRAGPipeline
from rag_core.retrieval.hybrid import HybridRetriever
from rag_core.query.gated_rewriter import GatedQueryRewriter
from rag_core.fusion.reranker import CrossEncoderReranker
from rag_core.postprocessing.expansion import ParentSectionExpander, LongContextReorder
from rag_core.generation.synthesizer import ResponseSynthesizer as Synthesizer, SynthesisResult


@pytest.fixture
def mock_pipeline_components():
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_rewriter = MagicMock(spec=GatedQueryRewriter)
    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_expander = MagicMock(spec=ParentSectionExpander)
    mock_reorderer = MagicMock(spec=LongContextReorder)
    mock_synthesizer = MagicMock(spec=Synthesizer)

    # Setup canned candidates
    sample_candidate = RetrievalCandidate(
        chunk_id="chunk_ch01_p12_01",
        section_id="sec_ch01_02",
        chapter=1,
        page=12,
        text="چگالی نسبت جرم به حجم یک ماده است و با یکای گرم بر سانتی‌متر مکعب بیان می‌شود.",
        breadcrumbs="فصل ۱ > اندازه گیری > چگالی",
        rrf_score=0.032
    )
    sample_reranked = RerankedCandidate(
        candidate=sample_candidate,
        cross_encoder_score=3.85,
        final_rank=1
    )
    sample_parent = ParentSection(
        section_id="sec_ch01_02",
        chapter=1,
        title="چگالی و اندازه‌گیری مواد",
        page_start=11,
        page_end=13,
        full_text="چگالی نسبت جرم به حجم یک ماده است. برای اندازه گیری حجم از استوانه مدرج استفاده می‌شود.",
        child_chunk_ids=["chunk_ch01_p12_01"],
        total_tokens=150
    )

    mock_rewriter.evaluate_and_rewrite = AsyncMock(return_value=RewriteDecision(
        needs_rewrite=False,
        confidence=1.0,
        rewritten_query="فرمول چگالی چیست؟",
        sub_queries=[]
    ))
    mock_retriever.retrieve = AsyncMock(return_value=[sample_candidate])
    mock_reranker.rerank = AsyncMock(return_value=[sample_reranked])
    mock_expander.expand.return_value = ExpandedContext(
        parent_sections=[sample_parent],
        total_tokens=150,
        source_chunk_ids=["chunk_ch01_p12_01"],
        reordered_texts=[sample_parent.full_text]
    )
    mock_reorderer.reorder.return_value = [sample_parent]
    mock_synthesizer.synthesize.return_value = SynthesisResult(
        answer="فرمول چگالی برابر است با جرم تقسیم بر حجم (p = m / V).",
        is_fallback=False,
        latency_seconds=0.12,
        model="gpt-4o",
        prompt_tokens=150,
        completion_tokens=30,
        total_tokens=180
    )

    return {
        "retriever": mock_retriever,
        "rewriter": mock_rewriter,
        "reranker": mock_reranker,
        "expander": mock_expander,
        "reorderer": mock_reorderer,
        "synthesizer": mock_synthesizer,
    }


@pytest.mark.asyncio
async def test_advanced_pipeline_e2e_latency_and_citations(mock_pipeline_components):
    """
    Assert end-to-end pipeline execution meets latency budget (<2.0s)
    and formats citations in Markdown links.
    """
    pipeline = AdvancedRAGPipeline(
        hybrid_retriever=mock_pipeline_components["retriever"],
        rewriter=mock_pipeline_components["rewriter"],
        reranker=mock_pipeline_components["reranker"],
        expander=mock_pipeline_components["expander"],
        reorderer=mock_pipeline_components["reorderer"],
        synthesizer=mock_pipeline_components["synthesizer"],
    )

    t0 = time.perf_counter()
    result = await pipeline.query(query_text="فرمول چگالی چیست؟")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(result, AdvancedRAGResult)
    assert not result.is_refusal
    assert "چگالی" in result.answer

    # Latency budget assertion: p95 < 2000ms
    assert latency_ms < 2000.0
    assert result.latency_profile["total_ms"] < 2000.0
    assert "retrieval_ms" in result.latency_profile
    assert "rerank_ms" in result.latency_profile
    assert "synthesis_ms" in result.latency_profile

    # Markdown link citations assertion (FR-009)
    assert len(result.citations) > 0
    assert result.citations[0].page == 12
    assert "#page=" in result.answer


@pytest.mark.asyncio
async def test_advanced_pipeline_zero_candidates_grounded_refusal():
    """
    When zero candidates are retrieved, pipeline returns grounded refusal
    rather than hallucinating facts.
    """
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve = AsyncMock(return_value=[])

    mock_rewriter = MagicMock(spec=GatedQueryRewriter)
    mock_rewriter.evaluate_and_rewrite = AsyncMock(return_value=RewriteDecision(
        needs_rewrite=False,
        confidence=1.0,
        rewritten_query="موضوع ناموجود",
        sub_queries=[]
    ))

    pipeline = AdvancedRAGPipeline(
        hybrid_retriever=mock_retriever,
        rewriter=mock_rewriter
    )

    result = await pipeline.query(query_text="موضوع ناموجود")

    assert result.is_refusal is True
    assert "یافت نشد" in result.answer
    assert len(result.retrieved_chunks) == 0


@pytest.mark.asyncio
async def test_advanced_pipeline_multi_concept_decomposition(mock_pipeline_components):
    """
    Composite queries decompose into multiple sub-queries and combine candidates.
    """
    mock_rewriter = mock_pipeline_components["rewriter"]
    mock_rewriter.evaluate_and_rewrite = AsyncMock(return_value=RewriteDecision(
        needs_rewrite=True,
        confidence=0.95,
        rewritten_query="نقطه جوش و اثر فشار",
        sub_queries=["نقطه جوش آب خالص و آب نمک", "تاثیر فشار بر نقطه جوش"]
    ))

    pipeline = AdvancedRAGPipeline(
        hybrid_retriever=mock_pipeline_components["retriever"],
        rewriter=mock_rewriter,
        reranker=mock_pipeline_components["reranker"],
        expander=mock_pipeline_components["expander"],
        reorderer=mock_pipeline_components["reorderer"],
        synthesizer=mock_pipeline_components["synthesizer"],
    )

    result = await pipeline.query(query_text="نقطه جوش آب را مقایسه کنید و بگویید فشار هوا چه تاثیری دارد")

    assert isinstance(result, AdvancedRAGResult)
    # Hybrid retriever should be queried for each sub-query
    assert mock_pipeline_components["retriever"].retrieve.await_count >= 2


@pytest.mark.asyncio
async def test_advanced_pipeline_streaming_ttft_and_citations(mock_pipeline_components):
    """
    Assert pipeline query_stream yields tokens, measures TTFT (<1.0s),
    and appends markdown page citations at the end of the stream per SC-003, FR-014.
    """
    async def mock_stream(*args, **kwargs):
        for tok in ["فرمول ", "چگالی ", "جرم ", "تقسیم بر ", "حجم است."]:
            yield tok

    mock_synth = mock_pipeline_components["synthesizer"]
    mock_synth.synthesize_stream = mock_stream

    pipeline = AdvancedRAGPipeline(
        hybrid_retriever=mock_pipeline_components["retriever"],
        rewriter=mock_pipeline_components["rewriter"],
        reranker=mock_pipeline_components["reranker"],
        expander=mock_pipeline_components["expander"],
        reorderer=mock_pipeline_components["reorderer"],
        synthesizer=mock_synth,
    )

    t0 = time.perf_counter()
    tokens = []
    async for token in pipeline.query_stream(query_text="فرمول چگالی چیست؟"):
        tokens.append(token)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    full_output = "".join(tokens)
    assert len(tokens) >= 5
    assert "چگالی" in full_output
    assert "**Sources:**" in full_output
    assert "#page=12" in full_output

    # TTFT measurement
    assert pipeline.last_ttft_ms is not None
    assert pipeline.last_ttft_ms < 1000.0  # SC-003: TTFT < 1.0s
    assert elapsed_ms < 2000.0  # Overall stream well within budget


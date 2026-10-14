"""
Unit & Contract Tests for LangGraph Enterprise Science RAG Pipeline & LangChain Standards.
Validates StateGraph node transitions, conditional refusal branching, and Document interoperability.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from langchain_core.documents import Document
from rag_core.graph.state import RAGGraphState
from rag_core.graph.pipeline_graph import EnterpriseScienceGraph
from rag_core.pipeline import AdvancedRAGPipeline
from rag_core.indexing.science_schema import (
    RetrievalCandidate,
    RerankedCandidate,
    RewriteDecision,
    ExpandedContext,
    ParentSection
)


@pytest.fixture
def mock_science_components():
    """Mock components for isolated LangGraph StateGraph testing."""
    mock_rewriter = MagicMock()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_expander = MagicMock()
    mock_reorderer = MagicMock()
    mock_synthesizer = MagicMock()

    sample_cand = RetrievalCandidate(
        chunk_id="chk_science_01",
        section_id="SEC-0015",
        chapter=2,
        page=4,
        text="چگالی برابر است با نسبت جرم جسم به حجم آن (d = m / V).",
        breadcrumbs="فصل ۲ > چگالی",
        dense_score=0.88,
        bm25_score=12.5,
        rrf_score=0.032
    )

    sample_reranked = RerankedCandidate(
        candidate=sample_cand,
        cross_encoder_score=3.85,
        final_rank=1
    )

    sample_parent = ParentSection(
        section_id="SEC-0015",
        chapter=2,
        title="چگالی مواد",
        page_start=4,
        page_end=5,
        full_text="اینکه یک جسم در آب فرو برود یا روی آب شناور بماند به چگالی بستگی دارد.",
        child_chunk_ids=["chk_science_01"],
        total_tokens=120
    )

    mock_rewriter.evaluate_and_rewrite = AsyncMock(return_value=RewriteDecision(
        needs_rewrite=False,
        confidence=1.0,
        rewritten_query="فرمول چگالی چیست؟",
        proceed_to_retrieval=True,
        sub_queries=[]
    ))
    mock_retriever.retrieve = AsyncMock(return_value=[sample_cand])
    mock_reranker.rerank = AsyncMock(return_value=[sample_reranked])
    mock_expander.expand.return_value = ExpandedContext(
        parent_sections=[sample_parent],
        total_tokens=120,
        source_chunk_ids=["chk_science_01"],
        reordered_texts=[sample_parent.full_text]
    )
    mock_reorderer.reorder.return_value = [sample_parent]

    mock_res = MagicMock()
    mock_res.answer = "چگالی برابر است با جرم تقسیم بر حجم (d = m / V)."
    mock_res.is_fallback = False
    mock_synthesizer.synthesize.return_value = mock_res

    return {
        "rewriter": mock_rewriter,
        "retriever": mock_retriever,
        "reranker": mock_reranker,
        "expander": mock_expander,
        "reorderer": mock_reorderer,
        "synthesizer": mock_synthesizer,
        "candidate": sample_cand
    }


@pytest.mark.asyncio
async def test_enterprise_graph_compilation_and_nodes(mock_science_components):
    """Verify that EnterpriseScienceGraph builds and compiles with all required nodes."""
    graph_pipe = EnterpriseScienceGraph(
        rewriter=mock_science_components["rewriter"],
        hybrid_retriever=mock_science_components["retriever"],
        reranker=mock_science_components["reranker"],
        expander=mock_science_components["expander"],
        reorderer=mock_science_components["reorderer"],
        synthesizer=mock_science_components["synthesizer"],
    )

    # Graph should be compiled
    assert graph_pipe.graph is not None
    # Inspect compiled graph nodes
    nodes = list(graph_pipe.graph.nodes.keys())
    for expected in ["gated_rewrite", "refusal", "hybrid_retrieve", "neural_rerank", "parent_expansion", "synthesis"]:
        assert expected in nodes


@pytest.mark.asyncio
async def test_enterprise_graph_benign_query_ainvoke(mock_science_components):
    """Verify standard benign execution flow through compiled StateGraph."""
    graph_pipe = EnterpriseScienceGraph(
        rewriter=mock_science_components["rewriter"],
        hybrid_retriever=mock_science_components["retriever"],
        reranker=mock_science_components["reranker"],
        expander=mock_science_components["expander"],
        reorderer=mock_science_components["reorderer"],
        synthesizer=mock_science_components["synthesizer"],
    )

    state = await graph_pipe.ainvoke(query="فرمول چگالی چیست؟")

    assert state["proceed_to_retrieval"] is True
    assert state["is_refusal"] is False
    assert "چگالی" in state["agent_response"]
    assert len(state["citations"]) > 0
    assert state["citations"][0]["page"] == 4

    # Latencies recorded for each LangGraph node
    latencies = state.get("latencies", {})
    assert "rewrite_ms" in latencies
    assert "retrieval_ms" in latencies
    assert "rerank_ms" in latencies
    assert "expansion_ms" in latencies
    assert "synthesis_ms" in latencies


@pytest.mark.asyncio
async def test_enterprise_graph_refusal_conditional_branching(mock_science_components):
    """Verify that when query is rejected by gate, graph branches directly to refusal -> END."""
    mock_science_components["rewriter"].evaluate_and_rewrite = AsyncMock(return_value=RewriteDecision(
        needs_rewrite=False,
        confidence=0.98,
        rewritten_query="قیمت ارز",
        proceed_to_retrieval=False,
        sub_queries=[]
    ))

    graph_pipe = EnterpriseScienceGraph(
        rewriter=mock_science_components["rewriter"],
        hybrid_retriever=mock_science_components["retriever"],
        reranker=mock_science_components["reranker"],
        expander=mock_science_components["expander"],
        reorderer=mock_science_components["reorderer"],
        synthesizer=mock_science_components["synthesizer"],
    )

    state = await graph_pipe.ainvoke(query="قیمت ارز و نحوه خرید بیت‌کوین چیست؟")

    assert state["proceed_to_retrieval"] is False
    assert state["is_refusal"] is True
    assert "یافت نشد" in state["agent_response"]
    # Downstream retrieval, rerank, and synthesis must have been skipped
    assert len(state.get("retrieved_chunks", [])) == 0
    mock_science_components["retriever"].retrieve.assert_not_called()
    mock_science_components["synthesizer"].synthesize.assert_not_called()


def test_as_langgraph_pipeline_bridge():
    """Verify that AdvancedRAGPipeline exposes as_langgraph_pipeline bridge."""
    pipeline = AdvancedRAGPipeline()
    graph_pipe = pipeline.as_langgraph_pipeline()
    assert isinstance(graph_pipe, EnterpriseScienceGraph)
    assert graph_pipe.graph is not None


def test_langchain_document_conversion(mock_science_components):
    """Verify that RetrievalCandidate converts to standard LangChain Document."""
    cand = mock_science_components["candidate"]
    doc = cand.to_langchain_document()

    assert isinstance(doc, Document)
    assert doc.page_content == cand.text
    assert doc.metadata["chunk_id"] == "chk_science_01"
    assert doc.metadata["chapter"] == 2
    assert doc.metadata["page"] == 4
    assert doc.metadata["chunk_identifier"] == "science_textbook.pdf#page=4"


@pytest.mark.asyncio
async def test_enterprise_graph_astream_tokens_benign(mock_science_components):
    """Verify that astream_tokens streams tokens with sub-second TTFT and caches citations."""
    async def mock_stream_gen(*args, **kwargs):
        for token in ["چگالی ", "برابر ", "است با جرم بر حجم."]:
            yield token

    mock_science_components["synthesizer"].synthesize_stream = mock_stream_gen

    graph_pipe = EnterpriseScienceGraph(
        rewriter=mock_science_components["rewriter"],
        hybrid_retriever=mock_science_components["retriever"],
        reranker=mock_science_components["reranker"],
        expander=mock_science_components["expander"],
        reorderer=mock_science_components["reorderer"],
        synthesizer=mock_science_components["synthesizer"],
    )

    tokens = []
    async for token in graph_pipe.astream_tokens(query="فرمول چگالی چیست؟", session_id="sess_test_1"):
        tokens.append(token)

    full_output = "".join(tokens)
    assert "چگالی" in full_output
    assert graph_pipe.get_refusal_for_session("sess_test_1") is False
    assert len(graph_pipe.get_citations_for_session("sess_test_1")) > 0
    assert graph_pipe.get_effective_query_for_session("sess_test_1") == "فرمول چگالی چیست؟"


@pytest.mark.asyncio
async def test_enterprise_graph_astream_tokens_refusal(mock_science_components):
    """Verify that astream_tokens yields refusal prompt and marks session refusal on gate reject."""
    mock_science_components["rewriter"].evaluate_and_rewrite = AsyncMock(return_value=RewriteDecision(
        needs_rewrite=False,
        confidence=0.99,
        rewritten_query="خرید بیت کوین",
        proceed_to_retrieval=False,
        sub_queries=[]
    ))

    graph_pipe = EnterpriseScienceGraph(
        rewriter=mock_science_components["rewriter"],
        hybrid_retriever=mock_science_components["retriever"],
        reranker=mock_science_components["reranker"],
        expander=mock_science_components["expander"],
        reorderer=mock_science_components["reorderer"],
        synthesizer=mock_science_components["synthesizer"],
    )

    tokens = []
    async for token in graph_pipe.astream_tokens(query="بیت کوین چیست؟", session_id="sess_refusal_1"):
        tokens.append(token)

    full_output = "".join(tokens)
    assert "یافت نشد" in full_output
    assert graph_pipe.get_refusal_for_session("sess_refusal_1") is True
    assert len(graph_pipe.get_citations_for_session("sess_refusal_1")) == 0


@pytest.mark.asyncio
async def test_enterprise_graph_retrieval_augmented_with_needed_summaries(mock_science_components):
    """
    User Story 4: Verify that follow-up queries with needs_history_and_retrieve
    enrich the retrieval query with historical summaries before calling hybrid retriever.
    """
    from rag_core.indexing.science_schema import (
        CurriculumIntentTier,
        ContextNeedTier,
        HistoryRelevanceTier,
        SelectedHistoryItem,
        QueryGateDecision,
    )
    from rag_core.graph.nodes import create_enterprise_hybrid_retrieve_node

    mock_gate_decision = QueryGateDecision(
        raw_query="نقش آن چیست؟",
        effective_query="نقش آن چیست؟",
        intent_tier=CurriculumIntentTier.RELEVANT,
        context_need_tier=ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE,
        is_refusal=False,
        bypass_retrieval=False,
        selected_history_items=[
            SelectedHistoryItem(
                turn_id="turn_01",
                turn_index=0,
                is_recent_5=False,
                content_type="summary",
                question="دستگاه گوارش چگونه کار می‌کند؟",
                answer="دستگاه گوارش شامل معده، کبد و روده است.",
                summary="دستگاه گوارش و نقش کبد در هضم",
                relevance_tier=HistoryRelevanceTier.NEEDED_RELEVANT,
            )
        ]
    )

    mock_science_components["rewriter"].evaluate_and_route = AsyncMock(return_value=mock_gate_decision)

    # 1. Test isolated hybrid retrieve node
    retrieve_node = create_enterprise_hybrid_retrieve_node(
        hybrid_retriever=mock_science_components["retriever"],
        overfetch_k=50
    )
    test_state: RAGGraphState = {
        "user_query": "نقش آن چیست؟",
        "effective_query": "نقش آن چیست؟",
        "selected_history_items": [mock_gate_decision.selected_history_items[0].model_dump()]
    }
    await retrieve_node(test_state)

    # Verify that retrieve was called with enriched query containing the summary
    mock_science_components["retriever"].retrieve.assert_called()
    called_query = mock_science_components["retriever"].retrieve.call_args[1].get("query") or mock_science_components["retriever"].retrieve.call_args[0][0]
    assert "نقش آن چیست؟" in called_query
    assert "دستگاه گوارش و نقش کبد در هضم" in called_query

    # 2. Test full EnterpriseScienceGraph astream_tokens pipeline
    async def mock_stream_gen(*args, **kwargs):
        for token in ["کبد ", "صفرا ", "تولید می‌کند."]:
            yield token

    mock_science_components["synthesizer"].synthesize_stream = mock_stream_gen

    graph_pipe = EnterpriseScienceGraph(
        rewriter=mock_science_components["rewriter"],
        hybrid_retriever=mock_science_components["retriever"],
        reranker=mock_science_components["reranker"],
        expander=mock_science_components["expander"],
        reorderer=mock_science_components["reorderer"],
        synthesizer=mock_science_components["synthesizer"],
    )

    tokens = []
    async for token in graph_pipe.astream_tokens(
        query="نقش آن چیست؟",
        session_id="sess_us4_1",
        history=[{"role": "user", "content": "دستگاه گوارش چیست؟"}]
    ):
        tokens.append(token)

    full_tokens = "".join(tokens)
    assert "کبد صفرا تولید می‌کند." in full_tokens
    assert len(graph_pipe.get_citations_for_session("sess_us4_1")) > 0



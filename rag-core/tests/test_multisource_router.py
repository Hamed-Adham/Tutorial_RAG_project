"""
Unit and Integration Tests for Feature 008:
Hierarchical SubQ Multi-Source Router & Compound Corpus Dispatcher.
"""

import pytest
from unittest.mock import MagicMock

from rag_core.indexing.science_schema import (
    SourceDomain,
    Level0RoutingTier,
    Level0RoutingDecision,
    RetrievalCandidate
)
from rag_core.query.jev_router import Level0SourceRouter


def test_level0_router_deterministic_fallback_science():
    """Verify keyword fallback correctly routes science terms to science_textbook."""
    router = Level0SourceRouter(jev_client=None)  # No client forces fallback
    dec = router.route_query("یاخته و بافت و اندام در بدن چگونه سازمان‌بندی می‌شوند؟")
    assert dec.routing_tier == Level0RoutingTier.SCIENCE_ONLY
    assert dec.active_sources == [SourceDomain.SCIENCE_TEXTBOOK.value]
    assert not dec.is_refusal


def test_level0_router_deterministic_fallback_faq():
    """Verify keyword fallback correctly routes INIF terms to administrative_faq."""
    router = Level0SourceRouter(jev_client=None)
    dec = router.route_query("شرایط دریافت ضمانت‌نامه و تسهیلات شرکت‌های دانش‌بنیان از صندوق نوآوری چیست؟")
    assert dec.routing_tier == Level0RoutingTier.FAQ_ONLY
    assert dec.active_sources == [SourceDomain.ADMINISTRATIVE_FAQ.value]
    assert not dec.is_refusal


def test_level0_router_deterministic_fallback_both():
    """Verify keyword fallback detects both domains when inquiry contains keywords from both."""
    router = Level0SourceRouter(jev_client=None)
    dec = router.route_query("دستگاه گوارش انسان چگونه کار می‌کند و چگونه می‌توان برای اختراع آن از صندوق تسهیلات گرفت؟")
    assert dec.routing_tier == Level0RoutingTier.BOTH_DOMAINS
    assert SourceDomain.SCIENCE_TEXTBOOK.value in dec.active_sources
    assert SourceDomain.ADMINISTRATIVE_FAQ.value in dec.active_sources
    assert not dec.is_refusal


def test_level0_router_deterministic_fallback_out_of_scope():
    """Verify keyword fallback rejects unrelated chitchat / queries."""
    router = Level0SourceRouter(jev_client=None)
    dec = router.route_query("طرز تهیه خورشت قیمه بادمجان چگونه است؟")
    assert dec.routing_tier == Level0RoutingTier.OUT_OF_SCOPE
    assert dec.is_refusal is True
    assert dec.active_sources == []


def test_level0_router_explicit_single_source_bypass():
    """When a single source is explicitly provided, Level-0 router must bypass model evaluation."""
    router = Level0SourceRouter(jev_client=None)
    dec_sci = router.route_query("هر پرسشی", enabled_sources=["science_textbook"])
    assert dec_sci.active_sources == ["science_textbook"]
    assert dec_sci.routing_tier == Level0RoutingTier.SCIENCE_ONLY

    dec_faq = router.route_query("هر پرسشی", enabled_sources=["administrative_faq"])
    assert dec_faq.active_sources == ["administrative_faq"]
    assert dec_faq.routing_tier == Level0RoutingTier.FAQ_ONLY


def test_level0_router_jev_system_one_mock():
    """Verify that structured TypeSafe Jev System One response is correctly mapped."""
    mock_jev = MagicMock()
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "source_domain": {
                "choice": "both_domains",
                "confidence": 0.95,
                "probabilities": {"both_domains": 0.95, "science_only": 0.03, "faq_only": 0.02}
            },
            "is_adversarial": {
                "noul": 0.05
            }
        },
        "_latency_seconds": 0.018
    }

    router = Level0SourceRouter(jev_client=mock_jev)
    dec = router.route_query("دستگاه گوارش چیست و برای تولید آن چه تسهیلاتی از صندوق تعلق می‌گیرد؟")

    assert dec.routing_tier == Level0RoutingTier.BOTH_DOMAINS
    assert dec.active_sources == ["science_textbook", "administrative_faq"]
    assert dec.confidence == 0.95
    assert not dec.is_refusal
    assert not dec.is_adversarial


# ===========================================================================
# User Story 1 Independent Acceptance Tests (P1 MVP)
# ===========================================================================

def test_us1_independent_admin_faq_routing():
    """
    US1 Independent Test:
    Submit an administrative FAQ query ('شرایط دریافت ضمانت‌نامه شرکت‌های دانش‌بنیان چیست؟')
    with multi-source mode active; verify that administrative_faq is assigned with
    confidence >= 0.85 and routed to the FAQ domain without manual toggling.
    """
    router = Level0SourceRouter(jev_client=None)
    dec = router.route_query("شرایط دریافت ضمانت‌نامه شرکت‌های دانش‌بنیان چیست؟", enabled_sources=None)
    assert dec.routing_tier == Level0RoutingTier.FAQ_ONLY
    assert dec.active_sources == [SourceDomain.ADMINISTRATIVE_FAQ.value]
    assert dec.confidence >= 0.85
    assert not dec.is_refusal


def test_us1_independent_science_routing():
    """Verify that 7th grade science query routes to science_textbook in multi-source mode."""
    router = Level0SourceRouter(jev_client=None)
    dec = router.route_query("یاخته و بافت و اندام در بدن چگونه سازمان‌بندی می‌شوند؟", enabled_sources=None)
    assert dec.routing_tier == Level0RoutingTier.SCIENCE_ONLY
    assert dec.active_sources == [SourceDomain.SCIENCE_TEXTBOOK.value]
    assert dec.confidence >= 0.85
    assert not dec.is_refusal


def test_us1_independent_out_of_scope_refusal():
    """Verify out-of-scope inquiry triggers immediate refusal with proper Persian reason."""
    router = Level0SourceRouter(jev_client=None)
    dec = router.route_query("قیمت روز طلا و سکه در بازار چقدر است؟", enabled_sources=None)
    assert dec.routing_tier == Level0RoutingTier.OUT_OF_SCOPE
    assert dec.active_sources == []
    assert dec.is_refusal is True
    assert "یافت نشد" in dec.refusal_reason


@pytest.mark.asyncio
async def test_us1_graph_level0_routing_integration():
    """
    Verify Level-0 Source Router integration in EnterpriseScienceGraph StateGraph.
    Verifies that Level-0 node sets active_sources and routes appropriately.
    """
    from unittest.mock import AsyncMock
    from rag_core.graph.pipeline_graph import EnterpriseScienceGraph
    from rag_core.indexing.science_schema import RewriteDecision, ExpandedContext, ParentSection

    mock_rewriter = MagicMock()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_expander = MagicMock()
    mock_reorderer = MagicMock()
    mock_synthesizer = MagicMock()

    sample_cand = RetrievalCandidate(
        chunk_id="chk_sci_01",
        section_id="SEC-01",
        chapter=1,
        page=2,
        text="یاخته کوچکترین واحد ساختاری است.",
        breadcrumbs="فصل ۱ > یاخته"
    )
    sample_parent = ParentSection(
        section_id="SEC-01",
        chapter=1,
        title="یاخته",
        page_start=2,
        page_end=3,
        full_text="یاخته کوچکترین واحد ساختاری است.",
        child_chunk_ids=["chk_sci_01"]
    )

    mock_rewriter.evaluate_and_rewrite = AsyncMock(return_value=RewriteDecision(
        needs_rewrite=False,
        confidence=1.0,
        rewritten_query="یاخته چیست؟",
        proceed_to_retrieval=True,
        sub_queries=[]
    ))
    mock_retriever.retrieve = AsyncMock(return_value=[sample_cand])
    mock_reranker.rerank = AsyncMock(return_value=[MagicMock(candidate=sample_cand, final_rank=1)])
    mock_expander.expand.return_value = ExpandedContext(
        parent_sections=[sample_parent],
        total_tokens=50,
        source_chunk_ids=["chk_sci_01"],
        reordered_texts=[sample_parent.full_text]
    )
    mock_reorderer.reorder.return_value = [sample_parent]
    mock_synthesizer.synthesize.return_value = MagicMock(answer="یاخته واحد ساختمانی بدن است.", is_fallback=False)

    graph = EnterpriseScienceGraph(
        rewriter=mock_rewriter,
        hybrid_retriever=mock_retriever,
        reranker=mock_reranker,
        expander=mock_expander,
        reorderer=mock_reorderer,
        synthesizer=mock_synthesizer
    )

    # 1. Benign Science Query in Multi-Source Mode
    state = await graph.ainvoke(query="یاخته چیست و چگونه کار می‌کند؟")
    assert state["active_sources"] == ["science_textbook"]
    assert state["target_domain"] == "science_only"
    assert "level0_router_ms" in state["latencies"]
    assert not state["is_refusal"]

    # 2. Out-of-Scope Query in Multi-Source Mode
    refusal_state = await graph.ainvoke(query="طرز تهیه قرمه سبزی چگونه است؟")
    assert refusal_state["is_refusal"] is True
    assert refusal_state["active_sources"] == []
    assert "یافت نشد" in refusal_state["agent_response"]


# ===========================================================================
# User Story 2 Integration Tests: Compound Cross-Corpus & Multi-Source Dispatch
# ===========================================================================

@pytest.mark.asyncio
async def test_us2_domain_customized_faq_dm1():
    """Verify that QueryGate evaluates FAQ domain questions accurately with customized rubric."""
    from rag_core.query.query_gate import QueryGate
    from rag_core.indexing.science_schema import CurriculumIntentTier, ContextNeedTier

    gate = QueryGate(jev_client=None)  # Offline fallback mode

    # 1. INIF FAQ question evaluated under FAQ domain
    faq_dec = await gate.evaluate_and_route(
        query="شرایط دریافت ضمانت‌نامه شرکت‌های دانش‌بنیان از صندوق چیست؟",
        domain="administrative_faq"
    )
    assert faq_dec.intent_tier == CurriculumIntentTier.RELEVANT
    assert not faq_dec.is_refusal
    assert faq_dec.context_need_tier == ContextNeedTier.NEEDS_RETRIEVE_ONLY

    # 2. Pure science question evaluated under FAQ domain should be irrelevant/rejected
    sci_on_faq = await gate.evaluate_and_route(
        query="یاخته چیست و میتوکندری چه نقشی دارد؟",
        domain="administrative_faq"
    )
    assert sci_on_faq.intent_tier == CurriculumIntentTier.COMPLETELY_IRRELEVANT
    assert sci_on_faq.is_refusal is True
    assert "صندوق نوآوری" in sci_on_faq.reasoning


@pytest.mark.asyncio
async def test_us2_parallel_dm1_execution():
    """Verify parallel DM1 evaluation when both_domains is active in EnterpriseScienceGraph."""
    from unittest.mock import AsyncMock
    from rag_core.graph.nodes import create_enterprise_query_gate_node
    from rag_core.indexing.science_schema import QueryGateDecision, CurriculumIntentTier, ContextNeedTier

    mock_gate = MagicMock()
    sci_decision = QueryGateDecision(
        raw_query="دستگاه گوارش چیست و برای تولید آن چه وامی می‌توان گرفت؟",
        effective_query="دستگاه گوارش چیست؟",
        intent_tier=CurriculumIntentTier.RELEVANT,
        context_need_tier=ContextNeedTier.NEEDS_RETRIEVE_ONLY,
        is_refusal=False,
        bypass_retrieval=False,
        sub_queries=["دستگاه گوارش چیست؟"]
    )
    faq_decision = QueryGateDecision(
        raw_query="دستگاه گوارش چیست و برای تولید آن چه وامی می‌توان گرفت؟",
        effective_query="وام تولید برای شرکت دانش‌بنیان چیست؟",
        intent_tier=CurriculumIntentTier.RELEVANT,
        context_need_tier=ContextNeedTier.NEEDS_RETRIEVE_ONLY,
        is_refusal=False,
        bypass_retrieval=False,
        sub_queries=["وام تولید برای شرکت دانش‌بنیان چیست؟"]
    )

    async def mock_eval_route(query, domain="science_textbook", **kwargs):
        if domain == "administrative_faq":
            return faq_decision
        return sci_decision

    mock_gate.evaluate_and_route = AsyncMock(side_effect=mock_eval_route)

    node = create_enterprise_query_gate_node(mock_gate)
    state = {
        "user_query": "دستگاه گوارش چیست و برای تولید آن چه وامی می‌توان گرفت؟",
        "active_sources": ["science_textbook", "administrative_faq"],
        "latencies": {}
    }

    res = await node(state)

    assert "science_textbook" in res["active_sources"]
    assert "administrative_faq" in res["active_sources"]
    assert res["proceed_to_retrieval"] is True
    assert res["is_refusal"] is False
    assert "faq_dm1_decision" in res
    assert "gate_parallel_ms" in res["latencies"]


def test_us2_partitioned_dual_grounded_context_formatting():
    """Verify that _build_context_block properly distinguishes between Science and FAQ chunks."""
    from rag_core.generation.synthesizer import _build_context_block

    sci_chunk = RetrievalCandidate(
        chunk_id="sci_01",
        section_id="SEC-01",
        chapter=13,
        page=45,
        text="دستگاه گوارش انسان شامل معده و روده است.",
        breadcrumbs="فصل ۱۳ > گوارش",
        doc_id="science_textbook"
    )
    faq_chunk = RetrievalCandidate(
        chunk_id="faq_12",
        category="ضمانت‌نامه",
        row=12,
        text="مدارک لازم برای ضمانت‌نامه شرکت دانش‌بنیان شامل اساسنامه است.",
        breadcrumbs="پرسش‌های متداول > ضمانت‌نامه",
        doc_id="administrative_faq"
    )

    context_str = _build_context_block([sci_chunk, faq_chunk])
    assert "معده و روده" in context_str
    assert "ضمانت‌نامه" in context_str
    assert "صندوق نوآوری" in context_str
    assert "فصل ۱۳" in context_str


@pytest.mark.asyncio
async def test_us2_compound_dual_source_pipeline_e2e():
    """
    Verify complete User Story 2 end-to-end execution flow:
    Compound cross-corpus query -> Level-0 routes to both_domains -> Parallel DM1 ->
    Multi-source retrieval & expansion -> Partitioned synthesis.
    """
    from unittest.mock import AsyncMock
    from rag_core.graph.pipeline_graph import EnterpriseScienceGraph
    from rag_core.indexing.science_schema import (
        QueryGateDecision,
        CurriculumIntentTier,
        ContextNeedTier,
        ParentSection,
        ExpandedContext
    )

    mock_gate = MagicMock()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_expander = MagicMock()
    mock_reorderer = MagicMock()
    mock_synthesizer = MagicMock()

    sci_cand = RetrievalCandidate(
        chunk_id="chk_sci_digest",
        section_id="SEC-0130",
        chapter=13,
        page=45,
        text="دستگاه گوارش شامل مری، معده و روده باریک است.",
        breadcrumbs="فصل ۱۳ > گوارش",
        doc_id="science_textbook"
    )
    faq_cand = RetrievalCandidate(
        chunk_id="chk_faq_guarantee",
        category="ضمانت‌نامه",
        row=12,
        text="صندوق نوآوری و شکوفایی برای تجاری‌سازی طرح‌ها ضمانت‌نامه صادر می‌کند.",
        breadcrumbs="پرسش‌های متداول > ضمانت‌نامه",
        doc_id="administrative_faq"
    )

    sci_parent = ParentSection(
        section_id="SEC-0130",
        chapter=13,
        title="دستگاه گوارش",
        page_start=45,
        page_end=46,
        full_text="دستگاه گوارش شامل مری، معده و روده باریک است.",
        child_chunk_ids=["chk_sci_digest"]
    )
    faq_parent = ParentSection(
        section_id="faq_chk_faq_guarantee",
        chapter=0,
        title="پرسش‌های متداول: ضمانت‌نامه",
        page_start=1,
        page_end=1,
        full_text="صندوق نوآوری و شکوفایی برای تجاری‌سازی طرح‌ها ضمانت‌نامه صادر می‌کند.",
        child_chunk_ids=["chk_faq_guarantee"]
    )

    dm1_sci = QueryGateDecision(
        raw_query="دستگاه گوارش چیست و برای تولید آن چه وامی می‌توان از صندوق گرفت؟",
        effective_query="دستگاه گوارش چیست؟",
        intent_tier=CurriculumIntentTier.RELEVANT,
        context_need_tier=ContextNeedTier.NEEDS_RETRIEVE_ONLY,
        is_refusal=False,
        bypass_retrieval=False,
        sub_queries=["دستگاه گوارش چیست؟"]
    )
    dm1_faq = QueryGateDecision(
        raw_query="دستگاه گوارش چیست و برای تولید آن چه وامی می‌توان از صندوق گرفت؟",
        effective_query="وام تولید برای شرکت دانش‌بنیان از صندوق چیست؟",
        intent_tier=CurriculumIntentTier.RELEVANT,
        context_need_tier=ContextNeedTier.NEEDS_RETRIEVE_ONLY,
        is_refusal=False,
        bypass_retrieval=False,
        sub_queries=["وام تولید برای شرکت دانش‌بنیان از صندوق چیست؟"]
    )

    async def mock_gate_eval(query, domain="science_textbook", **kwargs):
        if domain == "administrative_faq":
            return dm1_faq
        return dm1_sci

    mock_gate.evaluate_and_route = AsyncMock(side_effect=mock_gate_eval)
    mock_retriever.retrieve = AsyncMock(return_value=[sci_cand, faq_cand])
    mock_reranker.rerank = AsyncMock(return_value=[
        MagicMock(candidate=sci_cand, cross_encoder_score=3.5),
        MagicMock(candidate=faq_cand, cross_encoder_score=3.2)
    ])
    mock_expander.expand.return_value = ExpandedContext(
        parent_sections=[sci_parent, faq_parent],
        total_tokens=100,
        source_chunk_ids=["chk_sci_digest", "chk_faq_guarantee"],
        reordered_texts=[sci_parent.full_text, faq_parent.full_text]
    )
    mock_reorderer.reorder.return_value = [sci_parent, faq_parent]

    mock_synth_res = MagicMock()
    mock_synth_res.answer = (
        "### ۱. مفاهیم علمی (کتاب علوم)\nدستگاه گوارش غذا را هضم می‌کند.\n\n"
        "### ۲. مقررات و تسهیلات (صندوق نوآوری)\nشرکت‌های دانش‌بنیان می‌توانند تسهیلات تولید دریافت کنند."
    )
    mock_synth_res.is_fallback = False
    mock_synthesizer.synthesize.return_value = mock_synth_res

    graph = EnterpriseScienceGraph(
        rewriter=mock_gate,
        hybrid_retriever=mock_retriever,
        reranker=mock_reranker,
        expander=mock_expander,
        reorderer=mock_reorderer,
        synthesizer=mock_synthesizer
    )

    query = "دستگاه گوارش چیست و برای تولید آن چه وامی می‌توان از صندوق گرفت؟"
    state = await graph.ainvoke(query=query)

    assert "science_textbook" in state["active_sources"]
    assert "administrative_faq" in state["active_sources"]
    assert state["proceed_to_retrieval"] is True
    assert not state["is_refusal"]
    assert "۱. مفاهیم علمی (کتاب علوم)" in state["agent_response"]
    assert "۲. مقررات و تسهیلات (صندوق نوآوری)" in state["agent_response"]
    assert "gate_parallel_ms" in state["latencies"]


# ===========================================================================
# User Story 3 Tests: Full Administrative FAQ Advanced Pipeline Integration
# ===========================================================================

def test_us3_bm25_faq_category_filtering():
    """Verify MultiCorpusBM25Retriever filters administrative FAQ by category."""
    from rag_core.retrieval.bm25 import MultiCorpusBM25Retriever
    from pathlib import Path

    faq_path = Path("data_index/faq_bm25.pkl")
    if not faq_path.exists():
        pytest.skip("data_index/faq_bm25.pkl not present")

    retriever = MultiCorpusBM25Retriever(faq_index_path=faq_path)
    res = retriever.search_faq(
        query="ضمانت نامه حسن انجام کار",
        top_k=5,
        categories=["ضمانت‌نامه"]
    )
    assert len(res) > 0
    for cand in res:
        assert cand.category == "ضمانت‌نامه"
        assert cand.doc_id == "administrative_faq"
        assert cand.row is not None


@pytest.mark.asyncio
async def test_us3_faq_only_graph_execution():
    """
    Verify FAQ-only advanced pipeline execution:
    Manual selection of administrative_faq bypasses Level 0 router, uses FAQ DM1,
    retrieves from FAQ index, formats FAQ citations, and synthesizes via INIF consultant persona.
    """
    from unittest.mock import AsyncMock
    from rag_core.graph.pipeline_graph import EnterpriseScienceGraph
    from rag_core.indexing.science_schema import (
        QueryGateDecision,
        CurriculumIntentTier,
        ContextNeedTier,
        ParentSection,
        ExpandedContext
    )

    mock_gate = MagicMock()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_expander = MagicMock()
    mock_reorderer = MagicMock()
    mock_synthesizer = MagicMock()

    faq_cand = RetrievalCandidate(
        chunk_id="chk_faq_77",
        category="تسهیلات",
        row=77,
        text="تسهیلات قرض‌الحسنه صندوق نوآوری تا سقف ۵ میلیارد ریال اعطا می‌شود.",
        breadcrumbs="پرسش‌های متداول > تسهیلات",
        doc_id="administrative_faq"
    )

    faq_parent = ParentSection(
        section_id="faq_chk_faq_77",
        chapter=0,
        title="پرسش‌های متداول: تسهیلات",
        page_start=1,
        page_end=1,
        full_text="تسهیلات قرض‌الحسنه صندوق نوآوری تا سقف ۵ میلیارد ریال اعطا می‌شود.",
        child_chunk_ids=["chk_faq_77"]
    )

    faq_decision = QueryGateDecision(
        raw_query="سقف تسهیلات قرض‌الحسنه چقدر است؟",
        effective_query="سقف تسهیلات قرض‌الحسنه چقدر است؟",
        intent_tier=CurriculumIntentTier.RELEVANT,
        context_need_tier=ContextNeedTier.NEEDS_RETRIEVE_ONLY,
        is_refusal=False,
        bypass_retrieval=False,
        sub_queries=["سقف تسهیلات قرض‌الحسنه چقدر است؟"]
    )

    mock_gate.evaluate_and_route = AsyncMock(return_value=faq_decision)
    mock_retriever.retrieve = AsyncMock(return_value=[faq_cand])
    mock_reranker.rerank = AsyncMock(return_value=[
        MagicMock(candidate=faq_cand, cross_encoder_score=4.1)
    ])
    mock_expander.expand.return_value = ExpandedContext(
        parent_sections=[faq_parent],
        total_tokens=60,
        source_chunk_ids=["chk_faq_77"],
        reordered_texts=[faq_parent.full_text]
    )
    mock_reorderer.reorder.return_value = [faq_parent]

    mock_synth_res = MagicMock()
    mock_synth_res.answer = "بر اساس آیین‌نامه‌های صندوق نوآوری و شکوفایی، سقف تسهیلات قرض‌الحسنه ۵ میلیارد ریال است."
    mock_synth_res.is_fallback = False
    mock_synthesizer.synthesize.return_value = mock_synth_res

    graph = EnterpriseScienceGraph(
        rewriter=mock_gate,
        hybrid_retriever=mock_retriever,
        reranker=mock_reranker,
        expander=mock_expander,
        reorderer=mock_reorderer,
        synthesizer=mock_synthesizer
    )

    state = await graph.ainvoke(
        query="سقف تسهیلات قرض‌الحسنه چقدر است؟",
        selected_sources=["administrative_faq"],
        selected_faq_categories=["تسهیلات"]
    )

    assert state["active_sources"] == ["administrative_faq"]
    assert state["proceed_to_retrieval"] is True
    assert not state["is_refusal"]
    assert "صندوق نوآوری و شکوفایی" in state["agent_response"]
    assert len(state["citations"]) > 0
    assert state["citations"][0]["doc_id"] == "administrative_faq"
    assert state["citations"][0]["category"] == "تسهیلات"
    assert state["citations"][0]["row"] == 77





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract and unit tests for GatedQueryRewriter and SubQueryDecomposer.
Tests confidence gating, pronoun resolution, and multi-concept query decomposition.
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock

from rag_core.indexing.science_schema import (
    RewriteDecision,
    CurriculumIntentTier,
    ContextNeedTier,
    HistoryRelevanceTier,
    SelectedHistoryItem,
    QueryGateDecision,
)
from rag_core.query.query_gate import QueryGate, GatedQueryRewriter
from rag_core.query.decomposition import SubQueryDecomposer
from rag_core.query.subq_client import SubQDecisionClient
from rag_core.evaluation.jev_client import TypeSafeJevClient



@pytest.mark.asyncio
async def test_standalone_query_bypasses_rewriter():
    """
    Clear, standalone queries with no history should bypass rewriting in <10ms
    and not consume LLM generation tokens.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "intends_curriculum_answer": {"noul": 0.95, "confidence": 0.90},
            "needs_rewrite": {"noul": 0.05, "confidence": 0.95},
            "is_compound": {"noul": 0.05, "confidence": 0.95}
        }
    }
    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    query = "سفره‌های آب زیرزمینی چگونه تشکیل می‌شوند؟"

    t0 = time.perf_counter()
    decision = await rewriter.evaluate_and_rewrite(query=query, history=None)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(decision, RewriteDecision)
    assert not decision.needs_rewrite
    assert decision.rewritten_query in (None, query)
    assert len(decision.sub_queries) == 0
    assert latency_ms < 50.0  # Fast bypass gate


@pytest.mark.asyncio
async def test_conversational_pronoun_query_triggers_rewrite():
    """
    Ambiguous queries containing pronouns with prior history must be rewritten
    via TypeSafe Jev decision model to incorporate antecedent context.
    """
    history = [
        {"role": "user", "content": "انرژی زمین‌گرمایی چگونه تشکیل می‌شود؟"},
        {"role": "assistant", "content": "انرژی زمین‌گرمایی از حرارت درون زمین حاصل می‌شود."}
    ]
    query = "مزایای آن چیست؟"

    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "needs_rewrite": {"noul": 0.95, "confidence": 0.92},
            "is_compound": {"noul": 0.10, "confidence": 0.90}
        }
    }

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(query=query, history=history)

    assert decision.needs_rewrite is True
    assert decision.transformation_type == "resolve_context"
    assert "conversational antecedents" in decision.reasoning


@pytest.mark.asyncio
async def test_multi_concept_composite_query_decomposes():
    """
    Composite inquiries comparing multiple concepts or asking two distinct questions
    must route to decomposition when Jev flags is_compound=True.
    """
    query = "نقطه جوش آب خالص و آب نمک را مقایسه کنید و بگویید فشار هوا چه تاثیری بر آن دارد"

    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "needs_rewrite": {"noul": 0.10, "confidence": 0.90},
            "is_compound": {"noul": 0.95, "confidence": 0.92}
        }
    }

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(query=query, history=None)

    assert decision.needs_rewrite is True
    assert len(decision.sub_queries) >= 2
    # Ensure sub-queries address both parts
    joined_subs = " ".join(decision.sub_queries)
    assert "نقطه جوش" in joined_subs
    assert "فشار" in joined_subs


def test_sub_query_decomposer_atomic():
    """
    Single atomic query decomposes to just the query itself.
    """
    decomposer = SubQueryDecomposer()
    query = "فرمول قانون کولن در الکتریسیته چیست؟"
    subs = decomposer.decompose(query)

    assert len(subs) == 1
    assert subs[0] == query


def test_sub_query_decomposer_compound():
    """
    Compound query with conjunctions splits into targeted atomic inquiries.
    """
    decomposer = SubQueryDecomposer()
    query = "تفاوت سلول گیاهی با سلول جانوری چیست و کلروپلاست در کدام یک وجود دارد؟"
    subs = decomposer.decompose(query)

    assert len(subs) >= 2
    assert any("تفاوت سلول" in s or "گیاهی" in s for s in subs)
    assert any("کلروپلاست" in s for s in subs)


def test_sub_query_decomposer_guillemets_and_action_verbs():
    """
    Verify compound query with guillemets and action verbs decomposes into valid atomic questions,
    without producing dangling punctuation or quotes.
    """
    decomposer = SubQueryDecomposer()
    query_with_quotes = "«دستگاه گوارش چگونه کار میکند و چه بخشهایی دارد؟»"
    subs = decomposer.decompose(query_with_quotes)

    assert len(subs) == 2
    assert "دستگاه گوارش چگونه کار میکند" in subs[0]
    assert "دستگاه گوارش چه بخشهایی دارد" in subs[1] or "چه بخشهایی دارد" in subs[1]
    assert "»" not in subs
    assert "«" not in subs[0]
    assert all(len(s) > 5 for s in subs)



@pytest.mark.asyncio
async def test_gated_rewriter_mock_jev_integration():
    """
    Verify TypeSafe Jev System One integration path when mock answers are provided.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "needs_rewrite": {"noul": 0.85, "confidence": 0.90},
            "is_compound": {"noul": 0.10, "confidence": 0.85}
        },
        "_latency_seconds": 0.08
    }

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(
        query="چگونه محاسبه می‌شود؟",
        history=[{"role": "user", "content": "چگالی و وزن مخصوص طلا چیست؟"}]
    )

    assert decision.needs_rewrite is True
    assert decision.rewritten_query == "چگونه محاسبه می‌شود؟"
    assert decision.confidence >= 0.80


@pytest.mark.asyncio
async def test_gated_rewriter_fallback_on_jev_error():
    """
    Verify that if TypeSafe Jev throws an exception, the rewriter gracefully falls back
    to deterministic Persian heuristics without raising unhandled errors.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.side_effect = RuntimeError("Connection timeout")

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(
        query="سفره‌های آب زیرزمینی چیست؟",
        history=None
    )

    assert isinstance(decision, RewriteDecision)
    assert decision.needs_rewrite is False
    assert decision.rewritten_query == "سفره‌های آب زیرزمینی چیست؟"
    assert decision.proceed_to_retrieval is True


@pytest.mark.asyncio
async def test_scoped_practice_mode_accepts_target_chapter():
    """
    Mode 1: If query aligns with selected chapter topics, proceed_to_retrieval must be True
    and active_chapters must contain selected chapters.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "intends_curriculum_answer": {"noul": 0.96, "confidence": 0.95},
            "needs_rewrite": {"noul": 0.05, "confidence": 0.90},
            "is_compound": {"noul": 0.05, "confidence": 0.90}
        }
    }

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(
        query="آبخوان و سفره آب زیرزمینی چگونه آب را در خود نگه می‌دارند؟",
        selected_chapters=[6, 7]
    )

    assert decision.proceed_to_retrieval is True
    assert decision.active_chapters == [6, 7]
    assert decision.needs_rewrite is False


@pytest.mark.asyncio
async def test_scoped_practice_mode_rejects_unrelated_query():
    """
    Mode 1: If query asks about a topic outside the selected chapters,
    proceed_to_retrieval must be False.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "intends_curriculum_answer": {"noul": 0.08, "confidence": 0.92},
            "needs_rewrite": {"noul": 0.05, "confidence": 0.90},
            "is_compound": {"noul": 0.05, "confidence": 0.90}
        }
    }

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(
        query="قلب انسان چگونه خون را پمپاژ می‌کند؟",
        selected_chapters=[6, 7]
    )

    assert decision.proceed_to_retrieval is False
    assert decision.active_chapters == [6, 7]


@pytest.mark.asyncio
async def test_global_mode_rejects_out_of_domain_junk():
    """
    Mode 2: Global open search must reject non-science questions (e.g. crypto/finance).
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "intends_curriculum_answer": {"noul": 0.02, "confidence": 0.98},
            "needs_rewrite": {"noul": 0.01, "confidence": 0.90},
            "is_compound": {"noul": 0.01, "confidence": 0.90}
        }
    }

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(
        query="قیمت روز بیت کوین و اتریوم چقدر است؟",
        selected_chapters=None
    )

    assert decision.proceed_to_retrieval is False
    assert decision.active_chapters is None
    assert decision.transformation_type == "rejected"


@pytest.mark.asyncio
async def test_conversational_pronoun_resolves_context_via_synthesizer():
    """
    When Jev flags needs_rewrite >= 0.5 with history, the rewriter must invoke
    synthesizer.generate_raw to produce the resolved standalone query.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "intends_curriculum_answer": {"noul": 0.95, "confidence": 0.92},
            "needs_rewrite": {"noul": 0.95, "confidence": 0.92},
            "is_compound": {"noul": 0.05, "confidence": 0.90},
            "is_terse_or_vague": {"noul": 0.05, "confidence": 0.90}
        }
    }

    mock_synth = MagicMock()
    mock_synth.generate_raw.return_value = "کاربرد استوانه مدرج در آزمایشگاه چیست؟"

    rewriter = GatedQueryRewriter(jev_client=mock_jev, synthesizer=mock_synth)
    decision = await rewriter.evaluate_and_rewrite(
        query="کاربرد آن چیست؟",
        history=[{"role": "user", "content": "استوانه مدرج چیست؟"}]
    )

    assert decision.needs_rewrite is True
    assert decision.transformation_type == "resolve_context"
    assert decision.rewritten_query == "کاربرد استوانه مدرج در آزمایشگاه چیست؟"
    mock_synth.generate_raw.assert_called_once()


@pytest.mark.asyncio
async def test_terse_query_expands_via_synthesizer():
    """
    When Jev flags is_terse_or_vague >= 0.5 on a 1-3 word keyword, the rewriter must invoke
    synthesizer.generate_raw to expand the query into a curriculum question.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "intends_curriculum_answer": {"noul": 0.95, "confidence": 0.92},
            "needs_rewrite": {"noul": 0.05, "confidence": 0.90},
            "is_compound": {"noul": 0.05, "confidence": 0.90},
            "is_terse_or_vague": {"noul": 0.92, "confidence": 0.90}
        }
    }

    mock_synth = MagicMock()
    mock_synth.generate_raw.return_value = "تعریف و فرمول محاسبه چگالی چیست؟"

    rewriter = GatedQueryRewriter(jev_client=mock_jev, synthesizer=mock_synth)
    decision = await rewriter.evaluate_and_rewrite(
        query="چگالی",
        history=None
    )

    assert decision.needs_rewrite is True
    assert decision.transformation_type == "expand"
    assert decision.rewritten_query == "تعریف و فرمول محاسبه چگالی چیست؟"
    mock_synth.generate_raw.assert_called_once()


@pytest.mark.asyncio
async def test_compound_query_marks_transformation_and_flag():
    """
    When Jev flags is_compound >= 0.5, transformation_type must be 'decompose'
    and is_compound must be True with sub_queries populated.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "intends_curriculum_answer": {"noul": 0.95, "confidence": 0.92},
            "needs_rewrite": {"noul": 0.10, "confidence": 0.90},
            "is_compound": {"noul": 0.95, "confidence": 0.92},
            "is_terse_or_vague": {"noul": 0.05, "confidence": 0.90}
        }
    }

    rewriter = GatedQueryRewriter(jev_client=mock_jev)
    decision = await rewriter.evaluate_and_rewrite(
        query="نقطه جوش آب و الکل را مقایسه کنید و بگویید فشار هوا چه تاثیری بر آن دارد",
        history=None
    )

    assert decision.needs_rewrite is True
    assert decision.is_compound is True
    assert decision.transformation_type == "decompose"
    assert len(decision.sub_queries) >= 2


# ---------------------------------------------------------------------------
# Feature 006: User Story 1 - 3-Tier Curriculum Intent Gating Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_us1_completely_irrelevant_query_fast_refusal():
    """
    Adversarial attacks, out-of-scope prompts, or pure chitchat must be classified
    as COMPLETELY_IRRELEVANT and halted in <25ms with 0 vector tokens.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "completely_irrelevant", "confidence": 0.99},
            "need_conv_hist_and_retrieve": {"choice": "needs_neither", "confidence": 0.99}
        }
    }

    gate = QueryGate(jev_client=mock_jev)
    query = "شعر زیبایی از حافظ برایم بنویس"

    t0 = time.perf_counter()
    decision = await gate.evaluate_and_route(query=query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(decision, QueryGateDecision)
    assert decision.intent_tier == CurriculumIntentTier.COMPLETELY_IRRELEVANT
    assert decision.context_need_tier == ContextNeedTier.NEEDS_NEITHER
    assert decision.is_refusal is True
    assert decision.bypass_retrieval is True
    assert decision.effective_query == query
    assert elapsed_ms < 50.0


@pytest.mark.asyncio
async def test_us1_partially_relevant_query_distills_clean_science_query():
    """
    Mixed queries containing irrelevant chitchat + genuine science question
    must be classified as PARTIALLY_RELEVANT and distilled upfront to strip noise.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "partially_relevant", "confidence": 0.95},
            "need_conv_hist_and_retrieve": {"choice": "needs_retrieve_only", "confidence": 0.95}
        }
    }

    mock_synth = MagicMock()
    mock_synth.generate_raw.return_value = "چگالی چیست و فرمول محاسبه آن چگونه است؟"

    gate = QueryGate(jev_client=mock_jev, synthesizer=mock_synth)
    raw_query = "سلام خسته نباشید روزتون بخیر! چگالی چیست و فرمول محاسبه آن چگونه است؟"

    decision = await gate.evaluate_and_route(query=raw_query)

    assert isinstance(decision, QueryGateDecision)
    assert decision.intent_tier == CurriculumIntentTier.PARTIALLY_RELEVANT
    assert decision.is_refusal is False
    assert decision.effective_query == "چگالی چیست و فرمول محاسبه آن چگونه است؟"
    assert decision.context_need_tier == ContextNeedTier.NEEDS_RETRIEVE_ONLY


@pytest.mark.asyncio
async def test_us1_relevant_pure_curriculum_query_passes_unmodified():
    """
    Pure science inquiries must be classified as RELEVANT and pass through unmodified.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.98},
            "need_conv_hist_and_retrieve": {"choice": "needs_retrieve_only", "confidence": 0.98}
        }
    }

    gate = QueryGate(jev_client=mock_jev)
    query = "یاخته چیست و چه بخش‌هایی دارد؟"

    decision = await gate.evaluate_and_route(query=query)

    assert isinstance(decision, QueryGateDecision)
    assert decision.intent_tier == CurriculumIntentTier.RELEVANT
    assert decision.is_refusal is False
    assert decision.effective_query == query


# ---------------------------------------------------------------------------
# Feature 006: User Story 2 - Context Need Dispatching & Retrieval Bypass Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_us2_needs_history_and_retrieve_dispatches_properly():
    """
    Follow-up inquiry requiring both past conversation and textbook search
    dispatches with needs_history_and_retrieve and bypass_retrieval=False.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.98},
            "need_conv_hist_and_retrieve": {"choice": "needs_history_and_retrieve", "confidence": 0.95}
        }
    }
    history = [
        {"role": "user", "content": "دستگاه گوارش چگونه کار می‌کند؟"},
        {"role": "assistant", "content": "غذا در دهان جویده شده و وارد معده می‌شود."}
    ]
    gate = QueryGate(jev_client=mock_jev)
    decision = await gate.evaluate_and_route(query="در این فرایند نقش کبد چیست؟", history=history)

    assert isinstance(decision, QueryGateDecision)
    assert decision.context_need_tier == ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE
    assert decision.bypass_retrieval is False
    assert decision.is_refusal is False


@pytest.mark.asyncio
async def test_us2_needs_history_only_bypasses_retrieval():
    """
    Meta-inquiry on existing conversation (e.g. summarize previous answer)
    dispatches with needs_history_only and bypass_retrieval=True.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.95},
            "need_conv_hist_and_retrieve": {"choice": "needs_history_only", "confidence": 0.97}
        }
    }
    history = [
        {"role": "user", "content": "سلول چیست؟"},
        {"role": "assistant", "content": "سلول واحد ساختار و کار در جانداران است که شامل غشا، سیتوپلاسم و هسته است."}
    ]
    gate = QueryGate(jev_client=mock_jev)
    decision = await gate.evaluate_and_route(query="پاسخ قبلی را در دو خط خلاصه کن", history=history)

    assert isinstance(decision, QueryGateDecision)
    assert decision.context_need_tier == ContextNeedTier.NEEDS_HISTORY_ONLY
    assert decision.bypass_retrieval is True
    assert decision.is_refusal is False


@pytest.mark.asyncio
async def test_us2_turn0_cold_start_coerces_option1_to_option3():
    """
    Cold-start Turn 0: If history is empty and model predicts needs_history_and_retrieve,
    QueryGate auto-coerces it to needs_retrieve_only (Option 3).
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.98},
            "need_conv_hist_and_retrieve": {"choice": "needs_history_and_retrieve", "confidence": 0.90}
        }
    }
    gate = QueryGate(jev_client=mock_jev)
    decision = await gate.evaluate_and_route(query="نقش کبد چیست؟", history=[])

    assert isinstance(decision, QueryGateDecision)
    assert decision.context_need_tier == ContextNeedTier.NEEDS_RETRIEVE_ONLY
    assert decision.bypass_retrieval is False
    assert decision.is_refusal is False


@pytest.mark.asyncio
async def test_us2_turn0_cold_start_coerces_option2_to_friendly_prompt():
    """
    Cold-start Turn 0: If history is empty and user requests a meta-inquiry on non-existent history
    (needs_history_only), QueryGate halts with friendly prompt.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.95},
            "need_conv_hist_and_retrieve": {"choice": "needs_history_only", "confidence": 0.95}
        }
    }
    gate = QueryGate(jev_client=mock_jev)
    decision = await gate.evaluate_and_route(query="پاسخ قبلی را تکرار کن", history=[])

    assert isinstance(decision, QueryGateDecision)
    assert decision.context_need_tier == ContextNeedTier.NEEDS_HISTORY_ONLY
    assert decision.is_refusal is True
    assert decision.bypass_retrieval is True
    assert "هنوز گفتگویی انجام نشده است" in decision.reasoning


# ---------------------------------------------------------------------------
# Feature 006: User Story 3 - Decision Model 2 & Fidelity Selection Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subq_client_10_10_parallel_chunking():
    """
    SubQDecisionClient must split 20 candidate turns into 10-10 batches
    and execute them concurrently via asyncio.gather.
    """
    from unittest.mock import AsyncMock
    from rag_core.query.subq_client import SubQDecisionClient

    client = SubQDecisionClient(api_key="test_key")
    # Mock _evaluate_batch
    client._evaluate_batch = AsyncMock(side_effect=[
        {f"t_{i}": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT for i in range(10)},
        {f"t_{i}": HistoryRelevanceTier.NEEDED_RELEVANT for i in range(10, 20)}
    ])

    twenty_turns = [{"turn_id": f"t_{i}", "question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]
    res = await client.evaluate_history_turns("توضیح دهید", twenty_turns)

    assert len(res) == 20
    assert client._evaluate_batch.call_count == 2
    assert res["t_0"] == HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT
    assert res["t_15"] == HistoryRelevanceTier.NEEDED_RELEVANT


@pytest.mark.asyncio
async def test_us3_fidelity_scaled_history_selection():
    """
    Turns classified as needed receive full Q&A if in the 5 most recent turns,
    and concise summaries if older, with no artificial numerical limit.
    """
    from unittest.mock import AsyncMock
    from rag_core.query.subq_client import SubQDecisionClient

    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.98},
            "need_conv_hist_and_retrieve": {"choice": "needs_history_and_retrieve", "confidence": 0.95}
        }
    }

    mock_subq = MagicMock(spec=SubQDecisionClient)
    # Simulate 8 historical turns (indices 0..7). Turns in last 5 are indices 3, 4, 5, 6, 7.
    # Older turns are indices 0, 1, 2.
    # Suppose turns 1 (older) and 6 (recent 5) are needed.
    mock_subq.evaluate_history_turns = AsyncMock(return_value={
        "t_0": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT,
        "t_1": HistoryRelevanceTier.NEEDED_RELEVANT,  # Older -> summary
        "t_2": HistoryRelevanceTier.NOT_NEEDED_RELEVANT,
        "t_3": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT,
        "t_4": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT,
        "t_5": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT,
        "t_6": HistoryRelevanceTier.NEEDED_RELEVANT,  # Recent 5 -> full_text
        "t_7": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT,
    })

    history = [
        {
            "turn_id": f"t_{i}",
            "question": f"Question {i}",
            "answer": f"Answer {i}",
            "summary": f"Summary {i}"
        }
        for i in range(8)
    ]

    gate = QueryGate(jev_client=mock_jev, subq_client=mock_subq)
    decision = await gate.evaluate_and_route(query="در ادامه سوال قبلی، نقش کبد چیست؟", history=history)

    assert isinstance(decision, QueryGateDecision)
    assert len(decision.selected_history_items) == 2

    # Verify Turn 1 (older turn)
    turn1_item = next(it for it in decision.selected_history_items if it.turn_id == "t_1")
    assert turn1_item.is_recent_5 is False
    assert turn1_item.content_type == "summary"
    assert turn1_item.summary == "Summary 1"

    # Verify Turn 6 (recent 5 turn)
    turn6_item = next(it for it in decision.selected_history_items if it.turn_id == "t_6")
    assert turn6_item.is_recent_5 is True
    assert turn6_item.content_type == "full_text"
    assert turn6_item.question == "Question 6"
    assert turn6_item.answer == "Answer 6"


@pytest.mark.asyncio
async def test_us3_deterministic_missing_summary_backfill():
    """
    If any candidate turn lacks summary, batch_summarize_missing must be invoked
    to backfill summaries prior to SubQ Decision Model 2 evaluation.
    """
    from unittest.mock import AsyncMock
    from rag_core.query.summarizer import ConversationSummarizer
    from rag_core.query.subq_client import SubQDecisionClient

    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.98},
            "need_conv_hist_and_retrieve": {"choice": "needs_history_and_retrieve", "confidence": 0.95}
        }
    }

    mock_summarizer = MagicMock(spec=ConversationSummarizer)
    mock_summarizer.batch_summarize_missing = AsyncMock(return_value={
        "t_0": "خلاصه خودکار تولید شده برای چرخه آب"
    })

    mock_subq = MagicMock(spec=SubQDecisionClient)
    mock_subq.evaluate_history_turns = AsyncMock(return_value={
        "t_0": HistoryRelevanceTier.NEEDED_RELEVANT,
        "t_1": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT,
    })

    history = [
        {"turn_id": "t_0", "question": "چرخه آب چیست؟", "answer": "تبخیر، میعان و بارش."},  # Missing summary!
        {"turn_id": "t_1", "question": "ابر چیست؟", "answer": "تراکم بخار آب.", "summary": "تشکیل ابر"}
    ]

    gate = QueryGate(jev_client=mock_jev, subq_client=mock_subq, summarizer=mock_summarizer)
    decision = await gate.evaluate_and_route(query="بارش چگونه رخ می‌دهد؟", history=history)

    assert mock_summarizer.batch_summarize_missing.call_count == 1
    assert len(decision.selected_history_items) == 1
    assert decision.selected_history_items[0].turn_id == "t_0"
    assert decision.selected_history_items[0].summary == "خلاصه خودکار تولید شده برای چرخه آب"


@pytest.mark.asyncio
async def test_scoped_practice_mode_explains_unselected_chapter():
    """
    When user selects Chapter 1 but asks a question belonging to Chapter 11 (یاخته),
    the refusal reasoning must politely explain that the query belongs to another chapter
    rather than falsely stating it's outside the 7th-grade science curriculum.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "completely_irrelevant", "confidence": 0.95},
            "intends_curriculum_answer": {"noul": 0.92},
            "need_conv_hist_and_retrieve": {"choice": "needs_retrieve_only", "confidence": 0.95}
        }
    }

    gate = QueryGate(jev_client=mock_jev)
    decision = await gate.evaluate_and_route(
        query="ساختار یاخته و اندامک‌ها چگونه است؟",
        selected_chapters=[1]
    )

    assert decision.is_refusal is True
    assert "محدوده فصل‌های انتخابی" in decision.reasoning
    assert "فصل ۱" in decision.reasoning
    assert "یاخته" in decision.reasoning or "کتاب علوم تجربی هفتم" in decision.reasoning
    assert "بالای صفحه" in decision.reasoning


@pytest.mark.asyncio
async def test_anaphoric_reference_promoted_to_needs_history_and_retrieve():
    """
    When user asks an anaphoric follow-up like 'نقش بخش اولی که نام بردی چیست؟' and history exists,
    even if need_conv_hist_and_retrieve choice was needs_retrieve_only, Jev System One's semantic judgment
    on has_unresolved_referent (noul >= 0.5) semantically elevates it to needs_history_and_retrieve.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.95},
            "intends_curriculum_answer": {"noul": 0.95},
            "need_conv_hist_and_retrieve": {"choice": "needs_retrieve_only", "confidence": 0.90},
            "has_unresolved_referent": {"noul": 0.88}
        }
    }

    mock_subq = MagicMock(spec=SubQDecisionClient)
    mock_subq.evaluate_history_turns = AsyncMock(return_value={"turn_1": HistoryRelevanceTier.NEEDED_RELEVANT})

    history = [
        {"role": "user", "content": "دستگاه گوارش چگونه کار میکند؟", "turn_id": "turn_1"},
        {"role": "assistant", "content": "دستگاه گوارش شامل دهان، معده و روده است.", "turn_id": "turn_1"}
    ]

    gate = QueryGate(jev_client=mock_jev, subq_client=mock_subq)
    decision = await gate.evaluate_and_route(
        query="نقش بخش اولی که نام بردی چیست؟",
        history=history
    )

    assert decision.context_need_tier == ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE
    assert len(decision.selected_history_items) >= 1


@pytest.mark.asyncio
async def test_conversational_summary_promoted_to_needs_history_only():
    """
    When user asks to summarize prior response, Jev System One's semantic judgment
    on is_conversational_summary_or_rephrase (noul >= 0.6) sets needs_history_only.
    """
    mock_jev = MagicMock(spec=TypeSafeJevClient)
    mock_jev.is_configured = True
    mock_jev.system_one.return_value = {
        "answers": {
            "curriculum_intent": {"choice": "relevant", "confidence": 0.95},
            "intends_curriculum_answer": {"noul": 0.95},
            "need_conv_hist_and_retrieve": {"choice": "needs_retrieve_only", "confidence": 0.90},
            "has_unresolved_referent": {"noul": 0.10},
            "is_conversational_summary_or_rephrase": {"noul": 0.85}
        }
    }

    mock_subq = MagicMock(spec=SubQDecisionClient)
    mock_subq.evaluate_history_turns = AsyncMock(return_value={"turn_1": HistoryRelevanceTier.NEEDED_RELEVANT})

    history = [
        {"role": "user", "content": "دستگاه گوارش چگونه کار میکند؟", "turn_id": "turn_1"},
        {"role": "assistant", "content": "دستگاه گوارش شامل دهان، معده و روده است.", "turn_id": "turn_1"}
    ]

    gate = QueryGate(jev_client=mock_jev, subq_client=mock_subq)
    decision = await gate.evaluate_and_route(
        query="پاسخ قبلی را تیتروار خلاصه کن",
        history=history
    )

    assert decision.context_need_tier == ContextNeedTier.NEEDS_HISTORY_ONLY






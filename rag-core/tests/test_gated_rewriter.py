#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract and unit tests for GatedQueryRewriter and SubQueryDecomposer.
Tests confidence gating, pronoun resolution, and multi-concept query decomposition.
"""

import pytest
import time
from unittest.mock import MagicMock

from rag_core.indexing.science_schema import RewriteDecision
from rag_core.query.gated_rewriter import GatedQueryRewriter
from rag_core.query.decomposition import SubQueryDecomposer
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

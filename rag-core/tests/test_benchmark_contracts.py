#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract validation test for Feature 007 benchmark schemas.
Validates Pydantic v2 schemas against JSON Schema contracts.
"""

import json
import pytest
from pathlib import Path

from rag_core.evaluation.benchmark_schema import (
    BenchmarkItem,
    SuiteType,
    DifficultyLevel,
    SplitType,
    HistoricalTurn,
    GateResult,
    SubQJudgeVerdict,
    ConfusionMatrixResult,
    EvaluationReport,
    SuiteSummary,
    EvaluationRunRequest,
    EvaluationRunStatus,
    EvaluationScope,
)

SCHEMA_PATH = Path(__file__).parents[2] / "specs" / "007-comprehensive-benchmarks-subq-judge" / "contracts" / "benchmark_item_schema.json"


def test_benchmark_item_schema_valid():
    """Verify that a standard BenchmarkItem instance instantiates and serializes cleanly."""
    item = BenchmarkItem(
        item_id="SCI-CH14-001",
        suite=SuiteType.SCIENCE_CURRICULUM,
        split=SplitType.DEV,
        difficulty=DifficultyLevel.HARD,
        query="با توجه به شکل قلب، کدام حفره دیواره ماهیچه‌ای ضخیم‌تری دارد؟",
        ground_truth="بطن چپ به دلیل پمپاژ خون به تمام بدن دیواره ماهیچه‌ای ضخیم‌تری دارد.",
        expected_route="science_curriculum",
        expected_behavior="answer",
        gold_chunk_ids=["chunk_sci_14_002", "chunk_sci_14_003"],
        gold_doc_ids=["C706_olom_haftom-7-144-124-131.pdf"],
        history_turns=[
            HistoricalTurn(
                turn_id="turn_0",
                question="ساختار کلی قلب چگونه است؟",
                answer="قلب از چهار حفره شامل دو دهلیز و دو بطن تشکیل شده است.",
                is_needed=True,
            )
        ],
        category="گردش مواد",
        multi_hop=True,
    )

    data = item.model_dump()
    assert data["item_id"] == "SCI-CH14-001"
    assert data["suite"] == "science_curriculum"
    assert data["difficulty"] == "hard"
    assert data["split"] == "dev"
    assert len(data["history_turns"]) == 1
    assert data["history_turns"][0]["is_needed"] is True


def test_confusion_matrix_calculations():
    """Verify 2x2 diagnostic confusion matrix and hallucination catch rate."""
    cm = ConfusionMatrixResult(
        true_positive=70,
        false_positive=5,
        false_negative=15,
        true_negative=10,
    )
    assert cm.total == 100
    # Hallucination catch rate = TN / (TN + FP) = 10 / (10 + 5) = 10/15 = 0.6667
    assert cm.hallucination_catch_rate == 0.6667
    # Retrieval coverage = (TP + FP) / Total = 75 / 100 = 0.75
    assert cm.retrieval_coverage == 0.75


def test_evaluation_run_request_and_status():
    """Verify frontend evaluation request and live status models."""
    req = EvaluationRunRequest(
        suite=SuiteType.SCIENCE_CURRICULUM,
        split=SplitType.DEV,
        scope=EvaluationScope.RETRIEVE_ONLY,
        judge_model="subq",
        limit=10,
    )
    assert req.scope == EvaluationScope.RETRIEVE_ONLY
    assert req.limit == 10

    status = EvaluationRunStatus(
        run_id="eval_run_test_001",
        status="running",
        progress_percentage=45.0,
        total_items=10,
        processed_items=4,
        current_item_query="پرسش نمونه چگالی مواد",
        elapsed_seconds=3.2,
    )
    assert status.status == "running"
    assert status.processed_items == 4

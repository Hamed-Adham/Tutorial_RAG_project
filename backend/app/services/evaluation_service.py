#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Service (Feature 007 - US6).
Manages asynchronous background benchmark evaluations, live progress tracking,
and report caching for frontend dashboard visualization.
"""

import time
import uuid
import threading
import logging
from typing import Dict, Optional, List

from app.schemas.evaluation import (
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationRunStatus,
    SuiteListResponse,
    SuiteInfo,
)
from rag_core.evaluation.benchmark_schema import EvaluationReport
from rag_core.evaluation.run_eval import execute_benchmark_run

logger = logging.getLogger(__name__)


# In-memory store for evaluation runs
_EVALUATION_RUNS: Dict[str, EvaluationRunStatus] = {}

AVAILABLE_SUITES: List[SuiteInfo] = [
    SuiteInfo(
        id="all",
        name="All Suites (Full Golden Benchmark)",
        description="Comprehensive 360-item evaluation spanning all 5 calibrated suites",
        total_items=360,
        dev_items=90,
        test_items=270,
        difficulty_profile="20% Easy, 50% Medium, 30% Hard/Adversarial",
    ),
    SuiteInfo(
        id="science_curriculum",
        name="Grade 7 Science Curriculum",
        description="15 textbook chapters with multi-hop reasoning, diagrams, formulas, and experiments",
        total_items=120,
        dev_items=30,
        test_items=90,
        difficulty_profile="Biology, Chemistry, Physics, Geology (20% E, 50% M, 30% H)",
    ),
    SuiteInfo(
        id="administrative_faq",
        name="Administrative & Regulatory FAQ",
        description="Complex administrative inquiries, term collisions, multi-condition circulars",
        total_items=80,
        dev_items=20,
        test_items=60,
        difficulty_profile="Persian administrative jargon, loan regulations (20% E, 50% M, 30% H)",
    ),
    SuiteInfo(
        id="conversational_memory",
        name="Multi-Turn Conversational Memory",
        description="Long context dialogs (2-10 turns), pronoun resolution, Turn 0 empty history",
        total_items=60,
        dev_items=15,
        test_items=45,
        difficulty_profile="Antecedent tracking, topic shifts (20% E, 50% M, 30% H)",
    ),
    SuiteInfo(
        id="subquery_decomposition",
        name="Subquery Decomposition",
        description="Compound Persian inquiries requiring syntactic and semantic decomposition",
        total_items=50,
        dev_items=12,
        test_items=38,
        difficulty_profile="Multi-inquiry compound structures (20% E, 50% M, 30% H)",
    ),
    SuiteInfo(
        id="adversarial_refusal",
        name="Adversarial Refusal & Guardrails",
        description="Out-of-curriculum queries, false premises, prompt injection, commercial banking",
        total_items=50,
        dev_items=13,
        test_items=37,
        difficulty_profile="Strict Persian refusal formula verification (100% Adversarial)",
    ),
]


class EvaluationService:
    """Manages evaluation executions and status tracking."""

    @staticmethod
    def list_suites() -> SuiteListResponse:
        """Return available suites and their metadata."""
        return SuiteListResponse(suites=AVAILABLE_SUITES)

    @staticmethod
    def start_run(request: EvaluationRunRequest) -> EvaluationRunResponse:
        """Initiate an evaluation run asynchronously."""
        run_id = f"eval-{uuid.uuid4().hex[:8]}"

        status_obj = EvaluationRunStatus(
            run_id=run_id,
            status="queued",
            progress_percentage=0.0,
            total_items=request.limit or 10,
            processed_items=0,
            elapsed_seconds=0.0,
        )
        _EVALUATION_RUNS[run_id] = status_obj

        # Launch worker in daemon thread
        worker = threading.Thread(
            target=EvaluationService._run_worker,
            args=(run_id, request),
            daemon=True,
            name=f"eval-worker-{run_id}",
        )
        worker.start()

        return EvaluationRunResponse(
            run_id=run_id,
            status="queued",
            message=f"Evaluation run {run_id} scheduled for {request.suite or 'all suites'}",
        )

    @staticmethod
    def get_status(run_id: str) -> Optional[EvaluationRunStatus]:
        """Fetch current status of a run."""
        return _EVALUATION_RUNS.get(run_id)

    @staticmethod
    def list_runs() -> List[EvaluationRunStatus]:
        """Return all tracked runs sorted by recency."""
        return list(_EVALUATION_RUNS.values())

    @staticmethod
    def _run_worker(run_id: str, request: EvaluationRunRequest) -> None:
        """Background thread executing the benchmark evaluation."""
        status_entry = _EVALUATION_RUNS.get(run_id)
        if not status_entry:
            return

        status_entry.status = "running"
        t0 = time.perf_counter()

        def progress_callback(processed: int, total: int, current_query: str) -> None:
            status_entry.processed_items = processed
            status_entry.total_items = total
            status_entry.progress_percentage = round((processed / total) * 100.0, 1) if total > 0 else 0.0
            status_entry.current_item_query = current_query
            status_entry.elapsed_seconds = round(time.perf_counter() - t0, 2)

        try:
            suite_val = request.suite.value if request.suite else None
            split_val = request.split.value if request.split else "dev"
            scope_val = request.scope.value if request.scope else "end_to_end"

            report: EvaluationReport = execute_benchmark_run(
                suite=suite_val,
                split=split_val,
                scope=scope_val,
                judge_model=request.judge_model,
                limit=request.limit,
                pipeline_type="advanced",
                progress_callback=progress_callback,
            )

            status_entry.status = "completed"
            status_entry.progress_percentage = 100.0
            status_entry.processed_items = status_entry.total_items
            status_entry.elapsed_seconds = round(time.perf_counter() - t0, 2)
            status_entry.report = report

        except Exception as e:
            logger.exception(f"Evaluation run {run_id} failed: {e}")
            status_entry.status = "failed"
            status_entry.error = str(e)
            status_entry.elapsed_seconds = round(time.perf_counter() - t0, 2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation REST API Schemas (Feature 007 - US6).
Defines request, status, and reporting schemas for frontend-triggered evaluation runs.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field

from rag_core.evaluation.benchmark_schema import (
    EvaluationReport,
    SuiteType,
    SplitType,
    EvaluationScope,
)


class EvaluationRunRequest(BaseModel):
    """Payload to trigger a new benchmark evaluation run."""
    model_config = ConfigDict(extra="forbid")

    suite: Optional[SuiteType] = Field(
        default=None,
        description="Target suite or None for all suites (science_curriculum, administrative_faq, etc.)"
    )
    split: SplitType = Field(
        default=SplitType.DEV,
        description="Evaluation split: 'dev' (calibrated) or 'test' (sealed)"
    )
    scope: EvaluationScope = Field(
        default=EvaluationScope.END_TO_END,
        description="Execution mode: 'end_to_end', 'retrieve_only', 'synthesis_only', or 'gates_only'"
    )
    judge_model: str = Field(
        default="subq",
        description="Judge model: 'subq' (SubQ Decisions API), 'jev' (TypeSafe), 'gemini_llm'"
    )
    limit: Optional[int] = Field(
        default=10,
        ge=1,
        le=500,
        description="Max number of items to evaluate"
    )


class EvaluationRunResponse(BaseModel):
    """Response returned upon dispatching an evaluation run."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str = Field(..., description="'queued' | 'running' | 'completed' | 'failed'")
    message: str = "Evaluation run initiated successfully"


class EvaluationRunStatus(BaseModel):
    """Status polling response for an ongoing or finished evaluation run."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str = Field(..., description="'queued' | 'running' | 'completed' | 'failed'")
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    total_items: int = Field(default=0)
    processed_items: int = Field(default=0)
    current_item_query: Optional[str] = None
    elapsed_seconds: float = Field(default=0.0)
    report: Optional[EvaluationReport] = None
    error: Optional[str] = None


class SuiteInfo(BaseModel):
    """Metadata describing an available benchmark suite."""
    id: str
    name: str
    description: str
    total_items: int
    dev_items: int
    test_items: int
    difficulty_profile: str


class SuiteListResponse(BaseModel):
    """List of available benchmark suites."""
    suites: List[SuiteInfo]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic v2 Data Models and Contract Validators for Scaled Benchmark Dataset
and SubQ System One Evaluation Framework (Feature 007).
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# Feature 007 Enums & Schemas
# =============================================================================

class SuiteType(str, Enum):
    SCIENCE_CURRICULUM = "science_curriculum"
    ADMINISTRATIVE_FAQ = "administrative_faq"
    CONVERSATIONAL_MEMORY = "conversational_memory"
    SUBQUERY_DECOMPOSITION = "subquery_decomposition"
    ADVERSARIAL_REFUSAL = "adversarial_refusal"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class SplitType(str, Enum):
    DEV = "dev"
    TEST = "test"


class EvaluationScope(str, Enum):
    END_TO_END = "end_to_end"
    RETRIEVE_ONLY = "retrieve_only"
    SYNTHESIS_ONLY = "synthesis_only"
    GATES_ONLY = "gates_only"


class HistoricalTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_id: str
    question: str
    answer: str
    summary: Optional[str] = None
    is_needed: bool = False


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_id: str = Field(..., description="Unique alphanumeric identifier, e.g. SCI-CH14-001")
    suite: SuiteType = Field(..., description="Target benchmark suite")
    split: SplitType = Field(default=SplitType.TEST, description="Evaluation split (dev or test)")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Calibrated difficulty tier")

    query: str = Field(..., min_length=3, description="User input query in Persian")
    ground_truth: str = Field(..., description="Gold reference answer or expected refusal string")
    expected_route: str = Field(..., description="Expected system domain or routing target")
    expected_behavior: str = Field(default="answer", description="'answer' or 'refusal'")

    gold_chunk_ids: List[str] = Field(default_factory=list, description="List of expected chunk IDs")
    gold_doc_ids: List[str] = Field(default_factory=list, description="List of expected document file IDs")

    history_turns: List[HistoricalTurn] = Field(default_factory=list, description="Preceding turns for memory tests")

    category: Optional[str] = None
    multi_hop: bool = False
    requires_diagram: bool = False
    subqueries: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gated: bool = Field(..., description="True if metric was fully resolved deterministically")
    passed: int = Field(..., description="1 for Pass, 0 for Fail")
    metric: str = Field(..., description="Name of gate (refusal_gate, routing_gate, retrieval_hit_gate, citation_gate)")
    reason: str = Field(..., description="Deterministic resolution rationale")
    hit_at_k: int = Field(default=0, description="1 if ground truth chunk in top k, else 0")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank score (0.0 to 1.0)")
    latency_ms: float = Field(default=0.1, description="Execution duration in milliseconds")


class SubQJudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(..., description="Evaluated rubric: faithfulness, relevancy, citation, turn_relevance")
    verdict: int = Field(..., description="1 for Pass, 0 for Fail")
    score: float = Field(..., description="Continuous score or normalized probability (0.0 to 1.0)")
    confidence: float = Field(default=1.0, description="Model certainty score (0.0 to 1.0)")
    thought: str = Field(default="", description="Explanatory reasoning or failure clause")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Raw SubQ API JSON payload")
    latency_seconds: float = Field(..., description="Execution latency in seconds")
    fallback_used: bool = Field(default=False, description="True if local LLMJudge had to be invoked as fallback")
    tokens_used: int = Field(default=0, description="SubQ input tokens consumed")


class ConfusionMatrixResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true_positive: int = Field(default=0, description="Retrieval Success + Faithful Synthesis")
    false_positive: int = Field(default=0, description="Retrieval Success + Hallucination / Omission")
    false_negative: int = Field(default=0, description="Retrieval Failure + Hallucination")
    true_negative: int = Field(default=0, description="Retrieval Failure + Clean Refusal")

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def hallucination_catch_rate(self) -> float:
        denom = self.true_negative + self.false_positive
        return round(self.true_negative / denom, 4) if denom > 0 else 1.0

    @property
    def retrieval_coverage(self) -> float:
        return round((self.true_positive + self.false_positive) / self.total, 4) if self.total > 0 else 0.0


class SuiteSummary(BaseModel):
    suite: str
    total_items: int
    passed_items: int
    pass_rate: float
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    mean_latency_seconds: float = 0.0
    total_tokens: int = 0


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    timestamp: str
    split_evaluated: str
    judge_backend: str  # "subq" | "jev" | "gemini_llm"

    total_items: int
    overall_pass_rate: float
    baseline_pass_rate: float = 62.67
    score_delta: float = 0.0

    suite_summaries: Dict[str, SuiteSummary] = Field(default_factory=dict)
    confusion_matrix: ConfusionMatrixResult = Field(default_factory=ConfusionMatrixResult)

    mean_latency_seconds: float = 0.0
    p95_latency_seconds: float = 0.0
    total_cost_usd: float = 0.0

    failing_cases: List[Dict[str, Any]] = Field(default_factory=list)


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: Optional[SuiteType] = Field(default=None, description="Target suite or None for all suites")
    split: SplitType = Field(default=SplitType.DEV, description="Target split (dev or test)")
    scope: EvaluationScope = Field(default=EvaluationScope.END_TO_END, description="Evaluation execution mode")
    judge_model: str = Field(default="subq", description="'subq' | 'jev' | 'gemini_llm'")
    limit: Optional[int] = Field(default=10, ge=1, le=500, description="Max number of items to evaluate")


class EvaluationRunStatus(BaseModel):
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


# =============================================================================
# Legacy Models (Backward-Compatibility with 002)
# =============================================================================

LegacySuiteType = Literal[
    "administrative_faq",
    "science_textbook",
    "domain_routing",
    "adversarial_refusal"
]

QueryComplexityType = Literal[
    "factoid",
    "reasoning",
    "multi_hop",
    "colloquial_paraphrase",
    "hard_negative_keyword",
    "diagram",
    "experiment",
    "adversarial",
    "cross_domain_routing"
]

ExpectedBehaviorType = Literal["answer", "refusal", "disambiguate", "route_and_answer"]
ExpectedRouteType = Literal["administrative_faq", "science_textbook", "refuse", "domain_routing"]


class BenchmarkMetadataModel(BaseModel):
    suite: LegacySuiteType
    split: SplitType
    category: str = Field(min_length=1)
    query_type: QueryComplexityType
    ground_truth_doc_ids: Optional[List[str]] = Field(default_factory=list)
    ground_truth_chunks: List[str] = Field(default_factory=list)
    expected_behavior: ExpectedBehaviorType
    expected_route: ExpectedRouteType
    requires_multi_hop: Optional[bool] = False

    @field_validator("ground_truth_chunks")
    @classmethod
    def validate_chunks_not_empty(cls, v: List[str], info) -> List[str]:
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.expected_behavior != "refusal" and not self.ground_truth_chunks:
            raise ValueError("ground_truth_chunks cannot be empty for non-refusal items")


class BenchmarkItemModel(BaseModel):
    id: str = Field(min_length=3)
    input: str = Field(min_length=5)
    expected_output: str = Field(min_length=2)
    metadata: BenchmarkMetadataModel


class RetrievalMetricsModel(BaseModel):
    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    hit_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)


class GenerationMetricsModel(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    adversarial_refusal_rate: float = Field(ge=0.0, le=1.0)


class CitationMetricsModel(BaseModel):
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)


class DecoupledEvaluationReportModel(BaseModel):
    run_id: str
    timestamp: str
    total_queries: int = Field(gt=0)
    retrieval_metrics: RetrievalMetricsModel
    generation_metrics: GenerationMetricsModel
    citation_metrics: CitationMetricsModel
    suite_breakdown: Optional[Dict[str, Any]] = Field(default_factory=dict)

# Data Model: Comprehensive Benchmarks, Deterministic Gates & SubQ Judge (007)

**Feature Branch**: `007-comprehensive-benchmarks-subq-judge`  
**Date**: 2026-10-14  

---

## 1. Benchmark Data Entities

### 1.1 `BenchmarkItem` (Pydantic v2)
The authoritative data model for every test case across all 5 benchmark suites.

```python
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

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

class HistoricalTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_id: str
    question: str
    answer: str
    summary: Optional[str] = None
    is_needed: bool = False

class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    item_id: str = Field(..., description="Unique alphanumeric identifier, e.g. SCI-CH14-001")
    suite: SuiteType = Field(..., description="Target benchmark suite")
    split: SplitType = Field(default=SplitType.TEST, description="Evaluation split (dev or test)")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Calibrated difficulty tier")
    
    query: str = Field(..., min_length=3, description="User input query in Persian")
    ground_truth: str = Field(..., description="Gold reference answer or expected refusal string")
    expected_route: str = Field(..., description="Expected system domain or routing target")
    expected_behavior: str = Field(default="answer", description="'answer' or 'refusal'")
    
    # Ground Truth References for Retrieval Gates
    gold_chunk_ids: List[str] = Field(default_factory=list, description="List of expected chunk IDs")
    gold_doc_ids: List[str] = Field(default_factory=list, description="List of expected document file IDs")
    
    # Multi-Turn Conversational Context (Optional)
    history_turns: List[HistoricalTurn] = Field(default_factory=list, description="Preceding turns for memory tests")
    
    # Extended Metadata
    category: Optional[str] = None
    multi_hop: bool = False
    requires_diagram: bool = False
    subqueries: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

## 2. Evaluation Execution Entities

### 2.1 `GateResult` (Deterministic Layer 1)
Output of Layer 1 deterministic Python evaluations.

```python
class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    gated: bool = Field(..., description="True if metric was fully resolved deterministically")
    passed: int = Field(..., description="1 for Pass, 0 for Fail")
    metric: str = Field(..., description="Name of gate (refusal_gate, routing_gate, retrieval_hit_gate, citation_gate)")
    reason: str = Field(..., description="Deterministic resolution rationale")
    hit_at_k: int = Field(default=0, description="1 if ground truth chunk in top k, else 0")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank score (0.0 to 1.0)")
    latency_ms: float = Field(default=0.1, description="Execution duration in milliseconds")
```

### 2.2 `SubQJudgeVerdict` (Layer 2 System One Judge)
Structured verdict returned by the SubQ Decisions API (`subq-snap-preview`).

```python
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
```

---

## 3. Diagnostic & Reporting Entities

### 3.1 `ConfusionMatrixResult` (Decoupled 2x2 Matrix)
Diagnostic categorization for error isolation between retrieval and synthesis.

```python
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
```

### 3.2 `EvaluationReport` (Comprehensive Mentor Summary)
Complete aggregation model representing an evaluation benchmark execution.

```python
class SuiteSummary(BaseModel):
    suite: SuiteType
    total_items: int
    passed_items: int
    pass_rate: float
    hit_at_1: float
    hit_at_5: float
    mrr: float
    faithfulness: float
    answer_relevancy: float
    mean_latency_seconds: float
    total_tokens: int

class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    report_id: str
    timestamp: str
    split_evaluated: SplitType
    judge_backend: str  # "subq" | "jev" | "gemini_llm"
    
    total_items: int
    overall_pass_rate: float
    baseline_pass_rate: float
    score_delta: float
    
    suite_summaries: Dict[str, SuiteSummary]
    confusion_matrix: ConfusionMatrixResult
    
    mean_latency_seconds: float
    p95_latency_seconds: float
    total_cost_usd: float
    
    failing_cases: List[Dict[str, Any]] = Field(default_factory=list)
```

---

## 4. Frontend & Backend API Integration Entities

### 4.1 `EvaluationRunRequest` & `EvaluationRunStatus`

```python
class EvaluationScope(str, Enum):
    END_TO_END = "end_to_end"
    RETRIEVE_ONLY = "retrieve_only"
    SYNTHESIS_ONLY = "synthesis_only"
    GATES_ONLY = "gates_only"

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
```


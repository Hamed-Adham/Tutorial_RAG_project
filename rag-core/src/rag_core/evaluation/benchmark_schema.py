#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic Data Models and Contract Validators for Scaled Benchmark Dataset.
Corresponds to specs/002-benchmark-dataset-scaling/contracts/.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


SuiteType = Literal[
    "administrative_faq",
    "science_textbook",
    "domain_routing",
    "adversarial_refusal"
]

SplitType = Literal["dev", "test"]

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
    suite: SuiteType
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
        # Field validator ensures if passed empty list for non-refusal, error is raised if behavior requires it
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

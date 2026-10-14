#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoupled Retrieval vs. Generation Diagnostic Matrix (Feature 007 - US4).
Isolates retrieval failures from synthesis hallucinations using a 2x2 confusion matrix:
- TP: Retrieval Hit@k == 1 AND Faithful Synthesis == 1
- FP: Retrieval Hit@k == 1 AND Hallucination / Unfaithful Synthesis == 0
- FN: Retrieval Hit@k == 0 AND Synthesis Attempted / Failed Refusal
- TN: Retrieval Hit@k == 0 (or Adversarial Query) AND Clean Refusal

Computes Hallucination Catch Rate = TN / (TN + FP) and Retrieval Coverage.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, ConfigDict, Field

from rag_core.evaluation.benchmark_schema import (
    BenchmarkItem,
    ConfusionMatrixResult,
    SubQJudgeVerdict,
)
from rag_core.evaluation.code_gates import (
    evaluate_refusal_gate,
    evaluate_retrieval_hit_gate,
    is_standardized_refusal,
)


class DiagnosticQuadrant(str, Enum):
    """2x2 Decoupled Diagnostic Quadrants."""
    TRUE_POSITIVE = "TP"   # Retrieval Success + Faithful Synthesis
    FALSE_POSITIVE = "FP"  # Retrieval Success + Hallucination / Omission
    FALSE_NEGATIVE = "FN"  # Retrieval Failure + Hallucination / Missed Refusal
    TRUE_NEGATIVE = "TN"   # Retrieval Failure / Adversarial + Clean Refusal


class DecoupledItemResult(BaseModel):
    """Classification record for a single benchmark evaluation item."""
    model_config = ConfigDict(extra="forbid")

    item_id: str
    quadrant: DiagnosticQuadrant
    retrieval_hit: int = Field(..., description="1 if ground truth chunk in top k, else 0")
    synthesis_passed: int = Field(..., description="1 if answer is faithful or cleanly refused, else 0")
    is_refusal: bool = Field(..., description="True if answer conforms to refusal formulas")
    expected_refusal: bool = Field(..., description="True if benchmark item expects refusal")
    faithfulness_score: float = Field(default=0.0, description="Faithfulness score (0.0 to 1.0)")
    hit_at_k: int = Field(default=0)
    mrr: float = Field(default=0.0)
    diagnosis: str = Field(..., description="Human-readable attribution diagnosis")


class DecoupledEvaluator:
    """Evaluates and decouples retrieval errors from generation hallucinations."""

    def __init__(self, hit_k: int = 5, faithfulness_threshold: float = 0.7):
        self.hit_k = hit_k
        self.faithfulness_threshold = faithfulness_threshold

    def classify_item(
        self,
        item: BenchmarkItem,
        retrieved_chunk_ids: List[str],
        generated_answer: str,
        faithfulness_verdict: Optional[SubQJudgeVerdict] = None,
    ) -> DecoupledItemResult:
        """
        Classify a single query item into one of the 4 diagnostic quadrants (TP, FP, FN, TN).
        """
        # Step 1: Extract item attributes flexibly
        item_id = getattr(item, "item_id", getattr(item, "id", "unknown"))
        expected_output = getattr(item, "ground_truth", getattr(item, "expected_output", ""))
        expected_behavior = getattr(item, "expected_behavior", "answer")
        suite = getattr(item, "suite", "")
        if hasattr(suite, "value"):
            suite = suite.value

        gold_chunks = []
        if hasattr(item, "gold_chunk_ids") and item.gold_chunk_ids:
            gold_chunks = item.gold_chunk_ids
        elif hasattr(item, "metadata"):
            if isinstance(item.metadata, dict):
                gold_chunks = item.metadata.get("ground_truth_chunks", [])
            elif hasattr(item.metadata, "ground_truth_chunks"):
                gold_chunks = item.metadata.ground_truth_chunks

        hit_gate = evaluate_retrieval_hit_gate(
            gold_chunk_ids=gold_chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=self.hit_k
        )
        retrieval_hit = hit_gate.hit_at_k

        # Step 2: Determine if refusal was expected
        expected_refusal = (
            expected_behavior == "refusal" or
            suite == "adversarial_refusal" or
            len(gold_chunks) == 0
        )

        # Step 3: Check whether model answer is a clean refusal
        is_refusal = is_standardized_refusal(generated_answer)

        # Step 4: Determine faithfulness / synthesis quality
        faithfulness_score = 0.0
        if faithfulness_verdict is not None:
            faithfulness_score = faithfulness_verdict.score
        elif is_refusal and expected_refusal:
            # Clean refusal for unanswerable/adversarial is 100% faithful to context
            faithfulness_score = 1.0
        elif not is_refusal and not expected_refusal:
            # If no verdict provided, check heuristic overlap with expected output
            if expected_output and expected_output in generated_answer:
                faithfulness_score = 1.0

        is_faithful = faithfulness_score >= self.faithfulness_threshold

        # Step 5: Assign quadrant
        if expected_refusal:
            if is_refusal:
                # Correctly refused unanswerable / adversarial query
                quadrant = DiagnosticQuadrant.TRUE_NEGATIVE
                synthesis_passed = 1
                diagnosis = "Adversarial or out-of-domain query cleanly refused (True Negative)."
            else:
                # Expected refusal, but generator hallucinated an answer
                if retrieval_hit == 1:
                    quadrant = DiagnosticQuadrant.FALSE_POSITIVE
                    synthesis_passed = 0
                    diagnosis = "Spurious retrieval match led to hallucinated answer on adversarial query (False Positive)."
                else:
                    quadrant = DiagnosticQuadrant.FALSE_NEGATIVE
                    synthesis_passed = 0
                    diagnosis = "Adversarial query bypassed guardrails; generator hallucinated without relevant retrieval (False Negative)."
        else:
            # Standard in-domain query expecting an answer
            if retrieval_hit == 1:
                if is_faithful:
                    quadrant = DiagnosticQuadrant.TRUE_POSITIVE
                    synthesis_passed = 1
                    diagnosis = "End-to-end success: relevant chunk retrieved and faithful answer generated (True Positive)."
                else:
                    quadrant = DiagnosticQuadrant.FALSE_POSITIVE
                    synthesis_passed = 0
                    diagnosis = "Synthesis failure: relevant chunk retrieved, but generator hallucinated or omitted facts (False Positive)."
            else:
                # Retrieval failed (missing chunks)
                if is_refusal:
                    # Safe behavior: context was missing, so system refused instead of hallucinating
                    quadrant = DiagnosticQuadrant.TRUE_NEGATIVE
                    synthesis_passed = 1
                    diagnosis = "Graceful safety fallback: retrieval missed relevant chunks, but system cleanly refused rather than hallucinating (True Negative)."
                else:
                    # Retrieval failed, and generator hallucinated or attempted answering without context
                    quadrant = DiagnosticQuadrant.FALSE_NEGATIVE
                    synthesis_passed = 0
                    diagnosis = "Retrieval failure: relevant chunk missed, and generator produced ungrounded output (False Negative)."

        return DecoupledItemResult(
            item_id=item_id,
            quadrant=quadrant,
            retrieval_hit=retrieval_hit,
            synthesis_passed=synthesis_passed,
            is_refusal=is_refusal,
            expected_refusal=expected_refusal,
            faithfulness_score=faithfulness_score,
            hit_at_k=hit_gate.hit_at_k,
            mrr=hit_gate.mrr,
            diagnosis=diagnosis,
        )

    def compute_matrix(self, results: List[DecoupledItemResult]) -> ConfusionMatrixResult:
        """Aggregate item results into a ConfusionMatrixResult."""
        tp = sum(1 for r in results if r.quadrant == DiagnosticQuadrant.TRUE_POSITIVE)
        fp = sum(1 for r in results if r.quadrant == DiagnosticQuadrant.FALSE_POSITIVE)
        fn = sum(1 for r in results if r.quadrant == DiagnosticQuadrant.FALSE_NEGATIVE)
        tn = sum(1 for r in results if r.quadrant == DiagnosticQuadrant.TRUE_NEGATIVE)

        return ConfusionMatrixResult(
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            true_negative=tn,
        )

    def evaluate_batch(
        self,
        items: List[BenchmarkItem],
        retrieved_chunk_ids_list: List[List[str]],
        generated_answers: List[str],
        faithfulness_verdicts: Optional[List[Optional[SubQJudgeVerdict]]] = None,
    ) -> Tuple[ConfusionMatrixResult, List[DecoupledItemResult]]:
        """Evaluate a batch of benchmark items and return the aggregate matrix and details."""
        item_results: List[DecoupledItemResult] = []

        for i, item in enumerate(items):
            retrieved = retrieved_chunk_ids_list[i] if i < len(retrieved_chunk_ids_list) else []
            answer = generated_answers[i] if i < len(generated_answers) else ""
            verdict = (
                faithfulness_verdicts[i]
                if faithfulness_verdicts and i < len(faithfulness_verdicts)
                else None
            )

            result = self.classify_item(
                item=item,
                retrieved_chunk_ids=retrieved,
                generated_answer=answer,
                faithfulness_verdict=verdict,
            )
            item_results.append(result)

        matrix = self.compute_matrix(item_results)
        return matrix, item_results

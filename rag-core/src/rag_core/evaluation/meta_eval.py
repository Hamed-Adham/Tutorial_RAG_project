#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Evaluation & Production Reliability Scoring Harness for Unified RAG.
Implements the Enterprise AI Evaluation Workshop standards:
1. Meta-Evaluation of Judges (Confusion Matrix: TP, FP, TN, FN; Precision, Recall, F1)
2. Asymmetric Risk Alignment (Cancer Screening analogy: prioritizing Recall for hallucination & guardrail catches)
3. Pass^k (Pass-power-k) and Pass@k Production Reliability Scoring across multiple execution runs.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import math

@dataclass
class ConfusionMatrix:
    """
    Confusion Matrix for evaluating judge or guardrail accuracy against human ground truth.
    
    Binary Convention:
      - 1 (Positive / Pass): Response is faithful, relevant, or safe.
      - 0 (Negative / Fail): Response contains hallucination, error, or safety violation.
      
    Matrix Cells:
      - TP: Ground Truth = 1, Predicted = 1
      - FP: Ground Truth = 0, Predicted = 1 (CRITICAL ERROR: Judge missed a hallucination/violation!)
      - TN: Ground Truth = 0, Predicted = 0 (Judge correctly caught a hallucination/violation)
      - FN: Ground Truth = 1, Predicted = 0 (Judge rejected an acceptable response)
    """
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.tp + self.fp
        return (self.tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.tp + self.fn
        return (self.tp / denom) if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def hallucination_catch_rate(self) -> float:
        """
        Hallucination Detection Recall (Cancer Screening Analogy):
        Measures the judge's ability to catch true errors/hallucinations.
        Catch Rate = TN / (TN + FP)
        """
        denom = self.tn + self.fp
        return (self.tn / denom) if denom > 0 else 0.0

    @property
    def hallucination_leakage_rate(self) -> float:
        """
        Proportion of hallucinations that escaped undetected past the judge.
        Leakage Rate = FP / (TN + FP) = 1.0 - hallucination_catch_rate
        """
        denom = self.tn + self.fp
        return (self.fp / denom) if denom > 0 else 0.0

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "hallucination_catch_rate": round(self.hallucination_catch_rate, 4),
            "hallucination_leakage_rate": round(self.hallucination_leakage_rate, 4)
        }

    def render_markdown(self, title: str = "Judge Meta-Evaluation Summary") -> str:
        """Format the confusion matrix as a clean GitHub-style Markdown report."""
        lines = [
            f"### {title}",
            "",
            "| Metric | Formula | Value | Target Benchmark |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Total Evaluated** | `N` | {self.total} | - |",
            f"| **True Positives (TP)** | GT=1, Pred=1 | {self.tp} | - |",
            f"| **False Positives (FP)** | GT=0, Pred=1 (*Leakage*) | **{self.fp}** | 0 (Zero Tolerance) |",
            f"| **True Negatives (TN)** | GT=0, Pred=0 (*Caught*) | {self.tn} | High |",
            f"| **False Negatives (FN)** | GT=1, Pred=0 | {self.fn} | Low |",
            f"| **Accuracy** | `(TP + TN) / Total` | {self.accuracy * 100:.2f}% | >= 90.0% |",
            f"| **Precision** | `TP / (TP + FP)` | {self.precision * 100:.2f}% | >= 85.0% |",
            f"| **Recall** | `TP / (TP + FN)` | {self.recall * 100:.2f}% | >= 85.0% |",
            f"| **F1 Score** | `2 * (P * R) / (P + R)` | {self.f1_score * 100:.2f}% | >= 85.0% |",
            f"| **Hallucination Catch Rate** | `TN / (TN + FP)` | **{self.hallucination_catch_rate * 100:.2f}%** | >= 95.0% (Screening) |",
            f"| **Hallucination Leakage Rate** | `FP / (TN + FP)` | **{self.hallucination_leakage_rate * 100:.2f}%** | <= 5.0% |",
            ""
        ]
        return "\n".join(lines)


def compute_confusion_matrix(
    ground_truth: List[int],
    predictions: List[int]
) -> ConfusionMatrix:
    """
    Compute confusion matrix given lists of binary 0/1 ground truth and predictions.
    
    Args:
        ground_truth: List of human-verified binary labels (1 = pass, 0 = fail).
        predictions: List of LLM-as-a-judge binary predictions (1 = pass, 0 = fail).
        
    Returns:
        ConfusionMatrix instance with computed rates.
    """
    if len(ground_truth) != len(predictions):
        raise ValueError(f"Length mismatch: {len(ground_truth)} GT vs {len(predictions)} predictions")

    cm = ConfusionMatrix()
    for gt, pred in zip(ground_truth, predictions):
        gt_bool = bool(gt)
        pred_bool = bool(pred)
        if gt_bool and pred_bool:
            cm.tp += 1
        elif (not gt_bool) and pred_bool:
            cm.fp += 1
        elif (not gt_bool) and (not pred_bool):
            cm.tn += 1
        else: # gt_bool and not pred_bool
            cm.fn += 1

    return cm


# =============================================================================
# PRODUCTION RELIABILITY SCORING: Pass^k & Pass@k
# =============================================================================

@dataclass
class ReliabilityMetrics:
    """
    Multi-run reliability metrics for a test suite or single item evaluated over k runs.
    """
    k: int
    total_queries: int
    pass_power_k_count: int  # Number of queries where ALL k runs passed
    pass_power_k_rate: float # pass_power_k_count / total_queries
    pass_at_k_count: int     # Number of queries where AT LEAST ONE run passed
    pass_at_k_rate: float    # pass_at_k_count / total_queries
    mean_pass_rate: float    # Total passes across all runs / (total_queries * k)
    flaky_query_count: int   # Queries with mixed passes and fails (1 <= passes < k)

    def render_markdown(self, title: str = "Multi-Run Reliability Report") -> str:
        """Format the reliability metrics as Markdown."""
        lines = [
            f"### {title} (k={self.k} runs)",
            "",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
            f"| **Evaluated Queries (N)** | {self.total_queries} | Total independent benchmark queries |",
            f"| **Runs per Query (k)** | {self.k} | Number of repeated stochastic trials |",
            f"| **Pass^k (Pass-power-{self.k})** | **{self.pass_power_k_rate * 100:.2f}%** ({self.pass_power_k_count}/{self.total_queries}) | Queries passing 100% of {self.k} trials (Mission-Critical) |",
            f"| **Pass@{self.k}** | {self.pass_at_k_rate * 100:.2f}% ({self.pass_at_k_count}/{self.total_queries}) | Queries passing at least 1 trial |",
            f"| **Mean Pass Rate** | {self.mean_pass_rate * 100:.2f}% | Overall average success rate across all N * k trials |",
            f"| **Flaky Queries** | {self.flaky_query_count} ({self.flaky_query_count / max(self.total_queries, 1) * 100:.2f}%) | Queries exhibiting inconsistent nondeterministic outcomes |",
            ""
        ]
        return "\n".join(lines)


def calculate_pass_power_k(
    multi_run_results: List[List[int]]
) -> Tuple[int, float]:
    """
    Calculate Pass^k (Pass-power-k): Fraction of queries where ALL k runs passed.
    
    Args:
        multi_run_results: List of runs per query, e.g. [[1, 1, 1], [1, 0, 1], [0, 0, 0]].
        
    Returns:
        (passing_query_count, passing_ratio)
    """
    if not multi_run_results:
        return 0, 0.0

    total_queries = len(multi_run_results)
    perfect_passes = 0

    for query_runs in multi_run_results:
        if len(query_runs) == 0:
            continue
        if all(r == 1 for r in query_runs):
            perfect_passes += 1

    return perfect_passes, perfect_passes / total_queries


def calculate_pass_at_k(
    multi_run_results: List[List[int]],
    k: Optional[int] = None
) -> Tuple[int, float]:
    """
    Calculate Pass@k: Fraction of queries where AT LEAST ONE run passed.
    
    Args:
        multi_run_results: List of runs per query.
        k: Optional slice limit (default: evaluates all available runs per query).
        
    Returns:
        (passing_query_count, passing_ratio)
    """
    if not multi_run_results:
        return 0, 0.0

    total_queries = len(multi_run_results)
    any_pass = 0

    for query_runs in multi_run_results:
        runs_to_eval = query_runs[:k] if k is not None else query_runs
        if not runs_to_eval:
            continue
        if any(r == 1 for r in runs_to_eval):
            any_pass += 1

    return any_pass, any_pass / total_queries


def evaluate_reliability_harness(
    multi_run_results: List[List[int]]
) -> ReliabilityMetrics:
    """
    Complete production reliability harness computation across multi-run results.
    
    Args:
        multi_run_results: Matrix of shape (N_queries, k_runs), each element is 0 or 1.
        
    Returns:
        ReliabilityMetrics dataclass with all production reliability metrics.
    """
    if not multi_run_results:
        return ReliabilityMetrics(0, 0, 0, 0.0, 0, 0.0, 0.0, 0)

    total_queries = len(multi_run_results)
    k = len(multi_run_results[0])
    
    perfect_count = 0
    at_least_one_count = 0
    flaky_count = 0
    total_passes = 0
    total_trials = 0

    for runs in multi_run_results:
        passes = sum(1 for r in runs if r == 1)
        total_passes += passes
        total_trials += len(runs)

        if passes == len(runs):
            perfect_count += 1
        elif passes > 0:
            flaky_count += 1

        if passes > 0:
            at_least_one_count += 1

    mean_rate = (total_passes / total_trials) if total_trials > 0 else 0.0

    return ReliabilityMetrics(
        k=k,
        total_queries=total_queries,
        pass_power_k_count=perfect_count,
        pass_power_k_rate=perfect_count / total_queries,
        pass_at_k_count=at_least_one_count,
        pass_at_k_rate=at_least_one_count / total_queries,
        mean_pass_rate=mean_rate,
        flaky_query_count=flaky_count
    )


if __name__ == "__main__":
    # Self-test demonstration of Meta-Evaluation & Reliability Harness
    print("=== 1. Meta-Evaluation Self-Test ===")
    human_gt = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0] # 5 positive, 5 negative
    judge_p  = [1, 1, 1, 1, 0, 1, 0, 1, 0, 0] # Judge leaked 1 hallucination (FP=1)
    cm = compute_confusion_matrix(human_gt, judge_p)
    print(cm.render_markdown("Self-Test Judge Meta-Evaluation"))

    print("=== 2. Multi-Run Reliability Self-Test ===")
    # 5 queries tested 3 times each
    synthetic_multi_runs = [
        [1, 1, 1], # Query 1: 100% reliable
        [1, 1, 1], # Query 2: 100% reliable
        [1, 0, 1], # Query 3: Flaky
        [0, 1, 0], # Query 4: Flaky
        [0, 0, 0], # Query 5: Consistent failure
    ]
    rel = evaluate_reliability_harness(synthetic_multi_runs)
    print(rel.render_markdown("Self-Test Reliability Harness"))

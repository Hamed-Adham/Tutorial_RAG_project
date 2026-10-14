#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Mentor-Ready Evaluation Report Generator (Feature 007 - US5).
Produces publication-grade GitHub-flavored Markdown and JSON reports comparing:
- Ground Zero Naive Baseline vs. Current Run vs. Target KPIs
- 2x2 Decoupled Confusion Matrix & Hallucination Catch Rate
- 5-Suite Granular Performance Scorecard
- Deep-dive Failure Taxonomy with Langfuse Telemetry Links

CRITICAL: Strict adherence to plain-text / unicode math formulas (zero raw LaTeX).
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from rag_core.evaluation.benchmark_schema import (
    EvaluationReport,
    SuiteSummary,
    ConfusionMatrixResult,
)

# Reference Ground Zero Baseline and Mentor Target KPI definitions
GROUND_ZERO_BASELINE = {
    "pass_rate": 62.67,
    "hit_at_1": 41.2,
    "hit_at_5": 68.4,
    "mrr": 0.512,
    "faithfulness": 71.0,
    "answer_relevancy": 74.5,
    "hallucination_catch_rate": 52.0,
    "refusal_rate": 48.0,
    "mean_latency_seconds": 4.80,
}

TARGET_KPIS = {
    "pass_rate": 88.0,
    "hit_at_1": 78.0,
    "hit_at_5": 92.0,
    "mrr": 0.820,
    "faithfulness": 90.0,
    "answer_relevancy": 88.0,
    "hallucination_catch_rate": 95.0,
    "refusal_rate": 96.0,
    "mean_latency_seconds": 2.00,
}


def _delta_badge(current: float, baseline: float, is_latency: bool = False, is_mrr: bool = False) -> str:
    """Format delta string with indicator symbol."""
    delta = current - baseline
    if is_latency:
        # Lower latency is better
        if delta < -0.1:
            return f"🟢 {delta:.2f}s"
        elif delta > 0.1:
            return f"🔴 +{delta:.2f}s"
        return f"⚪ {delta:.2f}s"

    if is_mrr:
        if delta > 0.02:
            return f"🟢 +{delta:.3f}"
        elif delta < -0.02:
            return f"🔴 {delta:.3f}"
        return f"⚪ {delta:.3f}"

    # Percentages
    if delta > 1.0:
        return f"🟢 +{delta:.1f}%"
    elif delta < -1.0:
        return f"🔴 {delta:.1f}%"
    return f"⚪ {delta:.1f}%"


class ReportGenerator:
    """Compiles evaluation run results into executive Markdown reports."""

    def __init__(self, ground_zero: Optional[Dict[str, Any]] = None, targets: Optional[Dict[str, Any]] = None):
        self.ground_zero = ground_zero or GROUND_ZERO_BASELINE
        self.targets = targets or TARGET_KPIS

    def generate_markdown(self, report: EvaluationReport) -> str:
        """
        Generate a comprehensive GitHub-Flavored Markdown report.
        """
        cm = report.confusion_matrix
        tot = cm.total if cm.total > 0 else 1
        hcr_pct = cm.hallucination_catch_rate * 100.0
        cov_pct = cm.retrieval_coverage * 100.0

        # Calculate overall Hit@1, Hit@5, MRR, Faithfulness, Relevancy from suite summaries if present
        suites = report.suite_summaries
        if suites:
            total_items = sum(s.total_items for s in suites.values()) or 1
            avg_hit1 = sum(s.hit_at_1 * s.total_items for s in suites.values()) / total_items * 100.0
            avg_hit5 = sum(s.hit_at_5 * s.total_items for s in suites.values()) / total_items * 100.0
            avg_mrr = sum(s.mrr * s.total_items for s in suites.values()) / total_items
            avg_faith = sum(s.faithfulness * s.total_items for s in suites.values()) / total_items * 100.0
            avg_rel = sum(s.answer_relevancy * s.total_items for s in suites.values()) / total_items * 100.0
        else:
            avg_hit1 = 0.0
            avg_hit5 = 0.0
            avg_mrr = 0.0
            avg_faith = 0.0
            avg_rel = 0.0

        ts_formatted = report.timestamp[:19].replace("T", " ")

        passed_items = sum(s.passed_items for s in suites.values())
        lines = [
            f"# RAG Benchmark & Evaluation Executive Report",
            f"",
            f"**Report ID**: `{report.report_id}`  ",
            f"**Execution Timestamp**: `{ts_formatted}`  ",
            f"**Split**: `{report.split_evaluated.upper()}` | **Total Items**: `{report.total_items}` | **Passed**: `{passed_items}`  ",
            f"**Overall Pass Rate**: `{report.overall_pass_rate:.1f}%` ({_delta_badge(report.overall_pass_rate, self.ground_zero['pass_rate'])})  ",
            f"",
            f"---",
            f"",
            f"## 1. Executive KPI Scorecard",
            f"",
            f"Comparison of Ground Zero Naive Baseline vs. Current Production Architecture vs. Mentor Target KPIs:",
            f"",
            f"| Metric | Ground Zero Baseline | Current Run | Target KPI | Delta vs Baseline | Status |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Overall Pass Rate** | {self.ground_zero['pass_rate']:.1f}% | **{report.overall_pass_rate:.1f}%** | {self.targets['pass_rate']:.1f}% | {_delta_badge(report.overall_pass_rate, self.ground_zero['pass_rate'])} | {'✅ Reached' if report.overall_pass_rate >= self.targets['pass_rate'] else '⏳ In Progress'} |",
            f"| **Retrieval Hit@1** | {self.ground_zero['hit_at_1']:.1f}% | **{avg_hit1:.1f}%** | {self.targets['hit_at_1']:.1f}% | {_delta_badge(avg_hit1, self.ground_zero['hit_at_1'])} | {'✅ Reached' if avg_hit1 >= self.targets['hit_at_1'] else '⏳ In Progress'} |",
            f"| **Retrieval Hit@5** | {self.ground_zero['hit_at_5']:.1f}% | **{avg_hit5:.1f}%** | {self.targets['hit_at_5']:.1f}% | {_delta_badge(avg_hit5, self.ground_zero['hit_at_5'])} | {'✅ Reached' if avg_hit5 >= self.targets['hit_at_5'] else '⏳ In Progress'} |",
            f"| **Mean Reciprocal Rank (MRR)** | {self.ground_zero['mrr']:.3f} | **{avg_mrr:.3f}** | {self.targets['mrr']:.3f} | {_delta_badge(avg_mrr, self.ground_zero['mrr'], is_mrr=True)} | {'✅ Reached' if avg_mrr >= self.targets['mrr'] else '⏳ In Progress'} |",
            f"| **Synthesis Faithfulness** | {self.ground_zero['faithfulness']:.1f}% | **{avg_faith:.1f}%** | {self.targets['faithfulness']:.1f}% | {_delta_badge(avg_faith, self.ground_zero['faithfulness'])} | {'✅ Reached' if avg_faith >= self.targets['faithfulness'] else '⏳ In Progress'} |",
            f"| **Answer Relevancy** | {self.ground_zero['answer_relevancy']:.1f}% | **{avg_rel:.1f}%** | {self.targets['answer_relevancy']:.1f}% | {_delta_badge(avg_rel, self.ground_zero['answer_relevancy'])} | {'✅ Reached' if avg_rel >= self.targets['answer_relevancy'] else '⏳ In Progress'} |",
            f"| **Hallucination Catch Rate** | {self.ground_zero['hallucination_catch_rate']:.1f}% | **{hcr_pct:.1f}%** | {self.targets['hallucination_catch_rate']:.1f}% | {_delta_badge(hcr_pct, self.ground_zero['hallucination_catch_rate'])} | {'✅ Reached' if hcr_pct >= self.targets['hallucination_catch_rate'] else '⏳ In Progress'} |",
            f"| **Mean Pipeline Latency** | {self.ground_zero['mean_latency_seconds']:.2f}s | **{report.mean_latency_seconds:.2f}s** | < {self.targets['mean_latency_seconds']:.2f}s | {_delta_badge(report.mean_latency_seconds, self.ground_zero['mean_latency_seconds'], is_latency=True)} | {'✅ Reached' if report.mean_latency_seconds <= self.targets['mean_latency_seconds'] else '⏳ In Progress'} |",
            f"",
            f"---",
            f"",
            f"## 2. Decoupled 2x2 Diagnostic Confusion Matrix",
            f"",
            f"Disentangling retrieval failures (missing context) from synthesis hallucinations:",
            f"",
            f"| | **Retrieved (Hit@5 = 1)** | **Missed (Hit@5 = 0)** |",
            f"| :--- | :---: | :---: |",
            f"| **Faithful / Valid** | **True Positive (TP)**: `{cm.true_positive}` items | **True Negative (TN)**: `{cm.true_negative}` items |",
            f"| **Hallucination / Failure** | **False Positive (FP)**: `{cm.false_positive}` items | **False Negative (FN)**: `{cm.false_negative}` items |",
            f"",
            f"> [!NOTE]",
            f"> **Diagnostic Formulations**:",
            f"> - `Hallucination Catch Rate = TN / (TN + FP) = {cm.true_negative} / ({cm.true_negative} + {cm.false_positive}) = {hcr_pct:.1f}%`",
            f"> - `Retrieval Coverage = (TP + FP) / Total = ({cm.true_positive} + {cm.false_positive}) / {cm.total} = {cov_pct:.1f}%`",
            f"",
            f"---",
            f"",
            f"## 3. Calibrated 5-Suite Breakdown",
            f"",
            f"| Suite | Items | Passed | Pass Rate | Hit@1 | Hit@5 | MRR | Faithfulness | Relevancy | Mean Latency |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for suite_name, summary in suites.items():
            lines.append(
                f"| `{suite_name}` | {summary.total_items} | {summary.passed_items} | "
                f"**{summary.pass_rate:.1f}%** | {summary.hit_at_1 * 100:.1f}% | {summary.hit_at_5 * 100:.1f}% | "
                f"{summary.mrr:.3f} | {summary.faithfulness * 100:.1f}% | {summary.answer_relevancy * 100:.1f}% | "
                f"{summary.mean_latency_seconds:.2f}s |"
            )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Failure Taxonomy & Deep Dive",
            f"",
        ])

        if not report.failing_cases:
            lines.append("🎉 **No failing cases recorded in this evaluation run!** All test items passed.")
        else:
            lines.append(f"Recorded **{len(report.failing_cases)}** failing items. Displaying top instances for root-cause diagnosis:")
            lines.append("")

            for idx, fail in enumerate(report.failing_cases[:5], 1):
                item_id = fail.get("item_id", f"case-{idx}")
                suite = fail.get("suite", "unknown")
                query = fail.get("query", "")
                expected = fail.get("expected", "")
                actual = fail.get("actual", "")
                reason = fail.get("reason", "Unspecified failure")
                trace_url = fail.get("trace_url", "https://cloud.langfuse.com")

                lines.extend([
                    f"### Case {idx}: `{item_id}` (`{suite}`)",
                    f"- **Query**: {query}",
                    f"- **Expected Output**: {expected}",
                    f"- **Actual Model Output**: {actual}",
                    f"- **Diagnostic Rationale**: {reason}",
                    f"- **Langfuse Telemetry**: [View Execution Trace]({trace_url})",
                    f"",
                ])

        lines.extend([
            f"---",
            f"",
            f"*Generated automatically by Unified RAG Evaluation Suite (SubQ Decisions System One Judge & Deterministic Code Gates).*  ",
        ])

        return "\n".join(lines)

    def save_markdown(self, report: EvaluationReport, output_path: Union[str, Path]) -> Path:
        """Export generated markdown report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_markdown(report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def save_json(self, report: EvaluationReport, output_path: Union[str, Path]) -> Path:
        """Export evaluation report model to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        return path

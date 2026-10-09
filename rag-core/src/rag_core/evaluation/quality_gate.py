"""
Stage 10: Quality Gate — Threshold Validator

Checks that evaluation metric means meet the constitutional thresholds.
Can be used both programmatically and as a CI/CD gate.

Thresholds (Constitution v1.2.0, Principle II):
  Faithfulness      >= 0.80  (prevents hallucinations)
  Answer Relevancy  >= 0.75  (pertinence to query)
  Context Precision >= 0.75  (relevance of retrieved chunks)
  Context Recall    >= 0.70  (coverage of necessary information)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QualityGateConfig:
    """Threshold configuration for the quality gate."""

    faithfulness: float = 0.80
    answer_relevancy: float = 0.75
    context_precision: float = 0.75
    context_recall: float = 0.70
    hit_at_3: float = 0.70
    mrr: float = 0.60
    retrieval_only: bool = False


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class GateCheckResult:
    """Result of a quality gate check against an evaluation report."""

    passed: bool
    """True if all metrics meet their configured thresholds."""

    failures: List[str] = field(default_factory=list)
    """Human-readable list of failing metrics with actual vs. threshold values."""

    scores: Dict[str, float] = field(default_factory=dict)
    """Actual metric values from the report."""

    thresholds: Dict[str, float] = field(default_factory=dict)
    """Configured threshold values."""

    def summary_str(self) -> str:
        """Return a compact CLI-friendly summary."""
        icon = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [f"Quality Gate: {icon}"]
        metrics = [
            ("Faithfulness", "faithfulness"),
            ("Ans Relevancy", "answer_relevancy"),
            ("Ctx Precision", "context_precision"),
            ("Ctx Recall", "context_recall"),
            ("Hit@3", "hit_at_3"),
            ("MRR", "mrr"),
        ]
        for label, key in metrics:
            if key in self.scores:
                score = self.scores.get(key, 0.0)
                threshold = self.thresholds.get(key, 0.0)
                status = "✓" if score >= threshold else "✗"
                lines.append(f"  {status} {label}: {score:.3f} (>= {threshold:.2f})")
        if self.failures:
            lines.append("\nFailing metrics:")
            for f in self.failures:
                lines.append(f"  - {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QualityGate
# ---------------------------------------------------------------------------


class QualityGate:
    """
    Validates evaluation metric means against constitutional thresholds.

    Example::

        gate = QualityGate()
        result = gate.check_report(report)
        if not result.passed:
            print(result.summary_str())
            raise SystemExit(1)
    """

    def __init__(self, config: Optional[QualityGateConfig] = None) -> None:
        self._cfg = config or QualityGateConfig()

    def check(self, report: Any) -> bool:
        """
        Check an ``EvaluationReport`` against configured thresholds.

        Convenience method that returns a plain ``bool``.
        Use ``check_report`` for the full ``GateCheckResult``.
        """
        return self.check_report(report).passed

    def check_report(self, report: Any) -> GateCheckResult:
        """
        Check an ``EvaluationReport`` and return a detailed ``GateCheckResult``.

        Args:
            report: An ``EvaluationReport`` object.

        Returns:
            ``GateCheckResult`` describing the pass/fail status.
        """
        is_retrieval_only = self._cfg.retrieval_only or getattr(report, "retrieval_only", False)

        if is_retrieval_only:
            scores = {
                "hit_at_3": getattr(report, "mean_hit_at_3", 0.0),
                "mrr": getattr(report, "mean_mrr", 0.0),
                "context_recall": getattr(report, "mean_context_recall", 0.0),
            }
            thresholds = {
                "hit_at_3": self._cfg.hit_at_3,
                "mrr": self._cfg.mrr,
                "context_recall": self._cfg.context_recall,
            }
        else:
            scores = {
                "faithfulness": getattr(report, "mean_faithfulness", 0.0),
                "answer_relevancy": getattr(report, "mean_answer_relevancy", 0.0),
                "context_precision": getattr(report, "mean_context_precision", 0.0),
                "context_recall": getattr(report, "mean_context_recall", 0.0),
            }
            thresholds = {
                "faithfulness": self._cfg.faithfulness,
                "answer_relevancy": self._cfg.answer_relevancy,
                "context_precision": self._cfg.context_precision,
                "context_recall": self._cfg.context_recall,
            }

        failures = []
        for metric, actual in scores.items():
            required = thresholds[metric]
            if actual < required:
                failures.append(
                    f"{metric}: {actual:.3f} < {required:.2f} (delta: {actual - required:+.3f})"
                )

        result = GateCheckResult(
            passed=len(failures) == 0,
            failures=failures,
            scores=scores,
            thresholds=thresholds,
        )

        if result.passed:
            logger.info("Quality gate PASSED: all metrics meet thresholds")
        else:
            logger.warning("Quality gate FAILED:\n%s", "\n".join(failures))

        return result

    def check_dict(self, metrics: Dict[str, float]) -> GateCheckResult:
        """
        Check a plain ``{metric_name: value}`` dict against thresholds.

        Key names accepted: ``faithfulness``, ``answer_relevancy``,
        ``context_precision``, ``context_recall`` (and ``mean_`` prefixed variants).

        Useful for checking individual RAGAS adapter outputs.
        """
        # Normalise keys by stripping "mean_" prefix
        normalised = {}
        for k, v in metrics.items():
            key = k.replace("mean_", "")
            normalised[key] = v

        class _FakeReport:
            pass

        rpt = _FakeReport()
        for key, val in normalised.items():
            setattr(rpt, f"mean_{key}", val)

        return self.check_report(rpt)

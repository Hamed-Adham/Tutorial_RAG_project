"""
Stage 10: CLI Evaluation Entrypoint

Provides a command-line interface for running full RAGAS evaluations
against the golden benchmark and managing baseline snapshots.

Usage:
    python -m rag_core.evaluation.run_eval [OPTIONS]

Options:
    --suite        Filter to a specific test suite (science, admin_faq, ...)
    --limit        Maximum number of items to evaluate
    --save         Save the evaluation report as the new baseline
    --compare      Compare against the last saved baseline
    --output       Output file path for the JSON report
    --gates-only   Run Layer 1 deterministic code gates only (fast, 0 tokens)

Constitution v1.2.0:
- run_eval is the CLI entry point for Tier 3 (full semantic evaluation).
- Gate-only mode corresponds to Tier 2 (make test-gates, < 2s, 0 tokens).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline manager (simple JSON snapshot)
# ---------------------------------------------------------------------------


class BaselineManager:
    """
    Saves and compares evaluation report snapshots.

    Baselines are stored as JSON files in ``rag-core/src/rag_core/evaluation/baselines/``.
    Each new baseline overwrites the previous ``baseline_latest.json``.
    Historical snapshots are stored with a timestamp suffix.

    Example::

        manager = BaselineManager()
        manager.save(report)
        delta = manager.compare(report)
    """

    DEFAULT_DIR = Path(__file__).parent / "baselines"

    def __init__(self, baseline_dir: Optional[Path] = None) -> None:
        self._dir = baseline_dir or self.DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._latest_path = self._dir / "baseline_latest.json"

    def save(self, report: "EvaluationReport") -> Path:  # type: ignore[name-defined]  # noqa
        """
        Save *report* as the new baseline and a timestamped historical copy.

        Returns:
            Path to the saved ``baseline_latest.json``.
        """
        data = report.to_dict() if hasattr(report, "to_dict") else {}
        ts = report.timestamp.replace(":", "-").replace("+", "_")[:19]
        snapshot_path = self._dir / f"baseline_{ts}.json"

        with open(self._latest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Baseline saved to %s", self._latest_path)
        print(f"✅ Baseline saved: {self._latest_path}")
        return self._latest_path

    def load_latest(self) -> Optional[dict]:
        """Load the latest baseline snapshot as a plain dict."""
        if not self._latest_path.exists():
            logger.warning("No baseline found at %s", self._latest_path)
            return None
        with open(self._latest_path, encoding="utf-8") as f:
            return json.load(f)

    def compare(self, report: "EvaluationReport") -> dict:  # type: ignore[name-defined]
        """
        Compare *report* against the latest baseline.

        Returns:
            Dict of metric deltas (positive = improvement).
        """
        baseline = self.load_latest()
        if not baseline:
            print("⚠️  No baseline found. Save one first with --save.")
            return {}

        metrics = [
            "mean_faithfulness",
            "mean_answer_relevancy",
            "mean_context_precision",
            "mean_context_recall",
        ]
        current = report.to_dict() if hasattr(report, "to_dict") else {}
        deltas = {}

        print("\n📊 Baseline Comparison")
        print("-" * 50)
        for m in metrics:
            base_val = baseline.get(m, 0.0)
            curr_val = current.get(m, 0.0)
            delta = curr_val - base_val
            deltas[m] = delta
            arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "~")
            label = m.replace("mean_", "").replace("_", " ").title()
            print(f"  {arrow} {label}: {curr_val:.3f} (baseline: {base_val:.3f}, delta: {delta:+.3f})")
        print("-" * 50)
        return deltas


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for run_eval."""
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Run full RAGAS evaluation or deterministic code gates.",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Filter to a specific test suite (science, admin_faq, domain_routing, adversarial_refusal)",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["dev", "test"],
        help="Filter items by split partition ('dev' or 'test')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of benchmark items to evaluate",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save evaluation results as the new baseline snapshot",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results against the latest baseline",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save the JSON evaluation report to this file path",
    )
    parser.add_argument(
        "--gates-only",
        action="store_true",
        help="Run only Layer 1 deterministic code gates (fast, 0 tokens)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only deterministic retrieval evaluation (Hit@k, MRR, Context Recall) with zero LLM judge tokens",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the golden benchmark JSONL file",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the auto-generated evaluation run ID",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def run_gates_only(args: argparse.Namespace) -> int:
    """Run Layer 1 deterministic code gates and exit."""
    from rag_core.evaluation.code_gates import evaluate_code_gate
    from rag_core.evaluation.datasets.load_dataset import load_golden_dataset

    items = load_golden_dataset(split=None)
    if args.suite:
        items = [it for it in items if it.get("suite") == args.suite]
    if args.limit:
        items = items[: args.limit]

    refusal_items = [i for i in items if i.get("suite") == "adversarial_refusal"]
    routing_items = [i for i in items if i.get("suite") == "domain_routing"]

    # Refusal gate
    refusal_pass = sum(1 for it in refusal_items if evaluate_code_gate(it).passed)
    refusal_rate = refusal_pass / len(refusal_items) if refusal_items else 1.0

    # Domain routing gate
    routing_pass = sum(1 for it in routing_items if evaluate_code_gate(it).passed)
    routing_rate = routing_pass / len(routing_items) if routing_items else 1.0

    print(f"\n🔬 Layer 1 Code Gates")
    print(f"  Refusal Rate : {refusal_rate:.1%} ({refusal_pass}/{len(refusal_items)}) [threshold ≥ 90%]")
    print(f"  Routing Rate : {routing_rate:.1%} ({routing_pass}/{len(routing_items)}) [threshold ≥ 95%]")

    refusal_ok = refusal_rate >= 0.90
    routing_ok = routing_rate >= 0.95

    if refusal_ok and routing_ok:
        print("\n✅ All code gates PASSED")
        return 0
    else:
        failures = []
        if not refusal_ok:
            failures.append(f"Refusal rate {refusal_rate:.1%} < 90%")
        if not routing_ok:
            failures.append(f"Routing rate {routing_rate:.1%} < 95%")
        print(f"\n❌ Code gates FAILED: {'; '.join(failures)}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the evaluation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )

    if args.gates_only:
        return run_gates_only(args)

    from rag_core.evaluation.harness import EvaluationHarness
    from rag_core.evaluation.quality_gate import QualityGate, QualityGateConfig

    dataset_path = Path(args.dataset) if args.dataset else None

    if args.retrieval_only:
        print("🔍 Starting Decoupled Retrieval Evaluation (0 LLM Tokens, < 2s)...")
        gate = QualityGate(QualityGateConfig(retrieval_only=True, hit_at_3=0.0, mrr=0.0, context_recall=0.0))
        harness = EvaluationHarness(
            retriever=None,
            synthesizer=None,
            ragas_evaluator=None,
            quality_gate=gate,
            dataset_path=dataset_path,
            limit=args.limit,
            run_id=args.run_id,
            retrieval_only=True,
        )
    else:
        # Full RAGAS evaluation
        print("🚀 Starting RAGAS evaluation (Tier 3)...")
        print("⚠️  This requires a running LLM gateway at http://localhost:20128/v1")
        print("   Run `make eval` or configure --dataset / --limit as needed.\n")

        harness = EvaluationHarness(
            # retriever and synthesizer are None here; override for integration runs
            retriever=None,
            synthesizer=None,
            ragas_evaluator=None,
            quality_gate=QualityGate(),
            dataset_path=dataset_path,
            limit=args.limit,
            run_id=args.run_id,
            retrieval_only=False,
        )

    report = harness.run(suite_filter=args.suite, split_filter=args.split)
    print("\n" + report.summary_str())

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n📄 Report saved to {output_path}")

    manager = BaselineManager()

    if args.compare:
        manager.compare(report)

    if args.save:
        manager.save(report)

    return 0 if report.quality_gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())

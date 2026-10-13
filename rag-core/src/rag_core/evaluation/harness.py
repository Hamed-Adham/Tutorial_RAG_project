"""
Stage 10: RAGAS Evaluation Harness

Orchestrates the full evaluation pipeline against the Golden Benchmark:
1. Load golden test set from ``golden_set.jsonl``
2. For each item: retrieve context, synthesise answer
3. Compute Faithfulness, Answer Relevancy, Context Precision, Context Recall
   via ``LocalRagasEvaluator``
4. Aggregate metrics and return ``EvaluationReport``

Constitution v1.2.0:
- Evaluation is a first-class citizen, operational from Day 1.
- Metrics thresholds: Faithfulness >= 0.80, Relevancy >= 0.75,
  Precision >= 0.75, Recall >= 0.70.
- Results are persisted and compared against baseline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_core.evaluation.retrieval_scorer import RetrievalScorer
from rag_core.postprocessing.citations import parse_citations_from_text, evaluate_citation_provenance
from rag_core.evaluation.benchmark_schema import (
    DecoupledEvaluationReportModel,
    RetrievalMetricsModel,
    GenerationMetricsModel,
    CitationMetricsModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------


@dataclass
class ItemResult:
    """Evaluation result for a single golden benchmark item."""

    item_id: str
    question: str
    ground_truth: str
    answer: str
    suite: str

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    is_refusal: bool = False

    semantic_context_recall: float = 0.0
    recall_alignment: str = "N/A"

    latency_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all benchmark items."""

    run_id: str
    timestamp: str

    mean_faithfulness: float = 0.0
    mean_answer_relevancy: float = 0.0
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    mean_semantic_context_recall: float = 0.0

    mean_hit_at_1: float = 0.0
    mean_hit_at_3: float = 0.0
    mean_hit_at_5: float = 0.0
    mean_mrr: float = 0.0
    mean_citation_precision: float = 0.0
    mean_citation_recall: float = 0.0
    mean_adversarial_refusal_rate: float = 0.0

    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    mean_latency_seconds: float = 0.0

    quality_gate_passed: bool = False

    items: List[ItemResult] = field(default_factory=list)
    suite_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    alignment_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevancy": self.mean_answer_relevancy,
            "mean_context_precision": self.mean_context_precision,
            "mean_context_recall": self.mean_context_recall,
            "mean_hit_at_1": self.mean_hit_at_1,
            "mean_hit_at_3": self.mean_hit_at_3,
            "mean_hit_at_5": self.mean_hit_at_5,
            "mean_mrr": self.mean_mrr,
            "mean_citation_precision": self.mean_citation_precision,
            "mean_citation_recall": self.mean_citation_recall,
            "mean_adversarial_refusal_rate": self.mean_adversarial_refusal_rate,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "mean_latency_seconds": self.mean_latency_seconds,
            "quality_gate_passed": self.quality_gate_passed,
            "suite_breakdown": self.suite_breakdown,
        }

    def to_decoupled_report(self) -> DecoupledEvaluationReportModel:
        """Converts to validated DecoupledEvaluationReportModel matching contract."""
        return DecoupledEvaluationReportModel(
            run_id=self.run_id,
            timestamp=self.timestamp,
            total_queries=max(1, self.total_items),
            retrieval_metrics=RetrievalMetricsModel(
                hit_at_1=round(self.mean_hit_at_1, 4),
                hit_at_3=round(self.mean_hit_at_3, 4),
                hit_at_5=round(self.mean_hit_at_5, 4),
                mrr=round(self.mean_mrr, 4),
                context_recall=round(self.mean_context_recall, 4),
            ),
            generation_metrics=GenerationMetricsModel(
                faithfulness=round(self.mean_faithfulness, 4),
                answer_relevancy=round(self.mean_answer_relevancy, 4),
                adversarial_refusal_rate=round(self.mean_adversarial_refusal_rate, 4),
            ),
            citation_metrics=CitationMetricsModel(
                citation_precision=round(self.mean_citation_precision, 4),
                citation_recall=round(self.mean_citation_recall, 4),
            ),
            suite_breakdown=self.suite_breakdown,
        )

    def summary_str(self) -> str:
        """Return a formatted summary string for CLI output."""
        gate_icon = "✅" if self.quality_gate_passed else "❌"
        adv_items = [i for i in self.items if i.suite == "adversarial_refusal"]
        refusal_str = f"{self.mean_adversarial_refusal_rate:.3f}" if adv_items else "N/A (no adv items)"
        align_block = ""
        if self.alignment_summary:
            total_audited = sum(self.alignment_summary.values()) or 1
            agreed = self.alignment_summary.get("AGREED_HIT", 0) + self.alignment_summary.get("AGREED_MISS", 0)
            agree_pct = (agreed / total_audited) * 100.0
            align_block = (
                f"  [Scorer Alignment Audit: JevJudge vs RetrievalScorer]\n"
                f"    Semantic Rec : {self.mean_semantic_context_recall:.3f} (JevJudge Content Recall)\n"
                f"    URI Recall   : {self.mean_context_recall:.3f} (_matches_chunk URI Recall)\n"
                f"    Agreed Hits  : {self.alignment_summary.get('AGREED_HIT', 0)}\n"
                f"    Agreed Misses: {self.alignment_summary.get('AGREED_MISS', 0)}\n"
                f"    URI False Miss: {self.alignment_summary.get('URI_FALSE_MISS', 0)} (Jev found content, URI missed)\n"
                f"    URI False Hit : {self.alignment_summary.get('URI_FALSE_HIT', 0)} (URI matched, Jev found no content)\n"
                f"    Agreement    : {agree_pct:.1f}%\n"
            )
        return (
            f"Evaluation Report [{self.run_id}]\n"
            f"  Items          : {self.successful_items}/{self.total_items} successful\n"
            f"  [Retrieval Metrics]\n"
            f"    Hit@1        : {self.mean_hit_at_1:.3f}\n"
            f"    Hit@3        : {self.mean_hit_at_3:.3f}\n"
            f"    Hit@5        : {self.mean_hit_at_5:.3f}\n"
            f"    MRR          : {self.mean_mrr:.3f}\n"
            f"    Ctx Recall   : {self.mean_context_recall:.3f} (threshold ≥ 0.70)\n"
            f"{align_block}"
            f"  [Generation Metrics]\n"
            f"    Faithfulness : {self.mean_faithfulness:.3f} (threshold ≥ 0.80)\n"
            f"    Ans Relevancy: {self.mean_answer_relevancy:.3f} (threshold ≥ 0.75)\n"
            f"    Refusal Rate : {refusal_str}\n"
            f"  [Citation Metrics]\n"
            f"    Citation Prec: {self.mean_citation_precision:.3f}\n"
            f"    Citation Rec : {self.mean_citation_recall:.3f}\n"
            f"  Avg Latency    : {self.mean_latency_seconds:.2f}s\n"
            f"  Quality Gate   : {gate_icon}"
        )



# ---------------------------------------------------------------------------
# EvaluationHarness
# ---------------------------------------------------------------------------


class EvaluationHarness:
    """
    RAGAS Evaluation Harness.

    Connects the retrieval pipeline to the RAGAS metric adapters and
    produces a complete ``EvaluationReport``.

    Args:
        retriever:      A retriever with a ``retrieve(query) -> list`` method.
        synthesizer:    A synthesizer with a ``synthesize(query, chunks) -> SynthesisResult``.
        ragas_evaluator: A ``LocalRagasEvaluator`` instance.
        quality_gate:   A ``QualityGate`` instance to check thresholds.
        dataset_path:   Path to ``golden_set.jsonl``.
        limit:          Maximum number of items to evaluate (None = all).
        run_id:         Optional identifier for this evaluation run.

    Example::

        harness = EvaluationHarness(
            retriever=dense_retriever,
            synthesizer=synthesizer,
            ragas_evaluator=ragas_eval,
            quality_gate=QualityGate(),
        )
        report = harness.run()
        print(report.summary_str())
    """

    DEFAULT_DATASET_PATH = (
        Path(__file__).parent / "datasets" / "golden_set.jsonl"
    )

    def __init__(
        self,
        retriever: Optional[Any] = None,
        synthesizer: Optional[Any] = None,
        ragas_evaluator: Optional[Any] = None,
        quality_gate: Optional[Any] = None,
        dataset_path: Optional[Path] = None,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
        retrieval_only: bool = False,
        jev_judge: Optional[Any] = None,
        audit_jev: bool = False,
    ) -> None:
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._ragas_evaluator = ragas_evaluator
        self._quality_gate = quality_gate
        self._dataset_path = dataset_path or self.DEFAULT_DATASET_PATH
        self._limit = limit
        self._run_id = run_id or f"eval-{int(time.time())}"
        self._retrieval_only = retrieval_only
        self._retrieval_scorer = RetrievalScorer()
        self._jev_judge = jev_judge
        self._audit_jev = audit_jev
        if self._audit_jev and self._jev_judge is None:
            try:
                from rag_core.evaluation.jev_judge import JevJudge
                self._jev_judge = JevJudge()
            except Exception as e:
                logger.warning("Could not initialize JevJudge for audit: %s", e)

    def run(
        self,
        suite_filter: Optional[str] = None,
        split_filter: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Run the full evaluation pipeline.

        Args:
            suite_filter: If set, only evaluate items from this suite
                (e.g. ``"science"``, ``"admin_faq"``).
            split_filter: If set, only evaluate items from this split
                (e.g. ``"dev"``, ``"test"``).

        Returns:
            ``EvaluationReport`` with aggregated metrics.
        """
        items = self._load_dataset(suite_filter, split_filter)
        logger.info(
            "EvaluationHarness [%s]: running %d items (retrieval_only=%s)",
            self._run_id, len(items), self._retrieval_only
        )

        results: List[ItemResult] = []

        for item in items:
            result = self._evaluate_item(item)
            results.append(result)
            if result.error:
                logger.warning("Item %s failed: %s", item.get("id"), result.error)

        report = self._aggregate(results)
        if self._quality_gate:
            report.quality_gate_passed = self._quality_gate.check(report)
        else:
            report.quality_gate_passed = self._default_gate(report)

        logger.info("EvaluationHarness complete:\n%s", report.summary_str())
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(
        self,
        suite_filter: Optional[str],
        split_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load and optionally filter the golden benchmark dataset."""
        if not self._dataset_path.exists():
            logger.warning("Dataset not found at %s", self._dataset_path)
            return []

        items: List[Dict[str, Any]] = []
        with open(self._dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    meta = item.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}
                    item_suite = meta.get("suite") or item.get("suite")
                    if suite_filter and item_suite != suite_filter:
                        continue
                    item_split = meta.get("split") or item.get("split")
                    if split_filter and item_split != split_filter:
                        continue
                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line: %s", e)

        if self._limit:
            items = items[: self._limit]

        logger.info("Loaded %d items from %s", len(items), self._dataset_path)
        return items

    def _evaluate_item(self, item: Dict[str, Any]) -> ItemResult:
        """Run retrieval + synthesis + metrics for one benchmark item."""
        t_start = time.perf_counter()
        item_id = item.get("id") or item.get("item_id", "unknown")
        question = item.get("input") or item.get("question", "")
        ground_truth = item.get("expected_output") or item.get("ground_truth", "")

        meta = item.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        suite = meta.get("suite") or item.get("suite", "unknown")
        ground_truth_chunks = meta.get("ground_truth_chunks") or item.get("ground_truth_chunks", [])

        base = ItemResult(
            item_id=item_id,
            question=question,
            ground_truth=ground_truth,
            answer="",
            suite=suite,
        )

        try:
            # 1. Retrieve
            if self._retriever:
                chunks = self._retriever.retrieve(question)
            else:
                chunks = []

            # Extract chunk identifiers for deterministic retrieval evaluation
            retrieved_chunk_ids = []
            for c in chunks:
                if hasattr(c, "chunk_identifier"):
                    retrieved_chunk_ids.append(c.chunk_identifier)
                elif hasattr(c, "chunk_id") and c.chunk_id:
                    retrieved_chunk_ids.append(c.chunk_id)
                elif isinstance(c, dict):
                    retrieved_chunk_ids.append(c.get("chunk_id") or c.get("id") or "")
                elif hasattr(c, "doc_id") and hasattr(c, "page"):
                    retrieved_chunk_ids.append(f"{c.doc_id}#page={c.page}")
                else:
                    retrieved_chunk_ids.append(str(c))

            # Deterministic retrieval scoring (0 tokens)
            ret_eval = self._retrieval_scorer.score_query(
                gold_chunks=ground_truth_chunks,
                retrieved_chunks=retrieved_chunk_ids,
            )
            base.hit_at_1 = ret_eval.hit_at_1
            base.hit_at_3 = ret_eval.hit_at_3
            base.hit_at_5 = ret_eval.hit_at_5
            base.mrr = ret_eval.mrr
            base.context_recall = ret_eval.context_recall

            contexts = [
                (c.text if hasattr(c, "text") else c.get("text", ""))
                for c in chunks
            ]

            # Dual-Scorer Retrieval Audit: Compare mechanical RetrievalScorer vs semantic JevJudge
            if self._jev_judge and ground_truth and contexts:
                sem_verdict = self._jev_judge.evaluate_context_recall(
                    query=question,
                    ground_truth=ground_truth,
                    context="\n\n".join(contexts[:5])
                )
                base.semantic_context_recall = sem_verdict.score

                uri_hit = (base.hit_at_5 > 0.0)
                sem_hit = (base.semantic_context_recall >= 0.70)

                if uri_hit and sem_hit:
                    base.recall_alignment = "AGREED_HIT"
                elif not uri_hit and not sem_hit:
                    base.recall_alignment = "AGREED_MISS"
                elif sem_hit and not uri_hit:
                    base.recall_alignment = "URI_FALSE_MISS"
                else:
                    base.recall_alignment = "URI_FALSE_HIT"

            if self._retrieval_only:
                base.answer = "[retrieval only]"
                base.latency_seconds = time.perf_counter() - t_start
                return base

            # 2. Synthesize

            if self._synthesizer:
                synthesis = self._synthesizer.synthesize(question, chunks)
                answer = synthesis.answer
            else:
                answer = "[no synthesizer configured]"

            # 3. Generation metrics
            if self._ragas_evaluator and contexts and answer:
                metrics = self._ragas_evaluator.evaluate(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
                base.faithfulness = metrics.get("faithfulness", 0.0)
                base.answer_relevancy = metrics.get("answer_relevancy", 0.0)
                base.context_precision = metrics.get("context_precision", 0.0)
                if not ground_truth_chunks:
                    base.context_recall = metrics.get("context_recall", 0.0)
            elif answer and contexts:
                from rag_core.observability.trace_evaluators import (
                    compute_faithfulness,
                    compute_answer_relevancy,
                    compute_context_precision,
                    compute_context_recall,
                )
                is_ref = getattr(base, "is_refusal", False) or "اطلاعات کافی برای پاسخ" in answer
                base.faithfulness = compute_faithfulness(answer=answer, chunks=chunks, is_refusal=is_ref)
                base.answer_relevancy = compute_answer_relevancy(query=question, answer=answer, is_refusal=is_ref)
                base.context_precision = compute_context_precision(query=question, chunks=chunks)
                if not ground_truth_chunks:
                    base.context_recall = compute_context_recall(query=question, chunks=chunks)
            else:
                base.faithfulness = 0.0
                base.answer_relevancy = 0.0
                base.context_precision = 0.0

            # 4. Citation Provenance (US4)
            cited = parse_citations_from_text(answer)
            audit = evaluate_citation_provenance(cited_chunks=cited, gold_chunks=ground_truth_chunks)
            base.citation_precision = audit.precision
            base.citation_recall = audit.recall

            # 5. Adversarial Refusal Check
            if meta.get("expected_behavior") == "refusal" or suite == "adversarial_refusal":
                refusal_phrases = ["نمی‌توان", "اطلاعاتی یافت نشد", "خارج از دامنه", "پاسخگو نیست", "مربوط نمی‌شود"]
                base.is_refusal = any(p in answer for p in refusal_phrases)

            base.answer = answer
            base.latency_seconds = time.perf_counter() - t_start

        except Exception as exc:
            base.error = str(exc)
            base.latency_seconds = time.perf_counter() - t_start
            logger.error("Item %s evaluation error: %s", item_id, exc, exc_info=True)

        return base

    def _aggregate(self, results: List[ItemResult]) -> EvaluationReport:
        """Aggregate per-item results into an ``EvaluationReport``."""
        from datetime import datetime, timezone

        successful = [r for r in results if not r.error]
        n = len(successful) or 1  # avoid division by zero

        def mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        # Per-suite breakdown
        suites: Dict[str, List[ItemResult]] = {}
        for r in successful:
            suites.setdefault(r.suite, []).append(r)

        suite_breakdown = {}
        for suite, suite_items in suites.items():
            refusal_items = [i for i in suite_items if i.suite == "adversarial_refusal"]
            suite_breakdown[suite] = {
                "hit_at_1": mean([i.hit_at_1 for i in suite_items]),
                "hit_at_3": mean([i.hit_at_3 for i in suite_items]),
                "hit_at_5": mean([i.hit_at_5 for i in suite_items]),
                "mrr": mean([i.mrr for i in suite_items]),
                "faithfulness": mean([i.faithfulness for i in suite_items]),
                "answer_relevancy": mean([i.answer_relevancy for i in suite_items]),
                "context_precision": mean([i.context_precision for i in suite_items]),
                "context_recall": mean([i.context_recall for i in suite_items]),
                "citation_precision": mean([i.citation_precision for i in suite_items]),
                "citation_recall": mean([i.citation_recall for i in suite_items]),
                "count": len(suite_items),
            }

        adv_items = [r for r in successful if r.suite == "adversarial_refusal"]
        adv_refusal_rate = (sum(1.0 for r in adv_items if r.is_refusal) / len(adv_items)) if adv_items else 0.0

        alignment_counts: Dict[str, int] = {}
        for r in successful:
            if r.recall_alignment != "N/A":
                alignment_counts[r.recall_alignment] = alignment_counts.get(r.recall_alignment, 0) + 1

        sem_recalls = [r.semantic_context_recall for r in successful if r.recall_alignment != "N/A"]
        mean_sem_recall = mean(sem_recalls) if sem_recalls else 0.0

        return EvaluationReport(
            run_id=self._run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            mean_faithfulness=mean([r.faithfulness for r in successful]),
            mean_answer_relevancy=mean([r.answer_relevancy for r in successful]),
            mean_context_precision=mean([r.context_precision for r in successful]),
            mean_context_recall=mean([r.context_recall for r in successful]),
            mean_semantic_context_recall=mean_sem_recall,
            mean_hit_at_1=mean([r.hit_at_1 for r in successful]),
            mean_hit_at_3=mean([r.hit_at_3 for r in successful]),
            mean_hit_at_5=mean([r.hit_at_5 for r in successful]),
            mean_mrr=mean([r.mrr for r in successful]),
            mean_citation_precision=mean([r.citation_precision for r in successful]),
            mean_citation_recall=mean([r.citation_recall for r in successful]),
            mean_adversarial_refusal_rate=adv_refusal_rate,
            total_items=len(results),
            successful_items=len(successful),
            failed_items=len(results) - len(successful),
            mean_latency_seconds=mean([r.latency_seconds for r in results]),
            items=results,
            suite_breakdown=suite_breakdown,
            alignment_summary=alignment_counts,
        )

    @staticmethod
    def _default_gate(report: EvaluationReport) -> bool:
        """Apply constitutional quality thresholds without a QualityGate object."""
        return (
            report.mean_faithfulness >= 0.80
            and report.mean_answer_relevancy >= 0.75
            and report.mean_context_precision >= 0.75
            and report.mean_context_recall >= 0.70
        )


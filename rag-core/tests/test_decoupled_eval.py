#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract test for Decoupled Retrieval vs. Generation Evaluation (US3).
Tests deterministic scoring: Hit@1, Hit@3, Hit@5, MRR, Context Recall.
"""

import unittest
from rag_core.evaluation.retrieval_scorer import RetrievalScorer, RetrievalQueryEvaluation
from rag_core.evaluation.benchmark_schema import RetrievalMetricsModel, DecoupledEvaluationReportModel, GenerationMetricsModel, CitationMetricsModel


class TestDecoupledRetrievalScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = RetrievalScorer()

    def test_single_query_exact_first_hit(self):
        gold = ["docA#page=1"]
        retrieved = ["docA#page=1", "docB#page=2", "docC#page=3"]
        result = self.scorer.score_query(gold_chunks=gold, retrieved_chunks=retrieved)

        self.assertEqual(result.hit_at_1, 1.0)
        self.assertEqual(result.hit_at_3, 1.0)
        self.assertEqual(result.hit_at_5, 1.0)
        self.assertEqual(result.mrr, 1.0)
        self.assertEqual(result.context_recall, 1.0)

    def test_single_query_third_hit(self):
        gold = ["docC#page=3"]
        retrieved = ["docA#page=1", "docB#page=2", "docC#page=3", "docD#page=4"]
        result = self.scorer.score_query(gold_chunks=gold, retrieved_chunks=retrieved)

        self.assertEqual(result.hit_at_1, 0.0)
        self.assertEqual(result.hit_at_3, 1.0)
        self.assertEqual(result.hit_at_5, 1.0)
        self.assertAlmostEqual(result.mrr, 1.0 / 3.0, places=4)
        self.assertEqual(result.context_recall, 1.0)

    def test_single_query_miss(self):
        gold = ["docZ#page=99"]
        retrieved = ["docA#page=1", "docB#page=2", "docC#page=3"]
        result = self.scorer.score_query(gold_chunks=gold, retrieved_chunks=retrieved)

        self.assertEqual(result.hit_at_1, 0.0)
        self.assertEqual(result.hit_at_3, 0.0)
        self.assertEqual(result.hit_at_5, 0.0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.context_recall, 0.0)

    def test_partial_context_recall(self):
        gold = ["docA#page=1", "docB#page=2"]
        retrieved = ["docA#page=1", "docX#page=10"]
        result = self.scorer.score_query(gold_chunks=gold, retrieved_chunks=retrieved)

        self.assertEqual(result.hit_at_1, 1.0)
        self.assertEqual(result.mrr, 1.0)
        self.assertEqual(result.context_recall, 0.5)

    def test_aggregate_metrics(self):
        queries = [
            RetrievalQueryEvaluation(gold_chunks=["doc1"], retrieved_chunks=["doc1", "doc2"]),  # MRR: 1.0
            RetrievalQueryEvaluation(gold_chunks=["doc2"], retrieved_chunks=["docX", "doc2"]),  # MRR: 0.5
            RetrievalQueryEvaluation(gold_chunks=["doc3"], retrieved_chunks=["docA", "docB", "docC", "doc3"]), # MRR: 0.25 (rank 4)
            RetrievalQueryEvaluation(gold_chunks=["doc4"], retrieved_chunks=["none"]), # MRR: 0.0
        ]
        metrics = self.scorer.aggregate(queries)

        self.assertIsInstance(metrics, RetrievalMetricsModel)
        self.assertEqual(metrics.hit_at_1, 0.25)
        self.assertEqual(metrics.hit_at_3, 0.50)
        self.assertEqual(metrics.hit_at_5, 0.75)
        self.assertAlmostEqual(metrics.mrr, (1.0 + 0.5 + 0.25 + 0.0) / 4.0, places=4)

    def test_out_of_scope_query_exclusion(self):
        # Out-of-scope query has empty gold_chunks
        res = self.scorer.score_query(gold_chunks=[], retrieved_chunks=["docA#page=1"])
        self.assertTrue(res.is_out_of_scope)
        self.assertEqual(res.hit_at_1, 0.0)
        self.assertEqual(res.mrr, 0.0)

        # In-scope query: hit_1 = 1.0, mrr = 1.0
        in_scope = RetrievalQueryEvaluation(gold_chunks=["doc1"], retrieved_chunks=["doc1"])
        # Out-of-scope query
        out_scope = self.scorer.score_query(gold_chunks=[], retrieved_chunks=["random_chunk"])

        # Aggregate should calculate metrics strictly over in_scope queries (n=1)
        metrics = self.scorer.aggregate([in_scope, out_scope])
        self.assertEqual(metrics.hit_at_1, 1.0)
        self.assertEqual(metrics.mrr, 1.0)

    def test_decoupled_report_serialization(self):
        report = DecoupledEvaluationReportModel(
            run_id="run-test-001",
            timestamp="2026-10-08T17:00:00Z",
            total_queries=100,
            retrieval_metrics=RetrievalMetricsModel(
                hit_at_1=0.75,
                hit_at_3=0.88,
                hit_at_5=0.92,
                mrr=0.82,
                context_recall=0.85
            ),
            generation_metrics=GenerationMetricsModel(
                faithfulness=0.84,
                answer_relevancy=0.81,
                adversarial_refusal_rate=0.95
            ),
            citation_metrics=CitationMetricsModel(
                citation_precision=0.91,
                citation_recall=0.89
            )
        )
        data = report.model_dump()
        self.assertEqual(data["run_id"], "run-test-001")
        self.assertEqual(data["retrieval_metrics"]["hit_at_1"], 0.75)


class TestDecoupledDiagnosticMatrix(unittest.TestCase):
    """Unit tests for Feature 007 US4: 2x2 Decoupled Confusion Matrix."""

    def setUp(self):
        from rag_core.evaluation.decoupled_eval import DecoupledEvaluator
        self.evaluator = DecoupledEvaluator(hit_k=3, faithfulness_threshold=0.7)

    def _make_item(self, item_id: str, expected_behavior: str = "answer", gold_chunks=None):
        from rag_core.evaluation.benchmark_schema import BenchmarkItem, SuiteType, DifficultyLevel, SplitType
        return BenchmarkItem(
            item_id=item_id,
            suite=SuiteType.SCIENCE_CURRICULUM if expected_behavior != "refusal" else SuiteType.ADVERSARIAL_REFUSAL,
            split=SplitType.DEV,
            difficulty=DifficultyLevel.MEDIUM,
            query="آوندها در گیاهان چه نقشی دارند؟",
            ground_truth="انتقال آب و مواد غذایی" if expected_behavior != "refusal" else "اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد",
            expected_route="science_textbook" if expected_behavior != "refusal" else "refuse",
            expected_behavior=expected_behavior,
            gold_chunk_ids=gold_chunks or (["chunk_bio_01"] if expected_behavior != "refusal" else []),
            category="biology",
        )

    def test_quadrant_true_positive(self):
        """Retrieval Hit + Faithful Generation -> True Positive."""
        from rag_core.evaluation.decoupled_eval import DiagnosticQuadrant
        from rag_core.evaluation.benchmark_schema import SubQJudgeVerdict

        item = self._make_item("item-tp", "answer", ["chunk_bio_01"])
        retrieved = ["chunk_bio_01", "chunk_other_02"]
        answer = "آوندها وظیفه انتقال آب و مواد معدنی را در گیاه بر عهده دارند."
        verdict = SubQJudgeVerdict(
            metric_name="faithfulness",
            verdict=1,
            score=0.95,
            confidence=0.99,
            thought="Fully grounded in retrieved chunk.",
            latency_seconds=0.3,
        )

        res = self.evaluator.classify_item(item, retrieved, answer, verdict)
        self.assertEqual(res.quadrant, DiagnosticQuadrant.TRUE_POSITIVE)
        self.assertEqual(res.retrieval_hit, 1)
        self.assertEqual(res.synthesis_passed, 1)

    def test_quadrant_false_positive_hallucination(self):
        """Retrieval Hit + Unfaithful Generation -> False Positive."""
        from rag_core.evaluation.decoupled_eval import DiagnosticQuadrant
        from rag_core.evaluation.benchmark_schema import SubQJudgeVerdict

        item = self._make_item("item-fp", "answer", ["chunk_bio_01"])
        retrieved = ["chunk_bio_01"]
        answer = "آوندها سلول‌های عصبی هستند که تفکر گیاه را کنترل می‌کنند."
        verdict = SubQJudgeVerdict(
            metric_name="faithfulness",
            verdict=0,
            score=0.1,
            confidence=0.95,
            thought="Completely fabricated information.",
            latency_seconds=0.25,
        )

        res = self.evaluator.classify_item(item, retrieved, answer, verdict)
        self.assertEqual(res.quadrant, DiagnosticQuadrant.FALSE_POSITIVE)
        self.assertEqual(res.retrieval_hit, 1)
        self.assertEqual(res.synthesis_passed, 0)

    def test_quadrant_false_negative_missed_retrieval(self):
        """Retrieval Miss + Ungrounded synthesis attempt -> False Negative."""
        from rag_core.evaluation.decoupled_eval import DiagnosticQuadrant

        item = self._make_item("item-fn", "answer", ["chunk_bio_01"])
        retrieved = ["chunk_irrelevant_99"]
        answer = "آوندها احتمالا لوله‌های باریکی هستند."

        res = self.evaluator.classify_item(item, retrieved, answer)
        self.assertEqual(res.quadrant, DiagnosticQuadrant.FALSE_NEGATIVE)
        self.assertEqual(res.retrieval_hit, 0)
        self.assertEqual(res.synthesis_passed, 0)

    def test_quadrant_true_negative_adversarial_refusal(self):
        """Adversarial Query + Clean Refusal -> True Negative."""
        from rag_core.evaluation.decoupled_eval import DiagnosticQuadrant

        item = self._make_item("item-tn-adv", "refusal", [])
        retrieved = []
        answer = "اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد."

        res = self.evaluator.classify_item(item, retrieved, answer)
        self.assertEqual(res.quadrant, DiagnosticQuadrant.TRUE_NEGATIVE)
        self.assertEqual(res.is_refusal, True)
        self.assertEqual(res.synthesis_passed, 1)

    def test_quadrant_true_negative_retrieval_miss_safe_refusal(self):
        """In-domain Query + Retrieval Miss + Safe Refusal -> True Negative."""
        from rag_core.evaluation.decoupled_eval import DiagnosticQuadrant

        item = self._make_item("item-tn-miss", "answer", ["chunk_bio_01"])
        retrieved = ["chunk_unrelated"]
        answer = "اطلاعاتی در مورد این موضوع در اسناد یافت نشد."

        res = self.evaluator.classify_item(item, retrieved, answer)
        self.assertEqual(res.quadrant, DiagnosticQuadrant.TRUE_NEGATIVE)
        self.assertEqual(res.retrieval_hit, 0)
        self.assertEqual(res.is_refusal, True)

    def test_confusion_matrix_derived_metrics(self):
        """Verify Hallucination Catch Rate = TN / (TN + FP) and Retrieval Coverage."""
        from rag_core.evaluation.benchmark_schema import ConfusionMatrixResult

        # Case 1: Standard distribution
        # TP = 60, FP = 10, FN = 15, TN = 15
        # Total = 100
        # Hallucination Catch Rate = TN / (TN + FP) = 15 / (15 + 10) = 15 / 25 = 0.60
        # Retrieval Coverage = (TP + FP) / Total = 70 / 100 = 0.70
        matrix = ConfusionMatrixResult(
            true_positive=60,
            false_positive=10,
            false_negative=15,
            true_negative=15,
        )
        self.assertEqual(matrix.total, 100)
        self.assertAlmostEqual(matrix.hallucination_catch_rate, 0.60, places=4)
        self.assertAlmostEqual(matrix.retrieval_coverage, 0.70, places=4)

    def test_batch_evaluation(self):
        """Verify batch processing produces exact aggregate confusion matrix."""
        from rag_core.evaluation.decoupled_eval import DiagnosticQuadrant

        items = [
            self._make_item("1", "answer", ["c1"]),
            self._make_item("2", "answer", ["c2"]),
            self._make_item("3", "refusal", []),
            self._make_item("4", "answer", ["c4"]),
        ]
        retrieved_list = [
            ["c1"],              # Hit -> TP
            ["c2"],              # Hit but answer is ungrounded -> FP
            [],                  # Refusal clean -> TN
            ["irrelevant"],      # Miss + no refusal -> FN
        ]
        answers = [
            "پاسخ درست با انتقال آب و مواد غذایی",
            "پاسخ کاملا نامرتبط و اشتباه",
            "اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد",
            "پاسخ حدسی بدون استناد",
        ]

        matrix, details = self.evaluator.evaluate_batch(
            items=items,
            retrieved_chunk_ids_list=retrieved_list,
            generated_answers=answers,
        )

        self.assertEqual(matrix.true_positive, 1)
        self.assertEqual(matrix.false_positive, 1)
        self.assertEqual(matrix.true_negative, 1)
        self.assertEqual(matrix.false_negative, 1)
        self.assertEqual(matrix.total, 4)
        self.assertAlmostEqual(matrix.hallucination_catch_rate, 0.5, places=4)
        self.assertAlmostEqual(matrix.retrieval_coverage, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()


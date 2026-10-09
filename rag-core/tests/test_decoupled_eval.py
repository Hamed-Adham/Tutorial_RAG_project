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


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract validation unit tests for BenchmarkItem and EvaluationReport schemas.
Verifies compliance with specs/002-benchmark-dataset-scaling/contracts/.
"""

import unittest
from pydantic import ValidationError
from rag_core.evaluation.benchmark_schema import (
    BenchmarkItemModel,
    DecoupledEvaluationReportModel,
)


class TestBenchmarkSchemaContract(unittest.TestCase):
    def test_valid_benchmark_item(self):
        valid_item = {
            "id": "FAQ-COLLOQ-001",
            "input": "آیا شرکت فناور غیردانش‌بنیان می‌تونه از صندوق تسهیلات بگیره؟",
            "expected_output": "صندوق مستقیماً فقط به دانش‌بنیان‌ها تسهیلات می‌دهد، اما شرکت‌های غیردانش‌بنیان می‌توانند از مسیرهای جایگزین استفاده کنند.",
            "metadata": {
                "suite": "administrative_faq",
                "split": "dev",
                "category": "میز خدمت و پذیرش",
                "query_type": "colloquial_paraphrase",
                "ground_truth_doc_ids": ["faq_000"],
                "ground_truth_chunks": ["faq_qa_pairs.csv#row=0"],
                "expected_behavior": "answer",
                "expected_route": "administrative_faq",
                "requires_multi_hop": False
            }
        }
        item = BenchmarkItemModel.model_validate(valid_item)
        self.assertEqual(item.id, "FAQ-COLLOQ-001")
        self.assertEqual(item.metadata.suite, "administrative_faq")

    def test_invalid_benchmark_item_missing_required(self):
        invalid_item = {
            "id": "FAQ-BAD-001",
            "input": "سوال تستی"
        }
        with self.assertRaises(ValidationError):
            BenchmarkItemModel.model_validate(invalid_item)

    def test_invalid_empty_chunks(self):
        invalid_item = {
            "id": "FAQ-BAD-002",
            "input": "سوال معتبر درباره تسهیلات",
            "expected_output": "پاسخ معتبر برای سوال",
            "metadata": {
                "suite": "administrative_faq",
                "split": "dev",
                "category": "میز خدمت",
                "query_type": "factoid",
                "ground_truth_chunks": [],  # Min length 1 required
                "expected_behavior": "answer",
                "expected_route": "administrative_faq"
            }
        }
        with self.assertRaises(ValidationError):
            BenchmarkItemModel.model_validate(invalid_item)

    def test_valid_evaluation_report(self):
        valid_report = {
            "run_id": "eval_run_20261008_120000",
            "timestamp": "2026-10-08T12:00:00Z",
            "total_queries": 400,
            "retrieval_metrics": {
                "hit_at_1": 0.81,
                "hit_at_3": 0.90,
                "hit_at_5": 0.95,
                "mrr": 0.86,
                "context_recall": 0.92
            },
            "generation_metrics": {
                "faithfulness": 0.88,
                "answer_relevancy": 0.85,
                "adversarial_refusal_rate": 0.98
            },
            "citation_metrics": {
                "citation_precision": 0.91,
                "citation_recall": 0.89
            },
            "suite_breakdown": {
                "administrative_faq": {"hit_at_5": 0.96},
                "science_textbook": {"hit_at_5": 0.93}
            }
        }
        report = DecoupledEvaluationReportModel.model_validate(valid_report)
        self.assertEqual(report.total_queries, 400)
        self.assertEqual(report.retrieval_metrics.hit_at_5, 0.95)


if __name__ == "__main__":
    unittest.main()

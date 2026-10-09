#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and Contract Tests for FAQ Benchmark Generator and Catalog Mapping (US1).
"""

import unittest
from rag_core.evaluation.faq_catalog import FAQCatalog
from rag_core.evaluation.benchmark_schema import BenchmarkItemModel


class TestFAQBenchmark(unittest.TestCase):
    def setUp(self):
        self.catalog = FAQCatalog()

    def test_catalog_completeness(self):
        self.assertEqual(len(self.catalog), 134)
        for i in range(134):
            doc_id = f"faq_{i:03d}"
            entry = self.catalog.get(doc_id)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.doc_id, doc_id)
            self.assertEqual(entry.row_index, i)
            self.assertTrue(len(entry.question) > 5)
            self.assertTrue(len(entry.answer) > 0)

    def test_benchmark_item_schema_compliance(self):
        entry = self.catalog.get("faq_000")
        item_dict = {
            "id": "FAQ-COLLOQ-001",
            "input": "آیا شرکت فناور غیردانش‌بنیان می‌تونه از صندوق تسهیلات بگیره؟",
            "expected_output": entry.answer,
            "metadata": {
                "suite": "administrative_faq",
                "split": "dev",
                "category": entry.category,
                "query_type": "colloquial_paraphrase",
                "ground_truth_doc_ids": [entry.doc_id],
                "ground_truth_chunks": [entry.chunk_identifier],
                "expected_behavior": "answer",
                "expected_route": "administrative_faq",
                "requires_multi_hop": False
            }
        }
        item = BenchmarkItemModel.model_validate(item_dict)
        self.assertEqual(item.id, "FAQ-COLLOQ-001")
        self.assertEqual(item.metadata.ground_truth_chunks, ["faq_qa_pairs.csv#row=0"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and Contract Tests for 15-Chapter Science Benchmark Generator (US2).
"""

import unittest
from rag_core.evaluation.textbook_grounder import TextbookGrounder
from rag_core.evaluation.benchmark_schema import BenchmarkItemModel


class TestScienceBenchmark(unittest.TestCase):
    def setUp(self):
        self.grounder = TextbookGrounder()

    def test_curriculum_coverage(self):
        chapters = self.grounder.chapters()
        self.assertEqual(len(chapters), 15)
        self.assertEqual(chapters, list(range(1, 16)))
        self.assertEqual(self.grounder.total_sections(), 412)

    def test_science_benchmark_item_schema(self):
        sec = self.grounder.get_sections(1)[0]
        item_dict = {
            "id": "SCI-FACT-001",
            "input": "مراحل روش علمی در تفکر و تجربه چیست؟",
            "expected_output": "مشاهده، طرح پرسش، فرضیه‌سازی، آزمایش، ثبت داده‌ها و نتیجه‌گیری.",
            "metadata": {
                "suite": "science_textbook",
                "split": "dev",
                "category": "فصل ۱: تفکر و تجربه",
                "query_type": "factoid",
                "ground_truth_doc_ids": [sec.global_id],
                "ground_truth_chunks": [sec.chunk_identifier],
                "expected_behavior": "answer",
                "expected_route": "science_textbook",
                "requires_multi_hop": False
            }
        }
        item = BenchmarkItemModel.model_validate(item_dict)
        self.assertEqual(item.id, "SCI-FACT-001")
        self.assertTrue(item.metadata.ground_truth_chunks[0].endswith("#page=1"))


if __name__ == "__main__":
    unittest.main()

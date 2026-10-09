#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract test for Citation & Provenance Verification (US4).
Tests parsing citations from generated answers and evaluating Citation Precision and Recall.
"""

import unittest
from rag_core.postprocessing.citations import (
    parse_citations_from_text,
    evaluate_citation_provenance,
    CitationAuditResult,
)


class TestCitationProvenanceEval(unittest.TestCase):
    def test_parse_standard_citation_block(self):
        answer = (
            "انرژی پتانسیل گرانشی به ارتفاع و جرم وابسته است.\n\n"
            "**Sources:**\n"
            "[1] chapter_08.pdf – page 4\n"
            "[2] chapter_08.pdf – page 5\n"
        )
        cited = parse_citations_from_text(answer)
        self.assertEqual(len(cited), 2)
        self.assertIn("chapter_08.pdf#page=4", cited)
        self.assertIn("chapter_08.pdf#page=5", cited)

    def test_parse_uri_format_citation(self):
        answer = "پاسخ بر اساس (C706_olom_haftom-7-144-6-11.pdf#page=3) ارائه شده است."
        cited = parse_citations_from_text(answer)
        self.assertEqual(len(cited), 1)
        self.assertEqual(cited[0], "C706_olom_haftom-7-144-6-11.pdf#page=3")

    def test_parse_faq_catalog_citation(self):
        answer = "مطابق با آیین‌نامه [faq_042] وام قرض‌الحسنه پرداخت می‌شود."
        cited = parse_citations_from_text(answer)
        self.assertEqual(len(cited), 1)
        self.assertEqual(cited[0], "faq_042")

    def test_perfect_citation_provenance(self):
        gold = ["chapter_08.pdf#page=4", "chapter_08.pdf#page=5"]
        cited = ["chapter_08.pdf#page=4", "chapter_08.pdf#page=5"]
        audit = evaluate_citation_provenance(cited_chunks=cited, gold_chunks=gold)

        self.assertEqual(audit.precision, 1.0)
        self.assertEqual(audit.recall, 1.0)
        self.assertEqual(len(audit.hallucinated_citations), 0)

    def test_hallucinated_and_missing_citations(self):
        gold = ["chapter_08.pdf#page=4", "chapter_08.pdf#page=5"]
        # Model cited page 4 (valid) and phantom page 99 (invalid)
        cited = ["chapter_08.pdf#page=4", "phantom_doc.pdf#page=99"]
        audit = evaluate_citation_provenance(cited_chunks=cited, gold_chunks=gold)

        self.assertEqual(audit.precision, 0.5)  # 1 valid out of 2 cited
        self.assertEqual(audit.recall, 0.5)     # 1 covered out of 2 gold
        self.assertIn("phantom_doc.pdf#page=99", audit.hallucinated_citations)
        self.assertIn("chapter_08.pdf#page=5", audit.missing_citations)

    def test_no_citations_when_none_expected(self):
        audit = evaluate_citation_provenance(cited_chunks=[], gold_chunks=[])
        self.assertEqual(audit.precision, 1.0)
        self.assertEqual(audit.recall, 1.0)

    def test_no_citations_when_citations_expected(self):
        gold = ["docA#page=1"]
        audit = evaluate_citation_provenance(cited_chunks=[], gold_chunks=gold)
        self.assertEqual(audit.precision, 0.0)
        self.assertEqual(audit.recall, 0.0)


if __name__ == "__main__":
    unittest.main()

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
    def test_parse_markdown_link_citations(self):
        answer = (
            "انرژی پتانسیل گرانشی به ارتفاع و جرم وابسته است.\n\n"
            "**Sources:**\n"
            "[1] [chapter_08.pdf – page 4](chapter_08.pdf#page=4)\n"
            "[2] [chapter_08.pdf – page 5](file:///data/docs/chapter_08.pdf#page=5)\n"
            "همچنین در [آیین‌نامه دانش‌بنیان](faq_qa_pairs.csv#row=42) و [سوال ۴۲](faq_042) تشریح شده است."
        )
        cited = parse_citations_from_text(answer)
        self.assertEqual(len(cited), 4)
        self.assertIn("chapter_08.pdf#page=4", cited)
        self.assertIn("chapter_08.pdf#page=5", cited)
        self.assertIn("faq_qa_pairs.csv#row=42", cited)
        self.assertIn("faq_042", cited)

    def test_negative_rejection_of_plain_brackets_and_unlinked_text(self):
        # Clarification Q-3: Plain bracketed references or bare URIs must be rejected as unverified
        plain_bracket_block = (
            "انرژی پتانسیل گرانشی به ارتفاع و جرم وابسته است.\n\n"
            "**Sources:**\n"
            "[1] chapter_08.pdf – page 4\n"
            "[2] chapter_08.pdf – page 5\n"
        )
        self.assertEqual(parse_citations_from_text(plain_bracket_block), [])

        plain_catalog_token = "مطابق با آیین‌نامه [faq_042] وام قرض‌الحسنه پرداخت می‌شود."
        self.assertEqual(parse_citations_from_text(plain_catalog_token), [])

        bare_uri = "پاسخ بر اساس C706_olom_haftom-7-144-6-11.pdf#page=3 ارائه شده است."
        self.assertEqual(parse_citations_from_text(bare_uri), [])

        generic_bracket = "بر اساس [منبع: کتاب علوم تجربی هفتم] فرضیه مطرح شد."
        self.assertEqual(parse_citations_from_text(generic_bracket), [])

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

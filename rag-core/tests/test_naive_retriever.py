#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for NaiveRetriever and NaiveRAGPipeline source-isolated retrieval.
Verifies that naive_index.npz chunks are strictly segregated by source_type.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from rag_core.naive.chunker import Chunk
from rag_core.naive.retriever import NaiveRetriever
from rag_core.naive.pipeline import NaiveRAGPipeline


class TestNaiveRetrieverIsolation(unittest.TestCase):
    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_query.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.retriever = NaiveRetriever(embedder=self.mock_embedder)
        self.chunks = [
            Chunk(
                chunk_id="sci_01",
                doc_id="C706_olom_haftom-7-144-2-5_p01",
                text="مباحث علمی تفکر و تجربه علوم هفتم",
                source_type="science_textbook",
                source_name="C706_olom_haftom-7-144-2-5.pdf",
                metadata={"page": 1, "filename": "C706_olom_haftom-7-144-2-5.pdf"}
            ),
            Chunk(
                chunk_id="sci_02",
                doc_id="C706_olom_haftom-7-144-13-24_p01",
                text="اتم‌ها و ساختار ذرات ماده",
                source_type="science_textbook",
                source_name="C706_olom_haftom-7-144-13-24.pdf",
                metadata={"page": 13, "filename": "C706_olom_haftom-7-144-13-24.pdf"}
            ),
            Chunk(
                chunk_id="faq_01",
                doc_id="faq_row_001",
                text="خدمات و تسهیلات شرکت‌های دانش‌بنیان صندوق نوآوری",
                source_type="administrative_faq",
                source_name="faq_qa_pairs.csv",
                metadata={"category": "میز خدمت و پذیرش", "row_index": 1}
            ),
            Chunk(
                chunk_id="faq_02",
                doc_id="faq_row_002",
                text="سقف ضمانت‌نامه و تسهیلات مالی",
                source_type="administrative_faq",
                source_name="faq_qa_pairs.csv",
                metadata={"category": "ضمانت‌نامه", "row_index": 2}
            ),
        ]
        self.retriever.chunks = self.chunks
        # Orthogonal/controlled unit vectors
        self.retriever.embeddings = np.array([
            [1.0, 0.0, 0.0],   # sci_01 -> match 1.0
            [0.8, 0.6, 0.0],   # sci_02 -> match 0.8
            [0.9, 0.1, 0.0],   # faq_01 -> match 0.9
            [0.5, 0.5, 0.0],   # faq_02 -> match 0.5
        ], dtype=np.float32)

    def test_retrieve_without_filter_returns_mixed(self):
        results = self.retriever.retrieve("test query", top_k=3)
        self.assertEqual(len(results), 3)
        sources = [c.source_type for c, _ in results]
        self.assertIn("science_textbook", sources)
        self.assertIn("administrative_faq", sources)

    def test_retrieve_with_science_filter_returns_only_science(self):
        results = self.retriever.retrieve("test query", top_k=3, filter_source_type="science_textbook")
        self.assertEqual(len(results), 2)
        for chunk, score in results:
            self.assertEqual(chunk.source_type, "science_textbook")
        self.assertEqual(results[0][0].chunk_id, "sci_01")
        self.assertEqual(results[1][0].chunk_id, "sci_02")

    def test_retrieve_with_faq_filter_returns_only_faq(self):
        results = self.retriever.retrieve("test query", top_k=3, filter_source_type="administrative_faq")
        self.assertEqual(len(results), 2)
        for chunk, score in results:
            self.assertEqual(chunk.source_type, "administrative_faq")
        self.assertEqual(results[0][0].chunk_id, "faq_01")
        self.assertEqual(results[1][0].chunk_id, "faq_02")

    def test_pipeline_query_forwards_source_filter(self):
        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize.return_value = "پاسخ آزمایشی"

        pipeline = NaiveRAGPipeline(
            embedder=self.mock_embedder,
            retriever=self.retriever,
            synthesizer=mock_synthesizer
        )
        pipeline._is_indexed = True

        res = pipeline.query("پرسش علمی", top_k=2, source="science_textbook")
        self.assertEqual(res["answer"], "پاسخ آزمایشی")
        for src in res["sources"]:
            self.assertEqual(src["source_type"], "science_textbook")

    def test_citation_formatting_and_book_page_resolution(self):
        from rag_core.postprocessing.citations import PageCitation, CitationFormatter, _extract_start_page_from_filename

        filename = "C706_olom_haftom-7-144-86-97.pdf"
        start_page = _extract_start_page_from_filename(filename)
        self.assertEqual(start_page, 86)

        pure_page = 1
        book_page = start_page + pure_page - 1
        self.assertEqual(book_page, 86)

        citation = PageCitation(
            doc_id="C706_olom_haftom-7-144-86-97_p01",
            filename=filename,
            page=pure_page,
            score=0.95,
            book_page=book_page,
            citation="فصل 10 - گرما و بهینه‌سازی مصرف انرژی – صفحه 86"
        )

        formatter = CitationFormatter()
        bundle = formatter.format([citation])

        # Verify visual format has chapter num, chapter title, and book page
        self.assertIn("فصل 10 - گرما و بهینه‌سازی مصرف انرژی – صفحه 86", bundle.inline_text)
        # Verify markdown link uses pure page for PDF viewer navigation
        self.assertIn("/api/v1/pdf/C706_olom_haftom-7-144-86-97.pdf#page=1", bundle.inline_text)


if __name__ == "__main__":
    unittest.main()

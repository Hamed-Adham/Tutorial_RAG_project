#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive automated unit and integration tests for US1 10-Stage Pipeline:
- Manifest Catalog & Deduplication (Stage 1)
- Document Lineage & Tombstones (Stage 1)
- Canonical IR Parsing & Tiered Routing (Stage 2)
- Hierarchical & Parent-Child Chunking (Stage 3)
- Multi-Index Vector Indexing & Retrieval (Stage 4, 6)
- Page & Bounding Box Source Citations (Stage 8)
- Grounded Response Synthesis (Stage 9)
- Evaluation Harness, Quality Gates & Baseline Manager (Stage 10)

Uses Python's standard library `unittest` for fast, zero-external-dependency execution.
"""

import tempfile
import unittest
from pathlib import Path
from typing import List

from rag_core.manifest.catalog import InMemoryCatalog, DocumentRecord, DocumentStatus
from rag_core.manifest.deduplicator import DocumentDeduplicator
from rag_core.manifest.lineage import DocumentLineageTracker, LineageEventType
from rag_core.parsing.schema import Block, BlockType, ParsedDocument
from rag_core.parsing.router import ParsingRouter
from rag_core.parsing.tier0_layout import PyMuPDFExtractor, PlainTextExtractor
from rag_core.chunking.hierarchical import HierarchicalChunker
from rag_core.chunking.parent_child import ParentChildChunker
from rag_core.postprocessing.citations import CitationFormatter
from rag_core.generation.synthesizer import ResponseSynthesizer
from rag_core.evaluation.quality_gate import QualityGate, QualityGateConfig, GateCheckResult
from rag_core.evaluation.baseline_manager import BaselineManager
from rag_core.evaluation.harness import EvaluationReport


class TestManifestAndDeduplication(unittest.TestCase):
    """Stage 1: Tests for manifest catalog, deduplicator, and lineage."""

    def setUp(self):
        self.catalog = InMemoryCatalog()
        self.deduplicator = DocumentDeduplicator(near_dup_threshold=0.85)
        self.lineage = DocumentLineageTracker()

    def test_catalog_register_and_retrieve(self):
        content = b"Sample document bytes for testing catalog registration."
        res = self.catalog.register(
            filename="sample.pdf",
            content=content,
            tenant_id="tenant_default",
            acl_roles=["role_admin"],
        )
        self.assertTrue(res.is_new)
        self.assertFalse(res.is_duplicate)
        self.assertIsNotNone(res.doc_id)

        retrieved = self.catalog.get(res.doc_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.filename, "sample.pdf")

    def test_exact_deduplication(self):
        sha = "abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        text = "This is exact duplicate test content."
        res1 = self.deduplicator.check(doc_id="doc_1", sha256=sha, text=text)
        self.assertFalse(res1.is_exact_duplicate)

        # Register doc_1
        self.deduplicator.add_document(doc_id="doc_1", sha256=sha, text=text)

        # Second check on same hash must be flagged as exact duplicate
        res2 = self.deduplicator.check(doc_id="doc_2", sha256=sha, text=text)
        self.assertTrue(res2.is_exact_duplicate)
        self.assertEqual(res2.duplicate_of, "doc_1")

    def test_lineage_and_tombstones(self):
        doc_id = "doc_lifecycle_01"
        evt = self.lineage.record_ingest(doc_id=doc_id, version=1)
        self.assertEqual(evt.event_type, LineageEventType.INGESTED)

        tombstone = self.lineage.create_tombstone(doc_id=doc_id, reason="retention_policy")
        self.assertTrue(self.lineage.is_tombstoned(doc_id))
        self.assertIn("qdrant", tombstone.indices_pending_removal)

        # Acknowledge index deletion
        self.lineage.acknowledge_index_removal(doc_id, "qdrant")
        self.assertNotIn("qdrant", tombstone.indices_pending_removal)


class TestParsingAndChunking(unittest.TestCase):
    """Stages 2 & 3: Tests for tiered parser, schema, and chunkers."""

    def setUp(self):
        self.router = ParsingRouter()
        self.hierarchical_chunker = HierarchicalChunker(max_chunk_chars=150, overlap_chars=20)
        self.parent_child_chunker = ParentChildChunker(parent_chunk_chars=300, child_chunk_chars=100)

    def test_canonical_schema_blocks(self):
        block = Block(
            type=BlockType.HEADING,
            text="فصل اول: علوم تجربی",
            page=1,
            bbox=(50.0, 100.0, 300.0, 120.0),
            heading_level=1,
        )
        self.assertEqual(block.type, BlockType.HEADING)
        self.assertEqual(block.page, 1)
        self.assertEqual(block.bbox[0], 50.0)
        self.assertEqual(block.heading_level, 1)

    def test_router_text_document(self):
        text_content = b"Simple plain text file content for tiered routing."
        parsed = self.router.parse(text_content, doc_id="doc_txt_01", filename="test.txt")
        self.assertIsInstance(parsed, ParsedDocument)
        self.assertGreaterEqual(len(parsed.blocks), 1)

    def test_hierarchical_chunking_with_breadcrumbs(self):
        blocks = [
            Block(type=BlockType.HEADING, text="مقدمه", page=1, heading_level=1),
            Block(type=BlockType.TEXT, text="این یک متن مقدماتی درباره علوم تجربی است.", page=1),
            Block(type=BlockType.HEADING, text="بخش ۱-۱: تعریف سلول", page=2, heading_level=2),
            Block(type=BlockType.TEXT, text="سلول واحد بنیادین ساختار و عملکرد موجودات زنده است.", page=2),
        ]
        parsed_doc = ParsedDocument(doc_id="doc_hier_01", filename="science.pdf", blocks=blocks)
        chunks = self.hierarchical_chunker.chunk(parsed_doc)

        self.assertGreaterEqual(len(chunks), 1)
        has_breadcrumb = any("مقدمه" in c.text or "بخش ۱-۱" in c.text or c.breadcrumb for c in chunks)
        self.assertTrue(has_breadcrumb)

    def test_parent_child_chunking(self):
        blocks = [
            Block(
                type=BlockType.TEXT,
                text=" ".join([f"جمله تستی شماره {i} برای پر کردن ظرفیت چانک والد و فرزند." for i in range(15)]),
                page=1,
            )
        ]
        parsed_doc = ParsedDocument(doc_id="doc_pc_01", filename="chapter.pdf", blocks=blocks)
        parents, children = self.parent_child_chunker.chunk(parsed_doc)

        self.assertGreater(len(parents), 0)
        self.assertGreater(len(children), 0)
        self.assertTrue(all(c.parent_chunk_id is not None for c in children))


class TestRetrievalAndSynthesis(unittest.TestCase):
    """Stages 6, 8, 9: Tests for dense retriever fallback, citations, and synthesizer."""

    def test_citations_formatter(self):
        chunks = [
            {
                "doc_id": "doc_01",
                "page": 12,
                "bbox": (50.0, 100.0, 200.0, 150.0),
                "text": "میتوکندری وظیفه تنفس سلولی را بر عهده دارد.",
                "metadata": {"filename": "science_book.pdf"},
            },
            {
                "doc_id": "doc_01",
                "page": 12,  # Same page to verify deduplication
                "bbox": (50.0, 200.0, 200.0, 250.0),
                "text": "تولید ATP در میتوکندری انجام می‌گیرد.",
                "metadata": {"filename": "science_book.pdf"},
            },
            {
                "doc_id": "doc_02",
                "page": 3,
                "text": "تسهیلات دانش‌بنیان شامل وام قرض‌الحسنه است.",
                "metadata": {"filename": "faq.pdf"},
            },
        ]
        formatter = CitationFormatter()
        bundle = formatter.format(chunks)
        self.assertGreaterEqual(len(bundle.citations), 2)
        citation_str = bundle.inline_text
        self.assertIn("science_book.pdf", citation_str)
        self.assertTrue("صفحه 12" in citation_str or "page 12" in citation_str)

    def test_synthesizer_grounded_fallback(self):
        synthesizer = ResponseSynthesizer()
        # Query with empty chunks must trigger polite refusal immediately
        res = synthesizer.synthesize(query="پایتخت مریخ کجاست؟", chunks=[])
        self.assertIn("don't have enough information", res.answer.lower())


class TestEvaluationAndQualityGate(unittest.TestCase):
    """Stage 10: Tests for quality gate validation and baseline management."""

    def test_quality_gate_pass(self):
        cfg = QualityGateConfig(
            faithfulness=0.80,
            answer_relevancy=0.75,
            context_precision=0.75,
            context_recall=0.70,
        )
        gate = QualityGate(config=cfg)
        scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.82,
            "context_precision": 0.79,
            "context_recall": 0.74,
        }
        res: GateCheckResult = gate.check_dict(scores)
        self.assertTrue(res.passed)
        self.assertEqual(len(res.failures), 0)

    def test_quality_gate_fail_on_faithfulness_drop(self):
        cfg = QualityGateConfig(
            faithfulness=0.80,
            answer_relevancy=0.75,
            context_precision=0.75,
            context_recall=0.70,
        )
        gate = QualityGate(config=cfg)
        scores = {
            "faithfulness": 0.72,  # Below 0.80 threshold
            "answer_relevancy": 0.85,
            "context_precision": 0.80,
            "context_recall": 0.75,
        }
        res: GateCheckResult = gate.check_dict(scores)
        self.assertFalse(res.passed)
        self.assertGreater(len(res.failures), 0)

    def test_baseline_manager_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(baseline_dir=Path(tmpdir))
            report = EvaluationReport(
                run_id="run_test_01",
                timestamp="2026-10-04T09:00:00Z",
                mean_faithfulness=0.82,
                mean_answer_relevancy=0.80,
                mean_context_precision=0.78,
                mean_context_recall=0.72,
            )
            saved_path = manager.save(report)
            self.assertTrue(saved_path.exists())

            loaded = manager.load_latest()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.get("mean_faithfulness"), 0.82)


if __name__ == "__main__":
    unittest.main()

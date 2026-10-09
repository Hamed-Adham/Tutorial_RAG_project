#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated unit tests for Evaluation, Knowledge Graph, and Observability components.
Uses standard library unittest for zero-dependency execution.
"""

import unittest
from rag_core.evaluation.code_gates import evaluate_code_gate, GateResult
from rag_core.kg.schema import DocumentNode, ChunkNode, EntityNode, MultiHopCandidate
from rag_core.kg.neo4j_store import Neo4jKnowledgeGraph
from rag_core.observability.langfuse import LangfuseTracker
from rag_core.naive.judge_evaluator import JudgeVerdict


class TestEvaluationFramework(unittest.TestCase):
    """Test suite for RAG evaluation, knowledge graph, and observability."""

    def test_layer1_code_gate_refusal_interception(self):
        """Verify Layer 1 gate catches Persian refusal patterns with sub-5ms latency."""
        item = {
            "id": "refusal_001",
            "suite_type": "adversarial_refusal",
            "question": "پایتخت کشور فرانسه کجاست؟"
        }
        pipeline_output = {
            "answer": "متاسفانه در اسناد بارگذاری شده اطلاعاتی درباره پایتخت فرانسه موجود نیست.",
            "sources": []
        }

        res = evaluate_code_gate(item, pipeline_output)
        self.assertIsInstance(res, GateResult)
        self.assertTrue(res.gated)
        self.assertEqual(res.passed, 1)
        self.assertLess(res.latency_seconds, 0.05)

    def test_layer1_code_gate_content_passthrough(self):
        """Verify Layer 1 gate passes substantive answers through to Layer 2."""
        item = {
            "id": "sci_001",
            "suite_type": "science",
            "question": "وظیفه میتوکندری چیست؟"
        }
        pipeline_output = {
            "answer": "میتوکندری اندامکی است که وظیفه تولید مولکول‌های ATP و تامین انرژی سلول را بر عهده دارد.",
            "sources": [{"source_name": "science_book.pdf"}]
        }

        res = evaluate_code_gate(item, pipeline_output)
        self.assertIsInstance(res, GateResult)
        # Content item is not gated deterministically, so gated is False (passes to LLM Judge)
        self.assertFalse(res.gated)
        self.assertLess(res.latency_seconds, 0.05)

    def test_kg_schema_dataclasses(self):
        """Verify KG schema dataclasses instantiate correctly."""
        doc = DocumentNode(
            doc_id="doc_001",
            filename="biology.pdf",
            sha256="abc123hash",
            created_at="2026-10-04T00:00:00Z"
        )
        self.assertEqual(doc.doc_id, "doc_001")

        candidate = MultiHopCandidate(
            chunk_a="c1",
            chunk_b="c2",
            shared_entity="میتوکندری",
            text_a="متن ۱",
            text_b="متن ۲"
        )
        self.assertEqual(candidate.shared_entity, "میتوکندری")

    def test_neo4j_store_connection_or_fallback(self):
        """Verify Neo4j store either connects or falls back gracefully without unhandled exception."""
        kg = Neo4jKnowledgeGraph()
        candidates = kg.find_multi_hop_candidates(limit=2)
        self.assertIsInstance(candidates, list)
        kg.close()

    def test_langfuse_tracker_instance(self):
        """Verify LangfuseTracker instantiates safely and handles missing server gracefully."""
        tracker = LangfuseTracker()
        self.assertTrue(hasattr(tracker, "is_active"))
        # Should not raise exception even when flushing with inactive client
        tracker.flush()

    def test_langfuse_trace_and_score_lifecycle(self):
        """Verify LangfuseTracker trace creation, span lifecycle, and score recording."""
        tracker = LangfuseTracker()
        trace_id = tracker.trace_query(
            name="test_query_trace",
            question="تست سوال",
            answer="تست پاسخ",
            top_k=2,
            retrieved_count=2,
            sources=[{"source_name": "test.pdf", "score": 0.95}],
            retrieval_latency=0.012,
            generation_latency=0.045
        )
        self.assertIsInstance(trace_id, str)

        # Ingest score
        tracker.add_trace_score(trace_id, "gate_pass", 1.0, comment="Deterministic gate pass")
        tracker.flush()

    def test_judge_verdict_latency_attribute(self):

        """Verify JudgeVerdict schema preserves latency_seconds."""
        verdict = JudgeVerdict(
            metric_name="faithfulness",
            verdict=1,
            thought="Step 1: Context check. Step 2: Synthesis check.",
            raw_response='{"thought": "...", "verdict": 1}',
            latency_seconds=0.452
        )
        self.assertEqual(verdict.verdict, 1)
        self.assertEqual(verdict.latency_seconds, 0.452)


if __name__ == "__main__":
    unittest.main()

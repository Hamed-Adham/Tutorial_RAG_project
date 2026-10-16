#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated unit tests for the Advanced Enterprise Tracing Structure in Langfuse.
Validates the complete 5-stage observation hierarchy and 4 evaluation metrics.
"""

import unittest
from unittest.mock import MagicMock

from rag_core.indexing.science_schema import (
    RetrievalCandidate,
    RerankedCandidate,
    RewriteDecision,
    ParentSection,
    ExpandedContext,
)
from rag_core.observability.langfuse import LangfuseTracker
from rag_core.observability.trace_evaluators import (
    compute_faithfulness,
    compute_context_precision,
    compute_context_recall,
    compute_answer_relevancy,
    evaluate_trace_quality_async,
)


class TestAdvancedTracingStructure(unittest.TestCase):
    """Test suite for the 5-stage hierarchical tracing structure and quality evaluations."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.tracker = LangfuseTracker()
        self.tracker.client = self.mock_client

        # Mock start_observation chaining
        self.mock_root = MagicMock()
        self.mock_root.trace_id = "trace_root_123"
        self.mock_root.id = "obs_root_123"

        self.mock_span = MagicMock()
        self.mock_client.start_observation.return_value = self.mock_root
        self.mock_root.start_observation.return_value = self.mock_span
        self.mock_span.start_observation.return_value = MagicMock()

    def test_trace_advanced_pipeline_creates_all_five_stages(self):
        """Verify that trace_advanced_pipeline constructs all 5 stages with proper types and child spans."""
        # 1. Stage 1 inputs
        rewrite_decision = RewriteDecision(
            needs_rewrite=True,
            confidence=0.92,
            rewritten_query="وظیفه میتوکندری در سلول چیست؟",
            sub_queries=["ساختار میتوکندری", "وظیفه میتوکندری در تولید انرژی"],
            reasoning="Query resolved pronoun 'آن' to mitochondria",
        )

        # 2. Stage 2 inputs
        bm25_cands = [
            RetrievalCandidate(
                chunk_id="ch_01",
                section_id="sec_01",
                chapter=11,
                page=92,
                text="میتوکندری وظیفه تنفس سلولی را بر عهده دارد.",
                breadcrumbs="فصل ۱۱ > سلول",
                bm25_score=14.2,
                bm25_rank=1,
            )
        ]
        dense_cands = [
            RetrievalCandidate(
                chunk_id="ch_01",
                section_id="sec_01",
                chapter=11,
                page=92,
                text="میتوکندری اندامک تولید انرژی است.",
                breadcrumbs="فصل ۱۱ > سلول",
                dense_score=0.88,
                dense_rank=1,
            )
        ]
        fused_cands = [
            RetrievalCandidate(
                chunk_id="ch_01",
                section_id="sec_01",
                chapter=11,
                page=92,
                text="میتوکندری وظیفه تنفس سلولی را بر عهده دارد.",
                breadcrumbs="فصل ۱۱ > سلول",
                dense_score=0.88,
                dense_rank=1,
                bm25_score=14.2,
                bm25_rank=1,
                rrf_score=0.032,
            )
        ]

        # 3. Stage 3 inputs
        reranked = [
            RerankedCandidate(
                candidate=fused_cands[0],
                cross_encoder_score=3.85,
                final_rank=1,
            )
        ]

        # 4. Stage 4 inputs
        parent_sec = ParentSection(
            section_id="sec_01",
            chapter=11,
            title="سلول و سازمان‌بندی آن",
            page_start=90,
            page_end=95,
            full_text="میتوکندری اندامکی است با غشای دو لایه که در تولید مولکول‌های پرانرژی نقش دارد.",
            child_chunk_ids=["ch_01"],
            token_count=320,
        )
        expanded_context = ExpandedContext(
            parent_sections=[parent_sec],
            total_tokens=320,
            source_chunk_ids=["ch_01"],
            reordered_texts=[parent_sec.full_text],
        )

        # 5. Stage 5 inputs
        synth_mock = MagicMock()
        synth_mock.model = "open-code"
        synth_mock.prompt_tokens = 540
        synth_mock.completion_tokens = 85
        synth_mock.total_tokens = 625
        synth_mock.raw_completion = "میتوکندری مرکز تنفس سلولی و تولید ATP است."
        synth_mock.full_prompt = "System: Grounded instructions\n\nUser: وظیفه میتوکندری چیست؟"

        trace_id = self.tracker.trace_advanced_pipeline(
            name="advanced_science_rag",
            question="وظیفه آن در سلول چیست؟",
            answer="میتوکندری مرکز تنفس سلولی و تولید ATP است.\n\n[کتاب علوم](science.pdf#page=92)",
            is_refusal=False,
            rewrite_decision=rewrite_decision,
            turn_pair_id="turn_abc123",
            history=[{"question": "سلول چیست؟", "answer": "واحد ساختار جانداران"}],
            subq_tokens=150,
            jev_tokens=40,
            effective_query=rewrite_decision.rewritten_query,
            rewrite_latency_ms=12.5,
            dense_results=dense_cands,
            bm25_results=bm25_cands,
            fused_candidates=fused_cands,
            retrieval_latency_ms=45.2,
            input_candidates=fused_cands,
            reranked_candidates=reranked,
            circuit_breaker_status="active",
            rerank_latency_ms=160.0,
            expanded_context=expanded_context,
            reordered_sections=[parent_sec],
            expansion_latency_ms=8.0,
            synthesis_result=synth_mock,
            synthesis_latency_ms=820.0,
        )

        self.assertEqual(trace_id, "trace_root_123")

        # Verify Root Trace observation
        self.mock_client.start_observation.assert_called_once()
        root_call_kwargs = self.mock_client.start_observation.call_args.kwargs
        self.assertEqual(root_call_kwargs["name"], "advanced_science_rag")
        self.assertEqual(root_call_kwargs["as_type"], "chain")
        self.assertIn("cost_breakdown", root_call_kwargs["metadata"])

        # Verify that root called start_observation for all top-level child stages
        stage_names = [call.kwargs.get("name") for call in self.mock_root.start_observation.call_args_list]
        self.assertIn("query_gate", stage_names)
        self.assertIn("query_transformation", stage_names)
        self.assertIn("conversational_memory", stage_names)
        self.assertIn("retrieval_chain", stage_names)
        self.assertIn("response_synthesis", stage_names)

        # Verify observation types
        stage_types = {call.kwargs.get("name"): call.kwargs.get("as_type") for call in self.mock_root.start_observation.call_args_list}
        self.assertEqual(stage_types["query_gate"], "chain")
        self.assertEqual(stage_types["query_transformation"], "chain")
        self.assertEqual(stage_types["conversational_memory"], "chain")
        self.assertEqual(stage_types["retrieval_chain"], "chain")
        self.assertEqual(stage_types["response_synthesis"], "generation")

    def test_evaluation_metrics_computation(self):
        """Verify the 4 evaluation metrics: faithfulness, context_precision, context_recall, answer_relevancy."""
        query = "وظیفه میتوکندری در سلول چیست؟"
        answer = "میتوکندری وظیفه تنفس سلولی و تولید انرژی را در سلول بر عهده دارد."
        chunks = [
            {"text": "میتوکندری اندامک تنفس سلولی و تولید مولکول های انرژی است.", "page": 92},
            {"text": "دستگاه گلژی در بسته‌بندی پروتئین‌ها نقش دارد.", "page": 93},
        ]

        # 1. Faithfulness
        faith = compute_faithfulness(answer=answer, chunks=chunks, is_refusal=False)
        self.assertGreaterEqual(faith, 0.5)
        self.assertLessEqual(faith, 1.0)

        # 2. Refusal faithfulness is 1.0
        refusal_faith = compute_faithfulness(answer="اطلاعاتی یافت نشد", chunks=[], is_refusal=True)
        self.assertEqual(refusal_faith, 1.0)

        # 3. Context Precision (relevant chunk is at rank 1, so precision should be high)
        precision = compute_context_precision(query=query, chunks=chunks)
        self.assertGreaterEqual(precision, 0.5)

        # 4. Context Recall (mitochondria and cell terms are in context)
        recall = compute_context_recall(query=query, chunks=chunks)
        self.assertGreaterEqual(recall, 0.5)

        # 5. Answer Relevancy (answer directly addresses query terms)
        relevancy = compute_answer_relevancy(query=query, answer=answer, is_refusal=False)
        self.assertGreaterEqual(relevancy, 0.5)

    def test_offline_tracker_graceful_fallback(self):
        """Verify that offline LangfuseTracker returns offline_trace without raising errors."""
        offline_tracker = LangfuseTracker()
        offline_tracker.client = None

        trace_id = offline_tracker.trace_advanced_pipeline(
            name="advanced_pipeline",
            question="تست",
            answer="پاسخ",
        )
        self.assertEqual(trace_id, "offline_trace")

        naive_trace_id = offline_tracker.trace_naive_pipeline(
            name="naive_pipeline",
            question="تست",
            answer="پاسخ",
        )
        self.assertEqual(naive_trace_id, "offline_trace")

    def test_trace_naive_pipeline_creates_two_stage_hierarchy(self):
        """Verify that trace_naive_pipeline constructs retrieval and response_synthesis stages."""
        sources = [
            {
                "chunk_id": "ch_01",
                "source_type": "science_textbook",
                "source_name": "science.pdf",
                "score": 0.85,
                "metadata": {"chapter": 7, "page": 86, "book_page": 86}
            }
        ]

        trace_id = self.tracker.trace_naive_pipeline(
            name="naive_pipeline",
            question="چگالی چیست؟",
            answer="چگالی جرم واحد حجم ماده است.",
            session_id="sess_123",
            turn_pair_id="turn_456",
            source="science_textbook",
            top_k=3,
            retriever_type="qdrant",
            retrieved_chunks=sources,
            chapters=[7],
            retrieval_latency_ms=18.5,
            synthesis_latency_ms=450.0,
            prompt_tokens=150,
            completion_tokens=40,
            tags=["mode:stream", "webui"],
            latencies={"retrieval_ms": 18.5, "synthesis_ms": 450.0, "total_ms": 468.5}
        )

        self.assertEqual(trace_id, "trace_root_123")

        # Verify Root Trace observation
        self.mock_client.start_observation.assert_called_once()
        root_call_kwargs = self.mock_client.start_observation.call_args.kwargs
        self.assertEqual(root_call_kwargs["name"], "naive_pipeline")
        self.assertEqual(root_call_kwargs["as_type"], "chain")
        self.assertIn("cost_breakdown", root_call_kwargs["metadata"])
        self.assertEqual(root_call_kwargs["metadata"]["retriever_type"], "qdrant")

        # Verify child observations: retrieval and response_synthesis
        child_names = [call.kwargs.get("name") for call in self.mock_root.start_observation.call_args_list]
        self.assertIn("retrieval", child_names)
        self.assertIn("response_synthesis", child_names)

        child_types = {call.kwargs.get("name"): call.kwargs.get("as_type") for call in self.mock_root.start_observation.call_args_list}
        self.assertEqual(child_types["retrieval"], "retriever")
        self.assertEqual(child_types["response_synthesis"], "generation")

    def test_trace_advanced_pipeline_chronological_ordering(self):
        """Verify that all pipeline stages receive strictly increasing synthetic timestamps."""
        created_spans = []

        def mock_start(**kwargs):
            obs = MagicMock()
            obs.name = kwargs.get("name")
            obs.kwargs = kwargs
            obs._otel_span = MagicMock()
            obs.end = MagicMock(side_effect=lambda end_time=None: setattr(obs, "recorded_end_time", end_time))
            obs.start_observation = mock_start
            created_spans.append(obs)
            return obs

        self.mock_client.start_observation = mock_start

        from rag_core.indexing.science_schema import QueryGateDecision, CurriculumIntentTier, ContextNeedTier, SelectedHistoryItem, HistoryRelevanceTier
        rw = QueryGateDecision(
            raw_query="یکایش چیه؟",
            effective_query="یکایش چیه؟",
            intent_tier=CurriculumIntentTier.RELEVANT,
            context_need_tier=ContextNeedTier.NEEDS_HISTORY_ONLY,
            selected_history_items=[
                SelectedHistoryItem(
                    turn_id="t1",
                    turn_index=0,
                    is_recent_5=True,
                    content_type="summary",
                    question="چگالی چیست؟",
                    answer="جرم بر حجم",
                    summary="خلاصه چگالی",
                    relevance_tier=HistoryRelevanceTier.NEEDED_RELEVANT
                )
            ]
        )

        self.tracker.trace_advanced_pipeline(
            name="advanced_pipeline",
            question="یکایش چیه؟",
            answer="کیلوگرم بر متر مکعب",
            rewrite_decision=rw,
            history=[{"role": "user", "content": "چگالی چیست؟"}],
            full_prompt="System: Grounded\nUser: <context>...</context>\nQuestion: یکایش چیه؟",
            latencies={"gate1_ms": 50.0, "distill_ms": 20.0, "subq_model2_ms": 30.0, "synthesis_ms": 200.0}
        )

        stage_obs = {s.name: s for s in created_spans}
        self.assertIn("query_gate", stage_obs)
        self.assertIn("query_transformation", stage_obs)
        self.assertIn("conversational_memory", stage_obs)
        self.assertIn("response_synthesis", stage_obs)

        gate_start = stage_obs["query_gate"]._otel_span._start_time
        trans_start = stage_obs["query_transformation"]._otel_span._start_time
        mem_start = stage_obs["conversational_memory"]._otel_span._start_time
        synth_start = stage_obs["response_synthesis"]._otel_span._start_time

        self.assertLess(gate_start, trans_start)
        self.assertLess(trans_start, mem_start)
        self.assertLess(mem_start, synth_start)

        # Check that response_synthesis received the full prompt containing context
        synth_input = stage_obs["response_synthesis"].kwargs.get("input")
        self.assertIn("<context>", synth_input)
        self.assertIn("یکایش چیه؟", synth_input)


if __name__ == "__main__":
    unittest.main()

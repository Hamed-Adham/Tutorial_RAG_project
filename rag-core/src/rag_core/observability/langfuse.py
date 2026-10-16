#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langfuse OpenTelemetry Tracing Module.
Provides centralized trace, span, and score management across all RAG pipeline stages,
adhering to OpenInference standards, Langfuse SDK v4, and Constitution v1.4.0.
"""

import time
from typing import Dict, Any, List, Optional
from rag_core.config import settings

try:
    from langfuse import Langfuse, propagate_attributes
except ImportError:
    Langfuse = None
    propagate_attributes = None


class DummySpan:
    """Safe fallback span object when Langfuse is offline or disabled."""
    def __init__(self, *args, **kwargs):
        self.id = "offline_span"
        self.trace_id = "offline_trace"

    def span(self, *args, **kwargs):
        return DummySpan()

    def start_observation(self, *args, **kwargs):
        return DummySpan()

    def update(self, *args, **kwargs):
        return self

    def generation(self, *args, **kwargs):
        return DummySpan()

    def event(self, *args, **kwargs):
        return DummySpan()

    def score(self, *args, **kwargs):
        return DummySpan()

    def score_trace(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass


class DummyTrace(DummySpan):
    """Safe fallback trace object."""
    id: str = "offline_trace"
    trace_id: str = "offline_trace"


class LangfuseTracker:
    """
    Centralized Langfuse telemetry manager adhering to OpenInference standards,
    Langfuse SDK v4, and Constitution v1.4.0 with graceful offline fallback.
    """

    def __init__(self, release: str = "v1.4.0"):
        self.release = release
        self.client: Optional[Any] = None
        if settings.langfuse_enabled and Langfuse is not None:
            try:
                self.client = Langfuse(
                    host=settings.langfuse_host,
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key
                )
            except Exception as e:
                # Log once, fall back gracefully
                print(f"Notice: Langfuse client initialization deferred/offline: {e}")

    @property
    def is_active(self) -> bool:
        """Return True if Langfuse client is active and enabled."""
        return self.client is not None

    def get_langchain_handler(
        self,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Return native Langfuse CallbackHandler for LangChain / LangGraph execution.
        """
        if not self.client:
            return None
        try:
            from langfuse.langchain import CallbackHandler
            handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
            )
            return handler
        except Exception:
            return None

    def sync_dataset(
        self,
        dataset_name: str,
        items: List[Dict[str, Any]],
        description: str = "Unified RAG Golden Benchmark Dataset"
    ) -> int:
        """
        Upload or synchronize benchmark items to a persistent Langfuse Dataset.
        Returns count of synced items.
        """
        if not self.client:
            return 0
        try:
            try:
                self.client.create_dataset(name=dataset_name, description=description)
            except Exception:
                pass

            count = 0
            for it in items:
                input_payload = {
                    "query": it.get("query"),
                    "suite": it.get("suite"),
                    "difficulty": it.get("difficulty")
                }
                expected_output = {
                    "ground_truth": it.get("ground_truth"),
                    "expected_refusal": it.get("expected_refusal", False),
                    "expected_route": it.get("expected_route")
                }
                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=input_payload,
                    expected_output=expected_output,
                    metadata={"source_document": it.get("source_document")}
                )
                count += 1
            return count
        except Exception as e:
            print(f"Notice: Failed to sync dataset to Langfuse: {e}")
            return 0

    def create_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
    ) -> Any:
        """Create and return a root trace object."""
        if not self.client:
            return DummyTrace()
        try:
            combined_tags = tags or []
            combined_metadata = metadata or {}
            if propagate_attributes:
                with propagate_attributes(
                    tags=combined_tags,
                    session_id=session_id,
                    user_id=user_id,
                    trace_name=name,
                    version=self.release,
                    metadata=combined_metadata,
                ):
                    return self.client.start_observation(
                        name=name,
                        as_type="chain",
                        input=input_data,
                        metadata=combined_metadata,
                        version=self.release,
                    )
            return self.client.start_observation(
                name=name,
                as_type="chain",
                input=input_data,
                metadata=combined_metadata,
                version=self.release,
            )
        except Exception:
            return DummyTrace()

    def trace_query(
        self,
        name: str,
        question: str,
        answer: str,
        top_k: int,
        retrieved_count: int,
        sources: List[Dict[str, Any]],
        retrieval_latency: float,
        generation_latency: float,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_pair_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Create a structured parent trace with nested retrieval (as_type='retriever')
        and synthesis (as_type='generation') observations adhering to Langfuse v4 standards.
        Returns the trace ID for downstream score ingestion.
        """
        if not self.client:
            return "offline_trace"

        try:
            combined_tags = ["pipeline:rag"]
            if tags:
                combined_tags.extend(tags)
            if turn_pair_id:
                combined_tags.append(f"turn:{turn_pair_id}")
            combined_metadata = metadata or {}
            if turn_pair_id:
                combined_metadata["turn_pair_id"] = turn_pair_id

            def _build_observations():
                root = self.client.start_observation(
                    name=name,
                    as_type="chain",
                    input={"question": question, "top_k": top_k},
                    output={"answer": answer},
                    metadata=combined_metadata,
                    version=self.release,
                )

                # Retrieval span (Observation type: RETRIEVER)
                retrieval_span = root.start_observation(
                    name="retrieval",
                    as_type="retriever",
                    input={"question": question, "top_k": top_k},
                    output={"retrieved_count": retrieved_count, "sources": sources},
                    metadata={
                        "latency_seconds": round(retrieval_latency, 3),
                        "vector_db": "qdrant",
                    },
                )
                retrieval_span.end()

                # Generation span (Observation type: GENERATION)
                gen_span = root.start_observation(
                    name="synthesis",
                    as_type="generation",
                    input=question,
                    output=answer,
                    model=model_name or settings.llm_model,
                    usage_details=usage_details,
                    cost_details=cost_details,
                    metadata={"latency_seconds": round(generation_latency, 3)},
                )
                gen_span.end()

                root.end()
                return getattr(root, "trace_id", "offline_trace")

            if propagate_attributes:
                with propagate_attributes(
                    tags=combined_tags,
                    session_id=session_id,
                    user_id=user_id,
                    trace_name=name,
                    version=self.release,
                    metadata=combined_metadata,
                ):
                    return _build_observations()
            else:
                return _build_observations()

        except Exception as e:
            return "offline_trace"

    def trace_naive_pipeline(
        self,
        name: str = "naive_pipeline",
        question: str = "",
        answer: str = "",
        session_id: Optional[str] = None,
        turn_pair_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source: str = "science_textbook",
        top_k: int = 3,
        retriever_type: str = "qdrant",
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        chapters: Optional[List[int]] = None,
        categories: Optional[List[str]] = None,
        retrieval_latency_ms: float = 0.0,
        synthesis_latency_ms: float = 0.0,
        model_name: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        latencies: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Create a 2-stage hierarchical trace structure for the baseline pipeline:
        Root Trace: naive_pipeline (CHAIN)
        ├── 1. retrieval (RETRIEVER)
        └── 2. response_synthesis (GENERATION)

        Adheres to OpenInference standards and allows direct comparison with advanced_pipeline.
        """
        if not self.client:
            return "offline_trace"

        try:
            combined_tags = ["pipeline:naive", f"source:{source}"]
            if retriever_type:
                combined_tags.append(f"retriever:{retriever_type}")
            if tags:
                for t in tags:
                    if t not in combined_tags:
                        combined_tags.append(t)
            if turn_pair_id:
                combined_tags.append(f"turn:{turn_pair_id}")

            combined_metadata = metadata or {}
            combined_metadata["pipeline_type"] = "naive"
            combined_metadata["retriever_type"] = retriever_type
            combined_metadata["source"] = source
            if turn_pair_id:
                combined_metadata["turn_pair_id"] = turn_pair_id
            if latencies:
                combined_metadata["latency_profile"] = latencies

            # Multi-model token cost calculations
            llm_in_rate = getattr(settings, "cost_llm_input_per_million", 0.150)
            llm_out_rate = getattr(settings, "cost_llm_output_per_million", 0.600)
            p_tok = prompt_tokens or max(1, int(len(question.split()) * 1.5 + 400))
            c_tok = completion_tokens or max(1, int(len(answer.split()) * 1.3))
            cost_synth = ((p_tok * llm_in_rate) + (c_tok * llm_out_rate)) / 1_000_000.0
            projected_cost_1k = cost_synth * 1000.0

            combined_metadata["cost_breakdown"] = {
                "synthesis_cost_usd": round(cost_synth, 6),
                "total_query_cost_usd": round(cost_synth, 6),
                "projected_cost_per_1k_usd": round(projected_cost_1k, 3),
            }

            def _build_naive_observations():
                def _stamp_naive(obs, s_ns: int, e_ns: int):
                    if obs is None:
                        return
                    if hasattr(obs, "_otel_span") and obs._otel_span is not None:
                        try:
                            obs._otel_span._start_time = int(s_ns)
                        except Exception:
                            pass
                    try:
                        obs.end(end_time=int(e_ns))
                    except Exception:
                        try:
                            obs.end()
                        except Exception:
                            pass

                now_ns = time.time_ns()
                l_ret = max(10.0, float(retrieval_latency_ms or (latencies.get("retrieval_ms", 0.0) if latencies else 0.0) or 30.0))
                l_synth = max(20.0, float(synthesis_latency_ms or (latencies.get("synthesis_ms", 0.0) if latencies else 0.0) or 200.0))
                naive_tot_dur_ns = int((l_ret + l_synth + 20.0) * 1_000_000)
                cur_ns = now_ns - naive_tot_dur_ns
                root_s_ns = cur_ns

                # Root Trace: naive_pipeline (Type: CHAIN)
                root = self.client.start_observation(
                    name=name,
                    as_type="chain",
                    input={
                        "question": question,
                        "session_id": session_id,
                        "turn_pair_id": turn_pair_id,
                        "top_k": top_k,
                        "source": source,
                        "chapters": chapters,
                        "categories": categories,
                    },
                    output={"answer": answer},
                    metadata=combined_metadata,
                    version=self.release,
                )

                # -------------------------------------------------------------
                # 1. Stage: retrieval (Type: RETRIEVER)
                # -------------------------------------------------------------
                chunks = retrieved_chunks or []
                ret_s_ns = cur_ns
                ret_e_ns = ret_s_ns + int(l_ret * 1_000_000)
                retrieval_obs = root.start_observation(
                    name="retrieval",
                    as_type="retriever",
                    input={
                        "question": question,
                        "top_k": top_k,
                        "filter_source": source,
                        "filter_chapters": chapters,
                        "filter_categories": categories,
                    },
                    output={
                        "retrieved_count": len(chunks),
                        "chunks": chunks,
                    },
                    metadata={
                        "latency_ms": round(retrieval_latency_ms or (latencies.get("retrieval_ms", 0.0) if latencies else 0.0), 2),
                        "retriever_type": retriever_type,
                        "vector_db": "qdrant" if retriever_type == "qdrant" else "numpy_in_memory",
                        "collection_name": "naive_baseline_chunks" if retriever_type == "qdrant" else "in_memory_cache",
                    },
                    version=self.release,
                )
                _stamp_naive(retrieval_obs, ret_s_ns, ret_e_ns)
                cur_ns = ret_e_ns + int(1_000_000)

                # -------------------------------------------------------------
                # 2. Stage: response_synthesis (Type: GENERATION)
                # -------------------------------------------------------------
                syn_s_ns = cur_ns
                syn_e_ns = syn_s_ns + int(l_synth * 1_000_000)
                synthesis_obs = root.start_observation(
                    name="response_synthesis",
                    as_type="generation",
                    input=question,
                    output=answer,
                    model=model_name or settings.llm_model,
                    usage_details={
                        "input": p_tok,
                        "output": c_tok,
                        "total": p_tok + c_tok,
                    },
                    cost_details={
                        "total": round(cost_synth, 6),
                    },
                    metadata={
                        "latency_ms": round(synthesis_latency_ms or (latencies.get("synthesis_ms", 0.0) if latencies else 0.0), 2),
                    },
                    version=self.release,
                )
                _stamp_naive(synthesis_obs, syn_s_ns, syn_e_ns)
                cur_ns = syn_e_ns + int(1_000_000)

                _stamp_naive(root, root_s_ns, cur_ns)
                return getattr(root, "trace_id", "offline_trace")

            if propagate_attributes:
                with propagate_attributes(
                    tags=combined_tags,
                    session_id=session_id,
                    user_id=user_id,
                    trace_name=name,
                    version=self.release,
                    metadata=combined_metadata,
                ):
                    return _build_naive_observations()
            else:
                return _build_naive_observations()

        except Exception:
            return "offline_trace"

    def trace_advanced_pipeline(
        self,
        name: str = "advanced_pipeline",
        question: str = "",
        answer: str = "",
        is_refusal: bool = False,
        session_id: Optional[str] = None,
        turn_pair_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Gate 1 & Transformation
        rewrite_decision: Optional[Any] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        effective_query: str = "",
        gate_latency_ms: float = 0.0,
        rewrite_latency_ms: float = 0.0,
        jev_tokens: int = 0,
        transform_tokens: int = 0,
        # Conversational Memory & Decision Model 2
        subq_evaluations: Optional[Dict[str, Any]] = None,
        subq_tokens: int = 0,
        subq_latency_ms: float = 0.0,
        missing_summaries_backfilled: int = 0,
        # Retrieval Chain
        dense_results: Optional[List[Any]] = None,
        bm25_results: Optional[List[Any]] = None,
        fused_candidates: Optional[List[Any]] = None,
        retrieval_latency_ms: float = 0.0,
        # Neural Reranking
        input_candidates: Optional[List[Any]] = None,
        reranked_candidates: Optional[List[Any]] = None,
        circuit_breaker_status: str = "active",
        rerank_latency_ms: float = 0.0,
        # Parent Expansion
        expanded_context: Optional[Any] = None,
        reordered_sections: Optional[List[Any]] = None,
        expansion_latency_ms: float = 0.0,
        context_reordering_enabled: Optional[bool] = None,
        # Response Synthesis
        synthesis_result: Optional[Any] = None,
        full_prompt: str = "",
        raw_completion: str = "",
        synthesis_latency_ms: float = 0.0,
        ttft_ms: Optional[float] = None,
        # Latency breakdown
        latencies: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Create the hierarchical trace structure adhering to the enterprise contract:
        Root Trace: advanced_science_rag (CHAIN)
        ├── 1. query_gate (CHAIN)
        │     └── jev_decision_model_1 (GENERATION)
        ├── 2. query_transformation (CHAIN) [if applicable]
        ├── 3. conversational_memory (CHAIN) [if applicable]
        │     ├── 3.1 summary_backfill (GENERATION) [if applicable]
        │     ├── 3.2 decision_model_2 (GENERATION) [SubQ]
        │     └── 3.3 fidelity_selection (CHAIN)
        ├── 4. retrieval_chain (CHAIN) [if retrieval active]
        │     ├── 4.1 bm25_search (RETRIEVER)
        │     ├── 4.2 dense_qdrant_search (RETRIEVER)
        │     ├── 4.3 rrf_fusion (CHAIN)
        │     ├── 4.4 neural_reranking (CHAIN)
        │     └── 4.5 parent_expansion (CHAIN)
        └── 5. response_synthesis (GENERATION)
        """
        if not self.client:
            return "offline_trace"

        try:
            combined_tags = ["pipeline:advanced", "domain:science_textbook"]
            if tags:
                combined_tags.extend(tags)
            if turn_pair_id:
                combined_tags.append(f"turn:{turn_pair_id}")

            rw = rewrite_decision
            if isinstance(rw, dict) and rw:
                try:
                    from rag_core.indexing.science_schema import QueryGateDecision
                    rw = QueryGateDecision.model_validate(rw)
                except Exception:
                    pass

            is_compound = bool(getattr(rw, "is_compound", False) or (rw.get("is_compound", False) if isinstance(rw, dict) else False))
            if is_compound:
                combined_tags.append("query:compound")

            combined_metadata = metadata or {}
            if latencies:
                combined_metadata["latency_profile"] = latencies
            if turn_pair_id:
                combined_metadata["turn_pair_id"] = turn_pair_id

            # Multi-model token cost calculations
            jev_rate = getattr(settings, "cost_jev_input_per_million", 0.042)
            subq_rate = getattr(settings, "cost_subq_input_per_million", 0.050)
            llm_in_rate = getattr(settings, "cost_llm_input_per_million", 0.150)
            llm_out_rate = getattr(settings, "cost_llm_output_per_million", 0.600)

            actual_jev_tokens = jev_tokens or getattr(rw, "jev_tokens", 0)
            actual_subq_tokens = subq_tokens or getattr(rw, "subq_tokens", 0)
            actual_transform_tokens = transform_tokens or getattr(rw, "transform_tokens", 0)
            actual_backfilled = missing_summaries_backfilled or getattr(rw, "missing_summaries_backfilled", 0)

            cost_jev = (actual_jev_tokens * jev_rate) / 1_000_000.0
            cost_transform = (actual_transform_tokens * llm_in_rate) / 1_000_000.0
            cost_subq = (actual_subq_tokens * subq_rate) / 1_000_000.0

            p_tok = getattr(synthesis_result, "prompt_tokens", 0)
            c_tok = getattr(synthesis_result, "completion_tokens", 0)
            t_tok = getattr(synthesis_result, "total_tokens", p_tok + c_tok)
            cost_synth = ((p_tok * llm_in_rate) + (c_tok * llm_out_rate)) / 1_000_000.0

            total_query_cost = cost_jev + cost_transform + cost_subq + cost_synth
            projected_cost_1k = total_query_cost * 1000.0

            combined_metadata["cost_breakdown"] = {
                "jev_gate_cost_usd": round(cost_jev, 6),
                "transform_cost_usd": round(cost_transform, 6),
                "subq_memory_cost_usd": round(cost_subq, 6),
                "synthesis_cost_usd": round(cost_synth, 6),
                "total_query_cost_usd": round(total_query_cost, 6),
                "projected_cost_per_1k_usd": round(projected_cost_1k, 3),
            }

            def _build_advanced_observations():
                def _stamp(obs, s_ns: int, e_ns: int):
                    if obs is None:
                        return
                    if hasattr(obs, "_otel_span") and obs._otel_span is not None:
                        try:
                            obs._otel_span._start_time = int(s_ns)
                        except Exception:
                            pass
                    try:
                        obs.end(end_time=int(e_ns))
                    except Exception:
                        try:
                            obs.end()
                        except Exception:
                            pass

                now_ns = time.time_ns()
                l_gate = max(10.0, float(gate_latency_ms or (latencies.get("gate1_ms", 0.0) if latencies else 0.0) or (latencies.get("rewrite_ms", 0.0) if latencies else 0.0) or 50.0))
                l_trans = max(5.0, float(rewrite_latency_ms or (latencies.get("distill_ms", 0.0) if latencies else 0.0) or 20.0))
                l_subq = max(10.0, float(subq_latency_ms or (latencies.get("subq_model2_ms", 0.0) if latencies else 0.0) or (latencies.get("history_eval_ms", 0.0) if latencies else 0.0) or 40.0))
                l_ret = max(10.0, float(retrieval_latency_ms or (latencies.get("retrieval_ms", 0.0) if latencies else 0.0) or 50.0))
                l_rerank = max(10.0, float(rerank_latency_ms or (latencies.get("rerank_ms", 0.0) if latencies else 0.0) or 60.0))
                l_exp = max(5.0, float(expansion_latency_ms or (latencies.get("expansion_ms", 0.0) if latencies else 0.0) or 15.0))
                l_synth = max(20.0, float(synthesis_latency_ms or (latencies.get("synthesis_ms", 0.0) if latencies else 0.0) or 300.0))

                # Root Trace: advanced_science_rag (Type: CHAIN)
                root = self.client.start_observation(
                    name=name,
                    as_type="chain",
                    input={"question": question, "session_id": session_id, "turn_pair_id": turn_pair_id},
                    output={"answer": answer, "is_refusal": is_refusal},
                    metadata=combined_metadata,
                    version=self.release,
                )

                # Precompute total duration and back-calculate anchor start time
                tot_ms = l_gate + l_trans + l_subq + (l_ret + l_rerank + l_exp) + l_synth + 50.0
                cursor_ns = now_ns - int(tot_ms * 1_000_000)
                root_start_ns = cursor_ns

                # -------------------------------------------------------------
                # 1. Stage: query_gate (Type: CHAIN)
                # -------------------------------------------------------------
                conf = getattr(rw, "confidence", 1.0) if rw else 1.0
                sub_queries = getattr(rw, "sub_queries", []) if rw else []
                reasoning = getattr(rw, "reasoning", "") if rw else ""
                bypass_retrieval = getattr(rw, "bypass_retrieval", False) if rw else False

                intent_tier = getattr(rw, "intent_tier", None)
                if hasattr(intent_tier, "value"):
                    intent_tier = intent_tier.value
                context_need_tier = getattr(rw, "context_need_tier", None)
                if hasattr(context_need_tier, "value"):
                    context_need_tier = context_need_tier.value

                gate_start_ns = cursor_ns
                gate_end_ns = gate_start_ns + int(l_gate * 1_000_000)

                gate_obs = root.start_observation(
                    name="query_gate",
                    as_type="chain",
                    input={"query": question, "history_len": len(history or [])},
                    output={
                        "intent_tier": intent_tier,
                        "context_need_tier": context_need_tier,
                        "flags": {
                            "is_refusal": is_refusal,
                            "bypass_retrieval": bypass_retrieval,
                            "is_compound": is_compound,
                            "is_terse": getattr(rw, "is_terse", False) if rw else False,
                        },
                        "reasoning": reasoning,
                    },
                    metadata={
                        "latency_ms": round(gate_latency_ms or latencies.get("gate1_ms", 0.0) if latencies else gate_latency_ms, 2),
                        "confidence": conf,
                    },
                    version=self.release,
                )

                # 1.1 Decision Model 1 (Jev System One)
                jev_obs = gate_obs.start_observation(
                    name="jev_decision_model_1",
                    as_type="generation",
                    input=question,
                    output={
                        "intent_tier": intent_tier,
                        "context_need_tier": context_need_tier,
                        "is_compound": is_compound,
                        "confidence": conf,
                    },
                    model="oc/jev-1.13-free",
                    usage_details={"input": actual_jev_tokens, "output": 0, "total": actual_jev_tokens},
                    cost_details={"total": cost_jev},
                    metadata={"rate_per_million_usd": jev_rate},
                    version=self.release,
                )
                _stamp(jev_obs, gate_start_ns + int(500_000), gate_end_ns - int(500_000))
                _stamp(gate_obs, gate_start_ns, gate_end_ns)
                cursor_ns = gate_end_ns + int(1_000_000)

                # -------------------------------------------------------------
                # 2. Stage: query_transformation (Type: CHAIN)
                # -------------------------------------------------------------
                trans_type = getattr(rw, "transformation_type", "bypass") if rw else "bypass"
                if not is_refusal:
                    trans_start_ns = cursor_ns
                    trans_end_ns = trans_start_ns + int(l_trans * 1_000_000)
                    trans_obs = root.start_observation(
                        name="query_transformation",
                        as_type="chain",
                        input={"raw_query": question, "transformation_type": trans_type},
                        output={
                            "effective_query": effective_query or question,
                            "sub_queries": sub_queries,
                        },
                        metadata={
                            "latency_ms": round(rewrite_latency_ms or (latencies.get("distill_ms", 0.0) if latencies else 0.0), 2),
                            "transformation_type": trans_type,
                        },
                        version=self.release,
                    )

                    if actual_transform_tokens > 0:
                        trans_llm_obs = trans_obs.start_observation(
                            name="transformation_llm",
                            as_type="generation",
                            input=question,
                            output=effective_query or question,
                            model=settings.llm_model,
                            usage_details={"input": actual_transform_tokens, "output": 0, "total": actual_transform_tokens},
                            cost_details={"total": cost_transform},
                            version=self.release,
                        )
                        _stamp(trans_llm_obs, trans_start_ns + int(500_000), trans_end_ns)

                    _stamp(trans_obs, trans_start_ns, trans_end_ns)
                    cursor_ns = trans_end_ns + int(1_000_000)

                # -------------------------------------------------------------
                # 3. Stage: conversational_memory (Type: CHAIN)
                # -------------------------------------------------------------
                hist_items = getattr(rw, "selected_history_items", None) or []
                selected_count = len(hist_items)
                recent_5_count = sum(1 for item in hist_items if getattr(item, "is_recent_5", False))
                older_summary_count = selected_count - recent_5_count
                badge = getattr(rw, "conversational_context_badge", None)

                if history or selected_count > 0 or actual_subq_tokens > 0 or (context_need_tier in ("needs_history_only", "needs_history_and_retrieve")):
                    mem_start_ns = cursor_ns
                    mem_end_ns = mem_start_ns + int(l_subq * 1_000_000)
                    mem_obs = root.start_observation(
                        name="conversational_memory",
                        as_type="chain",
                        input={"history_len": len(history or []), "selected_context_turn_id": getattr(rw, "selected_context_turn_id", None)},
                        output={
                            "selected_turns_count": selected_count,
                            "recent_5_full_count": recent_5_count,
                            "older_summary_count": older_summary_count,
                            "badge": badge,
                        },
                        metadata={
                            "subq_latency_ms": round(subq_latency_ms or (latencies.get("subq_model2_ms", 0.0) if latencies else 0.0), 2),
                            "history_eval_ms": round(latencies.get("history_eval_ms", 0.0) if latencies else 0.0, 2),
                        },
                        version=self.release,
                    )

                    curr_child_s_ns = mem_start_ns + int(500_000)
                    # 3.1 summary_backfill (if any summaries were missing and generated)
                    if actual_backfilled > 0:
                        backfill_end_ns = curr_child_s_ns + int(1_000_000)
                        backfill_obs = mem_obs.start_observation(
                            name="summary_backfill",
                            as_type="generation",
                            input={"missing_summaries_count": actual_backfilled},
                            output={"backfilled_count": actual_backfilled},
                            model=settings.llm_model,
                            metadata={"latency_ms": round(latencies.get("summary_backfill_ms", 0.0) if latencies else 0.0, 2)},
                            version=self.release,
                        )
                        _stamp(backfill_obs, curr_child_s_ns, backfill_end_ns)
                        curr_child_s_ns = backfill_end_ns + int(500_000)

                    # 3.2 Decision Model 2 (SubQ System One)
                    subq_end_ns = mem_end_ns - int(2_000_000)
                    if subq_end_ns <= curr_child_s_ns:
                        subq_end_ns = curr_child_s_ns + int(2_000_000)
                        mem_end_ns = subq_end_ns + int(2_000_000)

                    subq_obs = mem_obs.start_observation(
                        name="decision_model_2",
                        as_type="generation",
                        input={"query": effective_query or question, "candidate_turns_count": len(history or [])},
                        output={
                            "turn_relevance_map": subq_evaluations or {
                                getattr(item, "turn_id", ""): getattr(getattr(item, "relevance_tier", None), "value", "needed_relevant")
                                for item in hist_items
                            }
                        },
                        model="subq-snap-preview",
                        usage_details={"input": actual_subq_tokens, "output": 0, "total": actual_subq_tokens},
                        cost_details={"total": cost_subq},
                        metadata={
                            "rate_per_million_usd": subq_rate,
                            "latency_ms": round(subq_latency_ms or (latencies.get("subq_model2_ms", 0.0) if latencies else 0.0), 2),
                        },
                        version=self.release,
                    )
                    _stamp(subq_obs, curr_child_s_ns, subq_end_ns)

                    # 3.3 Fidelity Selection
                    fidelity_obs = mem_obs.start_observation(
                        name="fidelity_selection",
                        as_type="chain",
                        input={"selected_count": selected_count},
                        output={
                            "recent_5_full_count": recent_5_count,
                            "older_summary_count": older_summary_count,
                            "conversational_context_badge": badge,
                        },
                        version=self.release,
                    )
                    _stamp(fidelity_obs, subq_end_ns + int(500_000), mem_end_ns)
                    _stamp(mem_obs, mem_start_ns, mem_end_ns)
                    cursor_ns = mem_end_ns + int(1_000_000)

                # -------------------------------------------------------------
                # 4. Stage: retrieval_chain (Type: CHAIN)
                # -------------------------------------------------------------
                if not bypass_retrieval and not is_refusal:
                    dense_list = dense_results or []
                    bm25_list = bm25_results or []
                    fused_list = fused_candidates or []

                    ret_start_ns = cursor_ns
                    ret_tot_ns = int((l_ret + l_rerank + l_exp) * 1_000_000)
                    ret_end_ns = ret_start_ns + ret_tot_ns

                    retrieval_chain_obs = root.start_observation(
                        name="retrieval_chain",
                        as_type="chain",
                        input={
                            "query": effective_query or question,
                            "sub_queries": sub_queries if sub_queries else [effective_query or question],
                        },
                        output={
                            "total_fused_candidates": len(fused_list),
                            "dense_candidates_count": len(dense_list),
                            "bm25_candidates_count": len(bm25_list),
                        },
                        metadata={"latency_ms": round(retrieval_latency_ms, 2)},
                        version=self.release,
                    )

                    # 4.1 Child: bm25_search (Type: RETRIEVER)
                    bm25_obs = retrieval_chain_obs.start_observation(
                        name="bm25_search",
                        as_type="retriever",
                        input={"query": effective_query or question},
                        output={
                            "candidates_count": len(bm25_list),
                            "candidates": [
                                {
                                    "chunk_id": getattr(c, "chunk_id", ""),
                                    "score": round(getattr(c, "bm25_score", 0.0), 4) if getattr(c, "bm25_score", None) is not None else None,
                                    "rank": getattr(c, "bm25_rank", None),
                                    "page": getattr(c, "page", None),
                                    "breadcrumbs": getattr(c, "breadcrumbs", ""),
                                }
                                for c in bm25_list[:15]
                            ],
                        },
                        version=self.release,
                    )
                    _stamp(bm25_obs, ret_start_ns + int(500_000), ret_start_ns + int(l_ret * 0.4 * 1_000_000))

                    # 4.2 Child: dense_qdrant_search (Type: RETRIEVER)
                    dense_obs = retrieval_chain_obs.start_observation(
                        name="dense_qdrant_search",
                        as_type="retriever",
                        input={"query": effective_query or question},
                        output={
                            "candidates_count": len(dense_list),
                            "candidates": [
                                {
                                    "chunk_id": getattr(c, "chunk_id", ""),
                                    "score": round(getattr(c, "dense_score", 0.0), 4) if getattr(c, "dense_score", None) is not None else None,
                                    "rank": getattr(c, "dense_rank", None),
                                    "page": getattr(c, "page", None),
                                    "breadcrumbs": getattr(c, "breadcrumbs", ""),
                                }
                                for c in dense_list[:15]
                            ],
                        },
                        metadata={"vector_db": "qdrant"},
                        version=self.release,
                    )
                    _stamp(dense_obs, ret_start_ns + int(500_000), ret_start_ns + int(l_ret * 0.8 * 1_000_000))

                    # 4.3 Child: rrf_fusion (Type: CHAIN)
                    rrf_start_ns = ret_start_ns + int(l_ret * 0.85 * 1_000_000)
                    rrf_end_ns = ret_start_ns + int(l_ret * 1_000_000)
                    rrf_obs = retrieval_chain_obs.start_observation(
                        name="rrf_fusion",
                        as_type="chain",
                        input={
                            "dense_count": len(dense_list),
                            "bm25_count": len(bm25_list),
                            "k_constant": 60,
                        },
                        output={
                            "fused_count": len(fused_list),
                            "fused_candidates": [
                                {
                                    "chunk_id": getattr(c, "chunk_id", ""),
                                    "rrf_score": round(getattr(c, "rrf_score", 0.0), 5),
                                    "dense_rank": getattr(c, "dense_rank", None),
                                    "bm25_rank": getattr(c, "bm25_rank", None),
                                    "rank_delta": abs((getattr(c, "dense_rank", 999) or 999) - (getattr(c, "bm25_rank", 999) or 999)) if (getattr(c, "dense_rank", None) and getattr(c, "bm25_rank", None)) else None,
                                    "page": getattr(c, "page", None),
                                }
                                for c in fused_list[:15]
                            ],
                        },
                        version=self.release,
                    )
                    _stamp(rrf_obs, rrf_start_ns, rrf_end_ns)

                    # 4.4 Child: neural_reranking (Type: CHAIN)
                    overfetched = input_candidates or fused_list
                    reranked_list = reranked_candidates or []
                    rerank_start_ns = rrf_end_ns + int(500_000)
                    rerank_end_ns = rerank_start_ns + int(l_rerank * 1_000_000)

                    rerank_obs = retrieval_chain_obs.start_observation(
                        name="neural_reranking",
                        as_type="chain",
                        input={
                            "query": effective_query or question,
                            "overfetched_candidates_count": len(overfetched),
                        },
                        output={
                            "top_reranked_count": len(reranked_list),
                            "top_candidates": [
                                {
                                    "chunk_id": getattr(rc.candidate, "chunk_id", ""),
                                    "cross_encoder_logits": round(getattr(rc, "cross_encoder_score", 0.0), 4),
                                    "rrf_rank": getattr(rc.candidate, "rrf_rank", None),
                                    "final_rank": getattr(rc, "final_rank", None),
                                    "rank_delta": (getattr(rc.candidate, "rrf_rank", getattr(rc, "final_rank", 1)) - getattr(rc, "final_rank", 1)) if getattr(rc.candidate, "rrf_rank", None) else 0,
                                    "page": getattr(rc.candidate, "page", None),
                                    "breadcrumbs": getattr(rc.candidate, "breadcrumbs", ""),
                                }
                                for rc in reranked_list
                            ],
                            "circuit_breaker_status": circuit_breaker_status,
                        },
                        metadata={
                            "model": "BAAI/bge-reranker-v2-m3",
                            "circuit_breaker_status": circuit_breaker_status,
                            "latency_ms": round(rerank_latency_ms, 2),
                        },
                        version=self.release,
                    )
                    _stamp(rerank_obs, rerank_start_ns, rerank_end_ns)

                    # 4.5 Child: parent_expansion (Type: CHAIN)
                    exp_start_ns = rerank_end_ns + int(500_000)
                    exp_end_ns = exp_start_ns + int(l_exp * 1_000_000)
                    is_reorder_on = bool(
                        context_reordering_enabled
                        if context_reordering_enabled is not None
                        else getattr(settings, "enable_context_reordering", False)
                    )
                    sections = (
                        reordered_sections
                        if (is_reorder_on and reordered_sections)
                        else (getattr(expanded_context, "parent_sections", []) if expanded_context else [])
                    )
                    before_tokens = sum(
                        getattr(rc.candidate, "token_count", 0) or len(getattr(rc.candidate, "text", "").split())
                        for rc in reranked_list
                    )
                    after_tokens = getattr(expanded_context, "total_tokens", 0) if expanded_context else sum(getattr(s, "token_count", 0) for s in sections)

                    parent_output_payload: Dict[str, Any] = {
                        "parent_sections_count": len(sections),
                        "child_to_parent_mappings": [
                            {
                                "child_chunk_id": getattr(rc.candidate, "chunk_id", ""),
                                "parent_section_id": getattr(rc.candidate, "section_id", ""),
                                "chapter": getattr(rc.candidate, "chapter", None),
                                "page": getattr(rc.candidate, "page", None),
                            }
                            for rc in reranked_list
                        ],
                        "token_budget": {
                            "before_expansion_tokens": before_tokens,
                            "after_expansion_tokens": after_tokens,
                            "token_ceiling": 2500,
                        },
                    }
                    if is_reorder_on and reordered_sections:
                        parent_output_payload["lost_in_the_middle_layout"] = [
                            {
                                "layout_position": idx,
                                "section_id": getattr(sec, "section_id", ""),
                                "chapter": getattr(sec, "chapter", None),
                                "title": getattr(sec, "title", ""),
                                "page_range": f"{getattr(sec, 'page_start', '')} - {getattr(sec, 'page_end', '')}",
                                "token_count": getattr(sec, "token_count", 0),
                            }
                            for idx, sec in enumerate(sections)
                        ]

                    parent_obs = retrieval_chain_obs.start_observation(
                        name="parent_expansion",
                        as_type="chain",
                        input={
                            "source_child_chunk_ids": [getattr(rc.candidate, "chunk_id", "") for rc in reranked_list],
                            "max_sections_budget": 3,
                            "token_ceiling_budget": 2500,
                        },
                        output=parent_output_payload,
                        metadata={
                            "latency_ms": round(expansion_latency_ms, 2),
                            "context_reordering_enabled": is_reorder_on,
                        },
                        version=self.release,
                    )
                    _stamp(parent_obs, exp_start_ns, exp_end_ns)
                    _stamp(retrieval_chain_obs, ret_start_ns, exp_end_ns)
                    cursor_ns = exp_end_ns + int(1_000_000)

                # -------------------------------------------------------------
                # 5. Stage: response_synthesis (Type: GENERATION)
                # -------------------------------------------------------------
                model_name = getattr(synthesis_result, "model", None) or settings.llm_model
                actual_raw = raw_completion or getattr(synthesis_result, "raw_completion", "") or answer
                actual_prompt = full_prompt or getattr(synthesis_result, "full_prompt", "") or question

                synth_start_ns = cursor_ns
                synth_end_ns = synth_start_ns + int(l_synth * 1_000_000)

                stage5 = root.start_observation(
                    name="response_synthesis",
                    as_type="generation",
                    input=actual_prompt,
                    output=actual_raw,
                    model=model_name,
                    usage_details={"input": p_tok, "output": c_tok, "total": t_tok},
                    cost_details={"total": cost_synth},
                    metadata={
                        "is_refusal": is_refusal,
                        "is_compound": is_compound,
                        "ttft_ms": ttft_ms,
                        "latency_seconds": round(synthesis_latency_ms / 1000.0, 3),
                    },
                    version=self.release,
                )
                _stamp(stage5, synth_start_ns, synth_end_ns)

                root_end_ns = synth_end_ns + int(1_000_000)
                _stamp(root, root_start_ns, root_end_ns)
                return getattr(root, "trace_id", getattr(root, "id", "offline_trace"))

            if propagate_attributes:
                with propagate_attributes(
                    tags=combined_tags,
                    session_id=session_id,
                    user_id=user_id,
                    trace_name=name,
                    version=self.release,
                    metadata=combined_metadata,
                ):
                    return _build_advanced_observations()
            else:
                return _build_advanced_observations()

        except Exception as e:
            return "offline_trace"

    def add_trace_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record an empirical evaluation score linked to a trace ID in Langfuse.
        Supported metric names: gate_pass, faithfulness, answer_relevancy, context_precision, context_recall, judge_latency.
        """
        if not self.client or not trace_id or trace_id == "offline_trace":
            return

        try:
            # Langfuse v4 API: create_score
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=float(value),
                comment=comment,
                metadata=metadata,
            )
        except Exception:
            try:
                if hasattr(self.client, "score"):
                    self.client.score(
                        trace_id=trace_id,
                        name=name,
                        value=float(value),
                        comment=comment,
                    )
            except Exception:
                pass

    def flush(self):
        """Flush queued telemetry events to Langfuse backend."""
        if self.client:
            try:
                self.client.flush()
            except Exception:
                pass

    def shutdown(self):
        """Gracefully shutdown telemetry client."""
        if self.client:
            try:
                if hasattr(self.client, "shutdown"):
                    self.client.shutdown()
                else:
                    self.client.flush()
            except Exception:
                pass

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
            combined_metadata = metadata or {}

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

    def trace_advanced_pipeline(
        self,
        name: str = "advanced_science_rag",
        question: str = "",
        answer: str = "",
        is_refusal: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Stage 1: Query Understanding
        rewrite_decision: Optional[Any] = None,
        history: Optional[List[Dict[str, str]]] = None,
        effective_query: str = "",
        rewrite_latency_ms: float = 0.0,
        # Stage 2: Hybrid Retrieval
        dense_results: Optional[List[Any]] = None,
        bm25_results: Optional[List[Any]] = None,
        fused_candidates: Optional[List[Any]] = None,
        retrieval_latency_ms: float = 0.0,
        # Stage 3: Neural Reranking
        input_candidates: Optional[List[Any]] = None,
        reranked_candidates: Optional[List[Any]] = None,
        circuit_breaker_status: str = "active",
        rerank_latency_ms: float = 0.0,
        # Stage 4: Parent Expansion & Context Ordering
        expanded_context: Optional[Any] = None,
        reordered_sections: Optional[List[Any]] = None,
        expansion_latency_ms: float = 0.0,
        # Stage 5: Response Synthesis
        synthesis_result: Optional[Any] = None,
        full_prompt: str = "",
        raw_completion: str = "",
        synthesis_latency_ms: float = 0.0,
        # Latency breakdown
        latencies: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Create the full 5-stage hierarchical trace structure adhering to the enterprise contract:
        Root Trace: advanced_science_rag (CHAIN)
        ├── 1. query_understanding (CHAIN)
        ├── 2. hybrid_retrieval (CHAIN)
        │     ├── 2.1 bm25_search (RETRIEVER)
        │     ├── 2.2 dense_qdrant_search (RETRIEVER)
        │     └── 2.3 rrf_fusion (CHAIN)
        ├── 3. neural_reranking (CHAIN)
        ├── 4. parent_expansion_and_ordering (CHAIN)
        └── 5. response_synthesis (GENERATION)
        """
        if not self.client:
            return "offline_trace"

        try:
            combined_tags = ["pipeline:advanced", "domain:science_textbook"]
            if tags:
                combined_tags.extend(tags)
            combined_metadata = metadata or {}
            if latencies:
                combined_metadata["latency_profile"] = latencies

            def _build_advanced_observations():
                # Root Trace: advanced_science_rag (Type: CHAIN)
                root = self.client.start_observation(
                    name=name,
                    as_type="chain",
                    input={"question": question, "session_id": session_id},
                    output={"answer": answer, "is_refusal": is_refusal},
                    metadata=combined_metadata,
                    version=self.release,
                )

                # -------------------------------------------------------------
                # 1. Stage: query_understanding (Type: CHAIN)
                # -------------------------------------------------------------
                rw = rewrite_decision
                needs_rw = getattr(rw, "needs_rewrite", False) if rw else False
                conf = getattr(rw, "confidence", 1.0) if rw else 1.0
                sub_queries = getattr(rw, "sub_queries", []) if rw else []
                reasoning = getattr(rw, "reasoning", "") if rw else ""
                rewritten = getattr(rw, "rewritten_query", None) if rw else None
                is_resolved = bool(needs_rw and rewritten and (rewritten != question))

                stage1 = root.start_observation(
                    name="query_understanding",
                    as_type="chain",
                    input={"query": question, "history": history or []},
                    output={
                        "effective_query": effective_query or question,
                        "needs_rewrite": needs_rw,
                        "confidence": conf,
                        "reasoning": reasoning,
                        "sub_query_decomposition": sub_queries,
                        "pronoun_resolution": {
                            "resolved": is_resolved,
                            "confidence_score": conf,
                            "resolved_query": rewritten if is_resolved else None,
                        },
                    },
                    metadata={"latency_ms": round(rewrite_latency_ms, 2)},
                    version=self.release,
                )
                stage1.end()

                # -------------------------------------------------------------
                # 2. Stage: hybrid_retrieval (Type: CHAIN)
                # -------------------------------------------------------------
                dense_list = dense_results or []
                bm25_list = bm25_results or []
                fused_list = fused_candidates or []

                stage2 = root.start_observation(
                    name="hybrid_retrieval",
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

                # 2.1 Child: bm25_search (Type: RETRIEVER)
                bm25_obs = stage2.start_observation(
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
                bm25_obs.end()

                # 2.2 Child: dense_qdrant_search (Type: RETRIEVER)
                dense_obs = stage2.start_observation(
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
                dense_obs.end()

                # 2.3 Child: rrf_fusion (Type: CHAIN)
                rrf_obs = stage2.start_observation(
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
                rrf_obs.end()

                stage2.end()

                # -------------------------------------------------------------
                # 3. Stage: neural_reranking (Type: CHAIN)
                # -------------------------------------------------------------
                overfetched = input_candidates or fused_list
                reranked_list = reranked_candidates or []

                stage3 = root.start_observation(
                    name="neural_reranking",
                    as_type="chain",
                    input={
                        "query": effective_query or question,
                        "overfetched_candidates_count": len(overfetched),
                        "input_candidates": [
                            {"chunk_id": getattr(c, "chunk_id", ""), "rrf_score": round(getattr(c, "rrf_score", 0.0), 5)}
                            for c in overfetched[:15]
                        ],
                    },
                    output={
                        "top_reranked_count": len(reranked_list),
                        "top_candidates": [
                            {
                                "chunk_id": getattr(rc.candidate, "chunk_id", ""),
                                "cross_encoder_logits": round(getattr(rc, "cross_encoder_score", 0.0), 4),
                                "final_rank": getattr(rc, "final_rank", None),
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
                stage3.end()

                # -------------------------------------------------------------
                # 4. Stage: parent_expansion_and_ordering (Type: CHAIN)
                # -------------------------------------------------------------
                sections = reordered_sections or (getattr(expanded_context, "parent_sections", []) if expanded_context else [])
                before_tokens = sum(
                    getattr(rc.candidate, "token_count", 0) or len(getattr(rc.candidate, "text", "").split())
                    for rc in reranked_list
                )
                after_tokens = getattr(expanded_context, "total_tokens", 0) if expanded_context else sum(getattr(s, "token_count", 0) for s in sections)

                stage4 = root.start_observation(
                    name="parent_expansion_and_ordering",
                    as_type="chain",
                    input={
                        "source_child_chunk_ids": [getattr(rc.candidate, "chunk_id", "") for rc in reranked_list],
                        "max_sections_budget": 3,
                        "token_ceiling_budget": 2500,
                    },
                    output={
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
                        "lost_in_the_middle_layout": [
                            {
                                "layout_position": idx,
                                "section_id": getattr(sec, "section_id", ""),
                                "chapter": getattr(sec, "chapter", None),
                                "title": getattr(sec, "title", ""),
                                "page_range": f"{getattr(sec, 'page_start', '')} - {getattr(sec, 'page_end', '')}",
                                "token_count": getattr(sec, "token_count", 0),
                            }
                            for idx, sec in enumerate(sections)
                        ],
                    },
                    metadata={"latency_ms": round(expansion_latency_ms, 2)},
                    version=self.release,
                )
                stage4.end()

                # -------------------------------------------------------------
                # 5. Stage: response_synthesis (Type: GENERATION)
                # -------------------------------------------------------------
                model_name = getattr(synthesis_result, "model", None) or settings.llm_model
                p_tok = getattr(synthesis_result, "prompt_tokens", 0)
                c_tok = getattr(synthesis_result, "completion_tokens", 0)
                t_tok = getattr(synthesis_result, "total_tokens", p_tok + c_tok)
                actual_raw = raw_completion or getattr(synthesis_result, "raw_completion", "") or answer
                actual_prompt = full_prompt or getattr(synthesis_result, "full_prompt", "") or question

                stage5 = root.start_observation(
                    name="response_synthesis",
                    as_type="generation",
                    input=actual_prompt,
                    output=actual_raw,
                    model=model_name,
                    usage_details={"input": p_tok, "output": c_tok, "total": t_tok},
                    metadata={
                        "is_refusal": is_refusal,
                        "latency_seconds": round(synthesis_latency_ms / 1000.0, 3),
                    },
                    version=self.release,
                )
                stage5.end()

                root.end()
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

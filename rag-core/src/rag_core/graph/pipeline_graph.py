#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Compiled RAG Pipeline Graph.
Assembles the complete end-to-end RAG state graph with conditional branches,
native Langfuse tracing, latency profiling, and token/cost attribution.
"""

import time
import re
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
from langgraph.graph import StateGraph, END

from rag_core.graph.state import RAGGraphState
from rag_core.indexing.science_schema import format_history_turn_for_context
from rag_core.graph.nodes import (
    create_guard_and_route_node,
    refusal_node,
    create_retrieve_node,
    create_rerank_node,
    create_synthesize_node,
    create_evaluate_node
)
from rag_core.observability.langfuse import LangfuseTracker


def should_refuse(state: RAGGraphState) -> str:
    """Conditional edge determining whether to short-circuit into refusal node."""
    if state.get("is_adversarial", False):
        return "refusal"
    return "retrieve"


def should_evaluate(state: RAGGraphState) -> str:
    """Conditional edge determining whether to run evaluation judge."""
    if state.get("ground_truth") is not None:
        return "evaluate"
    return "end"


class LangGraphRAGPipeline:
    """
    Production-grade LangGraph StateGraph RAG Pipeline.
    Supports baseline naive execution and improved secondary pipeline modes.
    """

    def __init__(
        self,
        retriever: Any,
        synthesizer: Any,
        router: Optional[Any] = None,
        reranker: Optional[Any] = None,
        judge: Optional[Any] = None,
        enable_reranking: bool = False,
        pipeline_name: str = "baseline",
        top_k: int = 5,
        top_n: int = 3
    ):
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.router = router
        self.reranker = reranker
        self.judge = judge
        self.enable_reranking = enable_reranking
        self.pipeline_name = pipeline_name
        self.top_k = top_k
        self.top_n = top_n
        self.tracker = LangfuseTracker()

        # Build and compile StateGraph
        self.graph = self._compile_graph()

    def _compile_graph(self):
        """Construct and compile the LangGraph workflow."""
        workflow = StateGraph(RAGGraphState)

        # Register nodes
        workflow.add_node("guard_and_route", create_guard_and_route_node(self.router))
        workflow.add_node("refusal", refusal_node)
        workflow.add_node("retrieve", create_retrieve_node(self.retriever, top_k=self.top_k))
        workflow.add_node("rerank", create_rerank_node(self.reranker, enable_reranking=self.enable_reranking, top_n=self.top_n))
        workflow.add_node("synthesize", create_synthesize_node(self.synthesizer))
        workflow.add_node("evaluate", create_evaluate_node(self.judge))

        # Define graph edges
        workflow.set_entry_point("guard_and_route")
        workflow.add_conditional_edges(
            "guard_and_route",
            should_refuse,
            {"refusal": "refusal", "retrieve": "retrieve"}
        )
        workflow.add_edge("refusal", END)
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "synthesize")
        workflow.add_conditional_edges(
            "synthesize",
            should_evaluate,
            {"evaluate": "evaluate", "end": END}
        )
        workflow.add_edge("evaluate", END)

        return workflow.compile()

    def invoke(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        suite: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RAGGraphState:
        """
        Execute query through the compiled LangGraph pipeline with native Langfuse telemetry.
        """
        trace_tags = [self.pipeline_name]
        if self.enable_reranking:
            trace_tags.append("jev-reranker")
        if tags:
            trace_tags.extend(tags)

        # Prepare initial state
        initial_state: RAGGraphState = {
            "user_query": query,
            "ground_truth": ground_truth,
            "suite": suite,
            "tags": trace_tags,
            "metadata": metadata or {},
            "latencies": {},
            "tokens": {}
        }

        # Configure Langfuse callback handler if active
        config: Dict[str, Any] = {"tags": trace_tags, "metadata": metadata or {}}
        cb_handler = self.tracker.get_langchain_handler(tags=trace_tags, metadata=metadata)
        if cb_handler:
            config["callbacks"] = [cb_handler]

        # Invoke StateGraph
        final_state: RAGGraphState = self.graph.invoke(initial_state, config=config)

        # Record top-level trace in tracker
        trace_id = self.tracker.trace_query(
            name=f"langgraph_{self.pipeline_name}",
            question=query,
            answer=final_state.get("agent_response", ""),
            top_k=self.top_n,
            retrieved_count=len(final_state.get("reranked_chunks", [])),
            sources=final_state.get("sources", []),
            retrieval_latency=final_state.get("latencies", {}).get("retrieval", 0.0),
            generation_latency=final_state.get("latencies", {}).get("synthesis", 0.0),
            metadata={
                "pipeline_name": self.pipeline_name,
                "latencies": final_state.get("latencies", {}),
                "tokens": final_state.get("tokens", {}),
                "cost_usd": final_state.get("cost_usd", 0.0),
                "tags": trace_tags
            }
        )
        final_state["trace_id"] = trace_id

        # Ingest evaluation verdicts into Langfuse trace scores if available
        if "evaluation_verdicts" in final_state and self.tracker.is_active:
            for metric_name, v_info in final_state["evaluation_verdicts"].items():
                self.tracker.add_trace_score(
                    trace_id=trace_id,
                    name=metric_name,
                    value=float(v_info.get("verdict", 1)),
                    comment=f"Score: {v_info.get('score')}, Latency: {v_info.get('latency_seconds')}s"
                )
            if "overall_verdict" in final_state:
                self.tracker.add_trace_score(
                    trace_id=trace_id,
                    name="overall_verdict",
                    value=float(final_state["overall_verdict"])
                )

        return final_state


# ---------------------------------------------------------------------------
# Enterprise Science LangGraph StateGraph Pipeline
# ---------------------------------------------------------------------------

def should_level0_refuse(state: RAGGraphState) -> str:
    """Conditional edge determining whether Level-0 routing rejected the query."""
    if state.get("is_refusal", False) or state.get("is_adversarial", False):
        return "refusal"
    return "gated_rewrite"


def should_enterprise_refuse(state: RAGGraphState) -> str:
    """Conditional edge determining whether query was halted by the pre-retrieval gate or routes directly to synthesis."""
    if state.get("is_refusal", False) or not state.get("proceed_to_retrieval", True):
        need = state.get("context_need_tier")
        if need == "needs_history_only":
            return "synthesis"
        return "refusal"
    return "hybrid_retrieve"


class EnterpriseScienceGraph:
    """
    Standard LangGraph StateGraph pipeline for the Enterprise Science RAG Engine.
    Implements a 5-stage declarative computation graph with conditional refusal branching,
    native Langfuse CallbackHandler support, and asynchronous token streaming.
    """

    def __init__(
        self,
        rewriter: Optional[Any] = None,
        hybrid_retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        expander: Optional[Any] = None,
        reorderer: Optional[Any] = None,
        synthesizer: Optional[Any] = None,
        level0_router: Optional[Any] = None,
        top_k: int = 5,
        overfetch_k: int = 50,
        warmup: bool = False
    ):
        from rag_core.graph.nodes import (
            create_level0_source_router_node,
            create_enterprise_gated_rewrite_node,
            create_enterprise_refusal_node,
            create_enterprise_hybrid_retrieve_node,
            create_enterprise_neural_rerank_node,
            create_enterprise_expansion_node,
            create_enterprise_synthesis_node
        )

        self.top_k = top_k
        self.overfetch_k = overfetch_k
        self.tracker = LangfuseTracker()
        self.rewriter = rewriter
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.expander = expander
        self.reorderer = reorderer
        self.synthesizer = synthesizer
        self.level0_router = level0_router

        self.last_citations: List[Any] = []
        self.last_is_refusal: bool = False
        self.last_effective_query: str = ""
        self.last_badge: Optional[str] = None
        self._session_citations: Dict[str, Any] = {}
        self._session_is_refusal: Dict[str, bool] = {}
        self._session_effective_query: Dict[str, str] = {}
        self._session_latencies: Dict[str, Dict[str, float]] = {}
        self.last_summary: Optional[str] = None
        self._session_summaries: Dict[str, Optional[str]] = {}
        self.last_full_prompt: Optional[str] = None
        self._session_full_prompts: Dict[str, Optional[str]] = {}

        # Build declarative StateGraph
        workflow = StateGraph(RAGGraphState)

        # Register nodes
        workflow.add_node("level0_router", create_level0_source_router_node(level0_router))
        workflow.add_node("gated_rewrite", create_enterprise_gated_rewrite_node(rewriter))
        workflow.add_node("refusal", create_enterprise_refusal_node())
        workflow.add_node("hybrid_retrieve", create_enterprise_hybrid_retrieve_node(hybrid_retriever, overfetch_k=overfetch_k))
        workflow.add_node("neural_rerank", create_enterprise_neural_rerank_node(reranker, top_k=top_k))
        workflow.add_node("parent_expansion", create_enterprise_expansion_node(expander, reorderer))
        workflow.add_node("synthesis", create_enterprise_synthesis_node(synthesizer))

        # Define graph edges
        workflow.set_entry_point("level0_router")
        workflow.add_conditional_edges(
            "level0_router",
            should_level0_refuse,
            {"refusal": "refusal", "gated_rewrite": "gated_rewrite"}
        )
        workflow.add_conditional_edges(
            "gated_rewrite",
            should_enterprise_refuse,
            {"refusal": "refusal", "hybrid_retrieve": "hybrid_retrieve", "synthesis": "synthesis"}
        )
        workflow.add_edge("refusal", END)
        workflow.add_edge("hybrid_retrieve", "neural_rerank")
        workflow.add_edge("neural_rerank", "parent_expansion")
        workflow.add_edge("parent_expansion", "synthesis")
        workflow.add_edge("synthesis", END)

        # Compile graph
        self.graph = workflow.compile()

        if warmup:
            try:
                if reranker and hasattr(reranker, "warmup"):
                    reranker.warmup()
            except Exception:
                pass

    def _resolve_level0_router(self) -> Any:
        if self.level0_router is not None:
            return self.level0_router
        from rag_core.query.jev_router import Level0SourceRouter
        self.level0_router = Level0SourceRouter()
        return self.level0_router

    def _resolve_synthesizer(self) -> Any:
        if self.synthesizer is not None:
            return self.synthesizer
        from rag_core.generation.synthesizer import ResponseSynthesizer
        self.synthesizer = ResponseSynthesizer()
        return self.synthesizer

    def _resolve_rewriter(self) -> Any:
        if self.rewriter is not None:
            return self.rewriter
        from rag_core.query.query_gate import QueryGate
        self.rewriter = QueryGate()
        return self.rewriter

    def _resolve_retriever(self) -> Any:
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever
        from rag_core.retrieval.hybrid import HybridRetriever
        self.hybrid_retriever = HybridRetriever()
        return self.hybrid_retriever

    def _resolve_reranker(self) -> Any:
        if self.reranker is not None:
            return self.reranker
        from rag_core.fusion.reranker import CrossEncoderReranker
        self.reranker = CrossEncoderReranker()
        return self.reranker

    def _resolve_expander(self) -> Any:
        if self.expander is not None:
            return self.expander
        from rag_core.postprocessing.expansion import ParentSectionExpander
        self.expander = ParentSectionExpander()
        return self.expander

    def _resolve_reorderer(self) -> Any:
        if self.reorderer is not None:
            return self.reorderer
        from rag_core.postprocessing.expansion import LongContextReorder
        self.reorderer = LongContextReorder()
        return self.reorderer

    def get_citations_for_session(self, session_id: Optional[str]) -> List[Any]:
        if session_id and session_id in self._session_citations:
            return self._session_citations[session_id]
        return self.last_citations

    def get_refusal_for_session(self, session_id: Optional[str]) -> bool:
        if session_id and session_id in self._session_is_refusal:
            return self._session_is_refusal[session_id]
        return self.last_is_refusal

    def get_effective_query_for_session(self, session_id: Optional[str]) -> str:
        if session_id and session_id in self._session_effective_query:
            return self._session_effective_query[session_id]
        return self.last_effective_query

    def get_latencies_for_session(self, session_id: Optional[str]) -> Dict[str, float]:
        if session_id and session_id in self._session_latencies:
            return self._session_latencies[session_id]
        return {}

    def get_summary_for_session(self, session_id: Optional[str]) -> Optional[str]:
        if session_id and session_id in self._session_summaries:
            return self._session_summaries[session_id]
        return self.last_summary

    def get_full_prompt_for_session(self, session_id: Optional[str]) -> Optional[str]:
        if session_id and session_id in self._session_full_prompts:
            return self._session_full_prompts[session_id]
        return self.last_full_prompt

    async def ainvoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        selected_chapters: Optional[List[int]] = None,
        selected_sources: Optional[List[str]] = None,
        selected_faq_categories: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> RAGGraphState:
        """Execute query asynchronously through the compiled LangGraph StateGraph."""
        initial_state: RAGGraphState = {
            "user_query": query,
            "session_id": session_id,
            "history": history,
            "selected_chapters": selected_chapters,
            "selected_sources": selected_sources or (config or {}).get("selected_sources"),
            "selected_faq_categories": selected_faq_categories or (config or {}).get("selected_faq_categories"),
            "selected_context_turn_id": (config or {}).get("selected_context_turn_id"),
            "metadata": {
                "history_enabled": (config or {}).get("history_enabled", True)
            },
            "latencies": {},
            "tokens": {}
        }
        exec_config = config or {}
        cb_handler = self.tracker.get_langchain_handler(tags=["langgraph", "enterprise_science"])
        if cb_handler and "callbacks" not in exec_config:
            exec_config["callbacks"] = [cb_handler]

        final_state: RAGGraphState = await self.graph.ainvoke(initial_state, config=exec_config)

        # Cache session metadata
        self.last_citations = final_state.get("citations", [])
        self.last_is_refusal = final_state.get("is_refusal", False)
        self.last_effective_query = final_state.get("effective_query", query)
        self.last_summary = final_state.get("q_and_a_summary")
        if session_id:
            self._session_citations[session_id] = self.last_citations
            self._session_is_refusal[session_id] = self.last_is_refusal
            self._session_effective_query[session_id] = self.last_effective_query
            self._session_latencies[session_id] = final_state.get("latencies", {})
            self._session_summaries[session_id] = self.last_summary

        return final_state

    def invoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        selected_chapters: Optional[List[int]] = None,
        selected_sources: Optional[List[str]] = None,
        selected_faq_categories: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> RAGGraphState:
        """Synchronously execute query through the compiled LangGraph StateGraph."""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return asyncio.run(self.ainvoke(
            query=query,
            session_id=session_id,
            history=history,
            selected_chapters=selected_chapters,
            selected_sources=selected_sources,
            selected_faq_categories=selected_faq_categories,
            config=config
        ))

    async def astream_tokens(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        selected_chapters: Optional[List[int]] = None,
        selected_sources: Optional[List[str]] = None,
        selected_faq_categories: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        status_callback: Optional[Any] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the Enterprise Science Graph with sub-second TTFT.
        Executes pre-retrieval gating, hybrid retrieval, cross-encoder reranking,
        and parent expansion, then yields synthesized LLM tokens progressively.
        """
        import asyncio
        from rag_core.pipeline import GROUNDED_REFUSAL_FA
        from rag_core.postprocessing.citations import CitationFormatter
        from rag_core.generation.prompts import SYSTEM_PROMPT_PERSIAN_GROUNDED, PROMPT_COMPOUND_DIRECTIVE_FA
        from rag_core.config import settings

        total_start = time.perf_counter()
        latencies: Dict[str, float] = {}

        async def _notify(step: str, title: str):
            if status_callback is not None:
                try:
                    res = status_callback(step, title)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass

        # 0. Level-0 Source Routing / Multi-Source Evaluation
        level0 = self._resolve_level0_router()
        effective_sources = selected_sources or (config or {}).get("selected_sources")
        l0_decision = level0.route_query(
            query=query,
            enabled_sources=effective_sources,
            history=history
        )
        latencies["level0_router_ms"] = l0_decision.latencies.get("router_ms", 0.0)
        if l0_decision.is_refusal:
            self.last_is_refusal = True
            if session_id:
                self._session_is_refusal[session_id] = True
            yield GROUNDED_REFUSAL_FA
            return

        # 1. Gated Rewriting / Intent Guard
        await _notify("rewrite", "تحلیل و بررسی سوال")
        t0 = time.perf_counter()
        rewriter = self._resolve_rewriter()

        use_route = False
        from rag_core.query.query_gate import QueryGate
        if isinstance(rewriter, QueryGate):
            use_route = True
        elif hasattr(rewriter, "evaluate_and_route"):
            from unittest.mock import MagicMock
            eval_fn = getattr(rewriter, "evaluate_and_route", None)
            if not isinstance(rewriter, MagicMock):
                use_route = True
            elif hasattr(eval_fn, "assert_awaited") or asyncio.iscoroutinefunction(eval_fn):
                use_route = True

        hist_enabled = (config or {}).get("history_enabled", True)
        active_sources = l0_decision.active_sources or ["science_textbook"]
        is_both = "science_textbook" in active_sources and "administrative_faq" in active_sources
        is_faq_only = "administrative_faq" in active_sources and "science_textbook" not in active_sources

        if use_route and is_both:
            selected_turn_id = (config or {}).get("selected_context_turn_id")
            sci_task = rewriter.evaluate_and_route(
                query=query,
                session_id=session_id,
                history=history,
                selected_chapters=selected_chapters,
                selected_context_turn_id=selected_turn_id,
                history_enabled=hist_enabled,
                domain="science_textbook"
            )
            faq_task = rewriter.evaluate_and_route(
                query=query,
                session_id=session_id,
                history=history,
                selected_faq_categories=selected_faq_categories,
                selected_context_turn_id=selected_turn_id,
                history_enabled=hist_enabled,
                domain="administrative_faq"
            )
            sci_dec, faq_dec = await asyncio.gather(sci_task, faq_task)
            sci_ok = not getattr(sci_dec, "is_refusal", False) and not getattr(sci_dec, "bypass_retrieval", False)
            faq_ok = not getattr(faq_dec, "is_refusal", False) and not getattr(faq_dec, "bypass_retrieval", False)

            if sci_ok and faq_ok:
                gate_decision = sci_dec
                combined_subs = list(dict.fromkeys(
                    (getattr(sci_dec, "sub_queries", []) or [sci_dec.effective_query or query]) +
                    (getattr(faq_dec, "sub_queries", []) or [faq_dec.effective_query or query])
                ))
            elif faq_ok:
                gate_decision = faq_dec
                combined_subs = getattr(faq_dec, "sub_queries", [])
            else:
                gate_decision = sci_dec
                combined_subs = getattr(sci_dec, "sub_queries", [])

            effective_query = getattr(gate_decision, "effective_query", query)
            proceed = sci_ok or faq_ok
            is_refusal = not proceed
            self.last_intent_tier = getattr(gate_decision.intent_tier, "value", str(gate_decision.intent_tier))
            self.last_context_need_tier = getattr(gate_decision.context_need_tier, "value", str(gate_decision.context_need_tier))
            is_history_only = (getattr(gate_decision.context_need_tier, "value", "") == "needs_history_only")
            sub_queries = combined_subs
            refusal_reason = getattr(gate_decision, "reasoning", "")
            self.last_gate_decision = gate_decision
        elif use_route:
            selected_turn_id = (config or {}).get("selected_context_turn_id")
            target_dom = "administrative_faq" if is_faq_only else "science_textbook"
            gate_decision = await rewriter.evaluate_and_route(
                query=query,
                session_id=session_id,
                history=history,
                selected_chapters=selected_chapters if target_dom == "science_textbook" else None,
                selected_faq_categories=selected_faq_categories if target_dom == "administrative_faq" else None,
                domain=target_dom,
                selected_context_turn_id=selected_turn_id,
                history_enabled=hist_enabled
            )
            effective_query = gate_decision.effective_query or query
            proceed = not gate_decision.is_refusal and not gate_decision.bypass_retrieval
            is_refusal = gate_decision.is_refusal
            self.last_intent_tier = getattr(gate_decision.intent_tier, "value", str(gate_decision.intent_tier))
            self.last_context_need_tier = getattr(gate_decision.context_need_tier, "value", str(gate_decision.context_need_tier))
            is_history_only = (gate_decision.context_need_tier.value == "needs_history_only")
            sub_queries = gate_decision.sub_queries
            refusal_reason = gate_decision.reasoning
            self.last_gate_decision = gate_decision
        else:
            raw_res = rewriter.evaluate_and_rewrite(
                query=query,
                history=history,
                active_chapters=selected_chapters
            )
            rewrite_decision = await raw_res if asyncio.iscoroutine(raw_res) else raw_res
            effective_query = getattr(rewrite_decision, "rewritten_query", None) or query
            proceed = getattr(rewrite_decision, "proceed_to_retrieval", True)
            is_refusal = not proceed
            is_history_only = False
            sub_queries = getattr(rewrite_decision, "sub_queries", [])
            refusal_reason = ""
            self.last_gate_decision = rewrite_decision

        latencies["rewrite_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if is_refusal:
            self.last_citations = []
            self.last_is_refusal = True
            self.last_effective_query = effective_query
            self.last_badge = None
            self.last_candidates = []
            self.last_reranked = []
            self.last_expanded_context = None
            self.last_reordered_sections = []
            self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            if session_id:
                self._session_citations[session_id] = []
                self._session_is_refusal[session_id] = True
                self._session_effective_query[session_id] = effective_query
            persian_refusal = any('\u0600' <= c <= '\u06FF' for c in refusal_reason)
            msg = refusal_reason if persian_refusal else GROUNDED_REFUSAL_FA
            yield msg
            return

        if is_history_only:
            # Bypass hybrid retrieval and neural rerank completely! (0ms retrieval latency)
            self.last_citations = []
            self.last_is_refusal = False
            self.last_effective_query = effective_query
            self.last_badge = "پاسخ بر اساس گفتگوی پیشین"
            latencies["retrieval_ms"] = 0.0
            latencies["rerank_ms"] = 0.0
            latencies["expansion_ms"] = 0.0

            await _notify("synthesis", "پاسخ بر اساس گفتگوی پیشین")
            synthesizer = self._resolve_synthesizer()
            chunks_for_synth = []
            if hasattr(gate_decision, "selected_history_items") and gate_decision.selected_history_items:
                for item in gate_decision.selected_history_items:
                    text_content = format_history_turn_for_context(item)
                    if text_content:
                        chunks_for_synth.append({
                            "text": text_content,
                            "page": 1,
                            "breadcrumb": "گفتگوی پیشین",
                            "doc_id": "conversation_history",
                            "chunk_id": getattr(item, "turn_id", "turn_hist")
                        })
            elif history:
                for t in history[-4:]:
                    content_str = format_history_turn_for_context(t)
                    if not content_str:
                        content_str = f"{t.get('role', 'user')}: {t.get('content', '')}"
                    chunks_for_synth.append({
                        "text": content_str,
                        "page": 1,
                        "breadcrumb": "گفتگوی پیشین",
                        "doc_id": "conversation_history",
                        "chunk_id": t.get("turn_id", "turn_hist")
                    })

            first_token_received = False
            synth_start = time.perf_counter()
            async for token in synthesizer.synthesize_stream(
                query=effective_query,
                chunks=chunks_for_synth,
                system_prompt=SYSTEM_PROMPT_PERSIAN_GROUNDED,
                session_id=session_id,
                history=None
            ):
                if not first_token_received:
                    first_token_received = True
                    self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
                yield token

            latencies["synthesis_ms"] = round((time.perf_counter() - synth_start) * 1000.0, 2)
            self.last_full_prompt = getattr(synthesizer, "last_full_prompt", "")
            if session_id:
                self._session_citations[session_id] = []
                self._session_is_refusal[session_id] = False
                self._session_effective_query[session_id] = effective_query
                self._session_latencies[session_id] = latencies
                self._session_full_prompts[session_id] = self.last_full_prompt
            return

        # 2. Hybrid Retrieval
        await _notify("retrieval", "جستجوی اسناد و کتاب درسی")
        t0 = time.perf_counter()
        retriever = self._resolve_retriever()

        # Augment retrieval query with historical summaries when needs_history_and_retrieve (User Story 4)
        if use_route and hasattr(gate_decision, "selected_history_items") and gate_decision.selected_history_items:
            from rag_core.query.query_gate import QueryGate
            effective_query = QueryGate.enrich_query_with_summaries(effective_query, gate_decision.selected_history_items)

        import inspect
        try:
            sig = inspect.signature(retriever.retrieve)
            accepts_source = "source_domain" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except Exception:
            accepts_source = False

        async def _run_stream_retrieve(q_str: str) -> List[Any]:
            if is_both and accepts_source:
                sci_task = retriever.retrieve(
                    query=q_str,
                    top_k=self.overfetch_k,
                    overfetch_k=self.overfetch_k,
                    chapters=selected_chapters,
                    source_domain="science_textbook"
                )
                faq_task = retriever.retrieve(
                    query=q_str,
                    top_k=self.overfetch_k,
                    overfetch_k=self.overfetch_k,
                    categories=selected_faq_categories,
                    source_domain="administrative_faq"
                )
                sci_cands, faq_cands = await asyncio.gather(sci_task, faq_task)
                combined = []
                seen_ids = set()
                for c in list(sci_cands) + list(faq_cands):
                    cid = getattr(c, "chunk_id", None) or str(c)
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        combined.append(c)
                return combined
            elif is_faq_only and accepts_source:
                return await retriever.retrieve(
                    query=q_str,
                    top_k=self.overfetch_k,
                    overfetch_k=self.overfetch_k,
                    categories=selected_faq_categories,
                    source_domain="administrative_faq"
                )
            else:
                kwargs: Dict[str, Any] = {"query": q_str, "top_k": self.overfetch_k, "overfetch_k": self.overfetch_k}
                if selected_chapters is not None and not is_faq_only:
                    kwargs["chapters"] = selected_chapters
                if accepts_source:
                    kwargs["source_domain"] = "science_textbook"
                return await retriever.retrieve(**kwargs)

        active_sub_queries = sub_queries if sub_queries else []
        if active_sub_queries and len(active_sub_queries) > 1:
            tasks = [_run_stream_retrieve(sq) for sq in active_sub_queries]
            sub_results = await asyncio.gather(*tasks)
            candidates = []
            seen = set()
            for cand_list in sub_results:
                for c in cand_list:
                    cid = getattr(c, "chunk_id", None) or str(c)
                    if cid not in seen:
                        candidates.append(c)
                        seen.add(cid)
        else:
            candidates = await _run_stream_retrieve(effective_query)
        latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if not candidates:
            self.last_citations = []
            self.last_is_refusal = True
            self.last_effective_query = effective_query
            self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            if session_id:
                self._session_citations[session_id] = []
                self._session_is_refusal[session_id] = True
                self._session_effective_query[session_id] = effective_query
                self._session_latencies[session_id] = latencies
            if selected_chapters:
                ch_str = "، ".join(str(c) for c in selected_chapters)
                refusal_msg = f"در محدوده فصل‌های انتخابی ({ch_str})، اطلاعاتی درباره این پرسش در کتاب یافت نشد. لطفاً دامنه جستجو را گسترش دهید یا تمام کتاب را انتخاب فرمایید."
            else:
                refusal_msg = GROUNDED_REFUSAL_FA
            yield refusal_msg
            return

        # 3. Cross-Encoder Neural Reranking
        await _notify("rerank", "بازرتبه‌بندی عصبی شواهد")
        t0 = time.perf_counter()
        reranker = self._resolve_reranker()
        timeout = getattr(settings, "rerank_timeout_ms", 3000.0)
        reranked = await reranker.rerank(
            query=effective_query,
            candidates=candidates[:8],
            top_k=self.top_k,
            timeout_ms=timeout
        )
        latencies["rerank_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 4. Parent Section Expansion & Context Reordering
        await _notify("expansion", "بسط و تکمیل بخش‌های مرتبط")
        t0 = time.perf_counter()
        expander = self._resolve_expander()
        reorderer = self._resolve_reorderer()
        expanded_ctx = expander.expand(candidates=reranked, max_sections=3, token_ceiling=2500)
        reordered_sections = reorderer.reorder(expanded_ctx.parent_sections)
        latencies["expansion_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        self.last_candidates = candidates
        self.last_reranked = reranked
        self.last_expanded_context = expanded_ctx
        self.last_reordered_sections = reordered_sections

        # 5. Citation Formatting
        citation_candidates = []
        from rag_core.postprocessing.citations import DOC_TO_CHAPTER
        for rc in reranked:
            c = getattr(rc, "candidate", rc)
            doc_id = getattr(c, "doc_id", "science_textbook") or "science_textbook"
            if getattr(c, "category", None) or doc_id == "administrative_faq":
                citation_candidates.append({
                    "doc_id": "administrative_faq",
                    "filename": "administrative_faq",
                    "category": getattr(c, "category", "عمومی"),
                    "row": getattr(c, "row", 1),
                    "chunk_id": getattr(c, "chunk_id", ""),
                    "score": getattr(rc, "cross_encoder_score", 0.0),
                    "metadata": {
                        "category": getattr(c, "category", "عمومی"),
                        "row": getattr(c, "row", 1),
                        "display_name": f"پرسش‌های متداول: {getattr(c, 'category', 'عمومی')}"
                    }
                })
            else:
                if not doc_id.endswith(".pdf"):
                    doc_id = f"{doc_id}.pdf"
                ch_info = DOC_TO_CHAPTER.get(doc_id)
                c_page = getattr(c, "page", 1)
                display_name = f"فصل {ch_info[0]}، صفحه {c_page}" if ch_info else f"کتاب درسی، صفحه {c_page}"
                citation_candidates.append({
                    "doc_id": doc_id,
                    "filename": doc_id,
                    "page": c_page,
                    "chunk_id": getattr(c, "chunk_id", ""),
                    "score": getattr(rc, "cross_encoder_score", 0.0),
                    "metadata": {"title": getattr(c, "breadcrumbs", ""), "display_name": display_name}
                })
        formatter = CitationFormatter()
        citation_bundle = formatter.format(citation_candidates)

        default_synth_doc = citation_candidates[0]["doc_id"] if citation_candidates else "science_textbook.pdf"

        hist_chunks = []
        if use_route and hasattr(gate_decision, "selected_history_items") and gate_decision.selected_history_items:
            for item in gate_decision.selected_history_items:
                text_content = format_history_turn_for_context(item)
                if text_content:
                    hist_chunks.append({
                        "text": text_content,
                        "page": 1,
                        "breadcrumb": "گفتگوی پیشین",
                        "doc_id": "conversation_history",
                        "chunk_id": getattr(item, "turn_id", "turn_hist")
                    })

        chunks_for_synth = list(hist_chunks)
        for sec in reordered_sections:
            is_faq_sec = getattr(sec, "chapter", 1) == 0
            sec_doc = "administrative_faq" if is_faq_sec else default_synth_doc
            bcrumb = sec.title if is_faq_sec else f"فصل {sec.chapter}: {sec.title}"
            chunks_for_synth.append({
                "text": sec.full_text,
                "page": sec.page_start,
                "breadcrumb": bcrumb,
                "doc_id": sec_doc,
                "chunk_id": sec.section_id
            })

        is_compound = False
        if use_route and hasattr(gate_decision, "is_compound"):
            is_compound = bool(gate_decision.is_compound)
        elif 'rewrite_decision' in locals():
            is_compound = bool(getattr(rewrite_decision, "is_compound", False))

        if is_both:
            from rag_core.generation.prompts import SYSTEM_PROMPT_COMPOSITE_GROUNDED
            synth_prompt = SYSTEM_PROMPT_COMPOSITE_GROUNDED
        elif is_faq_only:
            from rag_core.generation.prompts import SYSTEM_PROMPT_ADMIN_FAQ_GROUNDED
            synth_prompt = SYSTEM_PROMPT_ADMIN_FAQ_GROUNDED
        else:
            synth_prompt = SYSTEM_PROMPT_PERSIAN_GROUNDED

        if is_compound and not is_both:
            synth_prompt = f"{synth_prompt}\n\n{PROMPT_COMPOUND_DIRECTIVE_FA}"
        if hist_enabled:
            from rag_core.generation.prompts import PROMPT_INCREMENTAL_SUMMARY_DIRECTIVE_FA
            synth_prompt = f"{synth_prompt}\n\n{PROMPT_INCREMENTAL_SUMMARY_DIRECTIVE_FA}"

        # 6. Streaming Token Synthesis with TTFT Tracking & Clean Summary Interception
        await _notify("synthesis", "نگارش پاسخ مستند")
        t0 = time.perf_counter()
        synthesizer = self._resolve_synthesizer()
        first_token = True
        self.last_ttft_ms = None
        full_tokens = []
        in_summary = False
        summary_tokens = []
        pending_buffer = ""

        # Delimiters and prefixes indicating the start of the memory summary block
        delimiter_start_patterns = [
            "<<<MEMORY_SUMMARY_START>>>",
            "<!-- SUMMARY -->",
            "<!-- TURN_SUMMARY -->",
            "<!--SUMMARY-->",
            "دستورالعمل ثبت خلاصه",
        ]
        summary_regex = re.compile(
            r"<<<MEMORY_SUMMARY_START>>>|<!--\s*(?:TURN_)?SUMMARY\s*-->|دستورالعمل ثبت خلاصه",
            re.IGNORECASE
        )

        synth_history = None if (use_route or hist_chunks) else history
        async for token in synthesizer.synthesize_stream(
            query=effective_query,
            chunks=chunks_for_synth,
            system_prompt=synth_prompt,
            session_id=session_id,
            history=synth_history,
        ):
            if in_summary:
                summary_tokens.append(token)
                continue

            test_text = pending_buffer + token
            match = summary_regex.search(test_text)
            if match:
                # Text preceding the summary marker belongs to the actual answer
                prefix_clean = test_text[:match.start()]
                if prefix_clean:
                    if first_token:
                        self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
                        first_token = False
                    full_tokens.append(prefix_clean)
                    yield prefix_clean

                in_summary = True
                summary_tokens.append(test_text[match.start():])
                pending_buffer = ""
            else:
                # Check if test_text ends with a partial prefix of any delimiter marker
                split_idx = -1
                for d in delimiter_start_patterns:
                    max_k = min(len(test_text), len(d))
                    for k in range(max_k, 0, -1):
                        if test_text.endswith(d[:k]):
                            cand_idx = len(test_text) - k
                            if split_idx == -1 or cand_idx < split_idx:
                                split_idx = cand_idx
                if split_idx != -1 and (len(test_text) - split_idx) < 32:
                    # Partial match found at the tail; hold back the partial suffix
                    prefix_to_emit = test_text[:split_idx]
                    pending_buffer = test_text[split_idx:]
                    if prefix_to_emit:
                        if first_token:
                            self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
                            first_token = False
                        full_tokens.append(prefix_to_emit)
                        yield prefix_to_emit
                else:
                    to_emit = test_text
                    pending_buffer = ""
                    if to_emit:
                        if first_token:
                            self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
                            first_token = False
                        full_tokens.append(to_emit)
                        yield to_emit

        # Flush any trailing pending buffer if summary marker never materialized
        if pending_buffer and not in_summary:
            if first_token:
                self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
                first_token = False
            full_tokens.append(pending_buffer)
            yield pending_buffer

        latencies["synthesis_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        accumulated_answer = "".join(full_tokens).strip()
        summary_raw = "".join(summary_tokens)

        # Parse and sanitize answer and summary using TurnMemoryPayload
        raw_combined = f"{accumulated_answer}\n{summary_raw}" if summary_raw else accumulated_answer
        from rag_core.generation.synthesizer import ResponseSynthesizer
        try:
            clean_ans, clean_summary = ResponseSynthesizer.parse_answer_and_summary(raw_combined)
        except Exception:
            clean_ans, clean_summary = accumulated_answer, None
        if clean_ans:
            accumulated_answer = clean_ans

        from rag_core.generation.synthesizer import sanitize_math_markup
        accumulated_answer = sanitize_math_markup(accumulated_answer)

        is_refusal = (
            "اطلاعات کافی برای پاسخ" in accumulated_answer
            or "اطلاعاتی درباره این پرسش در کتاب یافت نشد" in accumulated_answer
            or "خارج از چارچوب کتاب" in accumulated_answer
        )
        if citation_bundle.inline_text and not is_refusal:
            yield f"\n\n---\n\n**Sources:**\n{citation_bundle.inline_text}"

        if not clean_summary and not is_refusal and accumulated_answer:
            sentences = [s.strip() for s in re.split(r"[.!?\n]+", accumulated_answer) if s.strip()]
            clean_summary = ". ".join(sentences[:2]) if sentences else accumulated_answer[:250].strip()
            if len(clean_summary) > 250:
                clean_summary = clean_summary[:250].strip()

        self.last_summary = clean_summary
        self.last_citations = citation_bundle.citations
        self.last_is_refusal = is_refusal
        self.last_effective_query = effective_query
        self.last_full_prompt = getattr(synthesizer, "last_full_prompt", "")
        if session_id:
            self._session_citations[session_id] = citation_bundle.citations
            self._session_is_refusal[session_id] = is_refusal
            self._session_effective_query[session_id] = effective_query
            self._session_latencies[session_id] = latencies
            self._session_summaries[session_id] = clean_summary
            self._session_full_prompts[session_id] = self.last_full_prompt



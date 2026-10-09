#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Compiled RAG Pipeline Graph.
Assembles the complete end-to-end RAG state graph with conditional branches,
native Langfuse tracing, latency profiling, and token/cost attribution.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from rag_core.graph.state import RAGGraphState
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

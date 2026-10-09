#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Nodes for RAG Pipeline.
Defines modular, composable node functions executing state transitions
with per-node latency profiling, token/cost estimation, and Langfuse attribution.
"""

import time
from typing import Dict, Any, Optional

from rag_core.graph.state import RAGGraphState
from rag_core.query.jev_router import JevRouter
from rag_core.fusion.jev_reranker import JevReranker
from rag_core.evaluation.jev_judge import JevJudge
from rag_core.evaluation.code_gates import (
    ADVERSARIAL_REFUSAL_PAYLOAD,
    is_standardized_refusal
)


def create_guard_and_route_node(router: Optional[JevRouter] = None):
    """Factory creating the guard_and_route LangGraph node."""
    _router = router or JevRouter()

    def guard_and_route(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("user_query", "")
        decision = _router.route_and_guard(q)
        lat = time.perf_counter() - t0

        latencies = state.get("latencies", {})
        latencies["guard_and_route"] = round(lat, 4)

        return {
            "target_domain": decision.destination,
            "is_adversarial": decision.is_adversarial,
            "adversarial_prob": decision.adversarial_prob,
            "routing_confidence": decision.confidence,
            "latencies": latencies
        }

    return guard_and_route


def refusal_node(state: RAGGraphState) -> Dict[str, Any]:
    """
    Terminal node executed when an adversarial attack or ungrounded query is detected.
    Returns deterministic standardized refusal message (FR-034, SC-012) in 0 ms.
    """
    t0 = time.perf_counter()
    refusal_msg = ADVERSARIAL_REFUSAL_PAYLOAD["response"]

    latencies = state.get("latencies", {})
    latencies["refusal"] = round(time.perf_counter() - t0, 4)

    return {
        "agent_response": refusal_msg,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retrieved_contexts": [],
        "sources": [],
        "latencies": latencies,
        "overall_verdict": 1  # Successfully gated
    }


def create_retrieve_node(retriever: Any, top_k: int = 5):
    """Factory creating the retrieval node using dense Qdrant or hybrid retriever."""
    def retrieve(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("user_query", "")
        retrieved = retriever.retrieve(q, top_k=top_k)
        lat = time.perf_counter() - t0

        latencies = state.get("latencies", {})
        latencies["retrieval"] = round(lat, 4)

        return {
            "retrieved_chunks": retrieved,
            "latencies": latencies
        }
    return retrieve


def create_rerank_node(reranker: Optional[JevReranker] = None, enable_reranking: bool = False, top_n: int = 3):
    """Factory creating the post-retrieval reranking node."""
    _reranker = reranker if enable_reranking else None

    def rerank(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("user_query", "")
        raw_candidates = state.get("retrieved_chunks", [])

        if _reranker and raw_candidates:
            reranked_objs = _reranker.rerank(q, raw_candidates, top_n=top_n)
            # Reconstruct (chunk, score) tuples from reranked objects
            reranked = [(r.chunk, r.jev_relevance) for r in reranked_objs]
        else:
            reranked = raw_candidates[:top_n]

        lat = time.perf_counter() - t0
        latencies = state.get("latencies", {})
        latencies["reranking"] = round(lat, 4)

        contexts = [chunk.text for chunk, _ in reranked]
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "source_name": chunk.source_name,
                "score": round(float(score), 4)
            }
            for chunk, score in reranked
        ]

        return {
            "reranked_chunks": reranked,
            "retrieved_contexts": contexts,
            "sources": sources,
            "latencies": latencies
        }
    return rerank


def create_synthesize_node(synthesizer: Any):
    """Factory creating the LLM generative synthesis node."""
    def synthesize(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("user_query", "")
        reranked = state.get("reranked_chunks", [])

        answer = synthesizer.synthesize(q, reranked)
        lat = time.perf_counter() - t0

        latencies = state.get("latencies", {})
        latencies["synthesis"] = round(lat, 4)

        # Estimate tokens and cost (OpenAI/Gemini pricing estimation)
        prompt_len = len(q) + sum(len(c.text) for c, _ in reranked)
        prompt_tokens = max(1, prompt_len // 4)
        completion_tokens = max(1, len(answer) // 4)
        total_tokens = prompt_tokens + completion_tokens

        # Approx $0.15 / 1M input tokens, $0.60 / 1M output tokens
        cost_usd = (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060)

        tokens = state.get("tokens", {})
        tokens["prompt_tokens"] = prompt_tokens
        tokens["completion_tokens"] = completion_tokens
        tokens["total_tokens"] = total_tokens

        return {
            "agent_response": answer,
            "latencies": latencies,
            "tokens": tokens,
            "cost_usd": round(cost_usd, 6)
        }
    return synthesize


def create_evaluate_node(judge: Optional[JevJudge] = None):
    """Factory creating the evaluation judge node for benchmark runs."""
    _judge = judge or JevJudge()

    def evaluate(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("user_query", "")
        ans = state.get("agent_response", "")
        gt = state.get("ground_truth", "")
        contexts = "\n\n".join(state.get("retrieved_contexts", []))
        suite = state.get("suite", "science_textbook")

        verdicts = _judge.evaluate_all_speculative(
            query=q,
            response=ans,
            context=contexts,
            ground_truth=gt or "",
            suite=suite
        )

        lat = time.perf_counter() - t0
        latencies = state.get("latencies", {})
        latencies["judge"] = round(lat, 4)

        f_pass = verdicts.get("faithfulness").verdict == 1 if "faithfulness" in verdicts else True
        r_pass = verdicts.get("answer_relevancy").verdict == 1 if "answer_relevancy" in verdicts else True
        overall = 1 if (f_pass and r_pass) else 0

        serializable_verdicts = {
            metric: {
                "verdict": v.verdict,
                "score": round(v.score, 4),
                "confidence": v.confidence,
                "latency_seconds": v.latency_seconds,
                "fallback_used": v.fallback_used
            }
            for metric, v in verdicts.items()
        }

        return {
            "evaluation_verdicts": serializable_verdicts,
            "overall_verdict": overall,
            "latencies": latencies
        }
    return evaluate

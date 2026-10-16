#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGGraphState: Strongly-typed state definition for LangGraph RAG workflows.
Complies with LangChain & LangGraph StateGraph standards and OpenInference telemetry.
"""

from typing import TypedDict, List, Dict, Any, Optional


class RAGGraphState(TypedDict, total=False):
    """
    Universal state tracked across all LangGraph nodes in the RAG execution graph.
    Supports both baseline multi-suite pipelines and enterprise science RAG pipelines.
    """
    user_query: str                          # Original user query
    effective_query: str                      # Disambiguated / rewritten query
    ground_truth: Optional[str]              # Optional ground truth for evaluation
    suite: Optional[str]                     # Benchmark suite name (science, faq, routing, refusal)
    session_id: Optional[str]                # Conversation session identifier
    history: Optional[List[Dict[str, str]]]  # Multi-turn conversation turns
    selected_chapters: Optional[List[int]]   # Mode 1 active chapter filter [1..15]
    selected_sources: Optional[List[str]]    # Explicit source selection (["science_textbook"], ["administrative_faq"], or ["all"])
    selected_faq_categories: Optional[List[str]] # Pre-filtered FAQ categories
    active_sources: Optional[List[str]]      # Active sources determined by L0 router or bypass
    level0_decision: Optional[Dict[str, Any]] # Serialized Level0RoutingDecision
    faq_dm1_decision: Optional[Dict[str, Any]] # FAQ DM1 decision metadata

    # Pre-Retrieval / Routing & Guardrail
    target_domain: str                       # Classified corpus ("science_textbook", "administrative_faq", "out_of_scope")
    is_adversarial: bool                     # Whether query attempts injection or out-of-scope bypass
    adversarial_prob: float                  # Calibrated adversarial probability from Jev
    routing_confidence: float                # Confidence score from Jev / keyword gate
    proceed_to_retrieval: bool               # Gated decision: whether query is in-scope
    rewrite_decision: Optional[Dict[str, Any]] # Full RewriteDecision metadata
    query_gate_decision: Optional[Dict[str, Any]] # Full QueryGateDecision metadata
    intent_tier: Optional[str]               # 3-tier intent: completely_irrelevant, partially_relevant, relevant
    context_need_tier: Optional[str]         # 4-tier need: needs_history_and_retrieve, needs_history_only, etc.
    selected_history_items: Optional[List[Dict[str, Any]]] # Historical items selected by Decision Model 2
    selected_context_turn_id: Optional[str]  # Manual single turn selected from UI
    q_and_a_summary: Optional[str]           # Turn summary generated during synthesis
    conversational_context_badge: Optional[str] # Persian badge for history-only responses
    sub_queries: List[str]                   # Atomic decomposed sub-queries for multi-hop

    # Retrieval & Reranking
    dense_candidates: List[Any]              # Raw Qdrant dense vector matches
    bm25_candidates: List[Any]               # Raw BM25 lexical matches
    fused_candidates: List[Any]              # Candidates blended via Reciprocal Rank Fusion
    retrieved_chunks: List[Any]              # Unified retrieved candidates
    reranked_chunks: List[Any]               # Chunks after cross-encoder neural reranking
    retrieved_contexts: List[str]            # Text bodies of retrieved/reranked chunks
    expanded_sections: List[Any]             # Deduplicated parent sections (max 3)
    context_text: str                        # Assembled prompt context string
    sources: List[Dict[str, Any]]            # Formatted source citations with chunk_id, score
    citations: List[Any]                     # PageCitation dataclass objects with #page=N

    # Generation
    agent_response: str                      # Synthesized response or deterministic refusal message
    is_refusal: bool                         # Whether response is a grounded refusal

    # Evaluation / Telemetry
    evaluation_verdicts: Dict[str, Any]      # Per-metric verdicts (faithfulness, relevancy, coverage, refusal)
    overall_verdict: int                     # 1 if all required metrics passed, 0 otherwise
    latencies: Dict[str, float]              # High-resolution per-node latencies in seconds or ms
    tokens: Dict[str, int]                   # Prompt tokens, completion tokens, total tokens
    cost_usd: float                          # Estimated or reported execution cost in USD
    tags: List[str]                          # Structured tags: ["baseline"] or ["enterprise", "hybrid"]
    metadata: Dict[str, Any]                 # Execution metadata, parameters, session IDs
    trace_id: Optional[str]                  # Langfuse trace ID


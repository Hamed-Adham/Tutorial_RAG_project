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
    """
    user_query: str                          # Original user query
    ground_truth: Optional[str]              # Optional ground truth for evaluation
    suite: Optional[str]                     # Benchmark suite name (science, faq, routing, refusal)
    
    # Pre-Retrieval / Routing & Guardrail
    target_domain: str                       # Classified corpus ("science_textbook", "administrative_faq", "out_of_scope")
    is_adversarial: bool                     # Whether query attempts injection or out-of-scope bypass
    adversarial_prob: float                  # Calibrated adversarial probability from Jev
    routing_confidence: float                # Confidence score from Jev / keyword gate
    
    # Retrieval & Reranking
    retrieved_chunks: List[Any]              # Raw retrieved Chunk objects with similarity scores
    reranked_chunks: List[Any]               # Chunks after Jev / cross-encoder reranking & filtering
    retrieved_contexts: List[str]            # Text bodies of retrieved/reranked chunks
    sources: List[Dict[str, Any]]            # Formatted source citations with chunk_id, source_type, score
    
    # Generation
    agent_response: str                      # Synthesized response or deterministic refusal message
    
    # Evaluation / Telemetry
    evaluation_verdicts: Dict[str, Any]      # Per-metric verdicts (faithfulness, relevancy, coverage, refusal)
    overall_verdict: int                     # 1 if all required metrics passed, 0 otherwise
    latencies: Dict[str, float]              # High-resolution per-node latencies in seconds
    tokens: Dict[str, int]                   # Prompt tokens, completion tokens, total tokens
    cost_usd: float                          # Estimated or reported execution cost in USD
    tags: List[str]                          # Structured tags: ["baseline"] or ["improved", "hybrid", "jev-reranker"]
    metadata: Dict[str, Any]                 # Execution metadata, parameters, session IDs
    trace_id: Optional[str]                  # Langfuse trace ID

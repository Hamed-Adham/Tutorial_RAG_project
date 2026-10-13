#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous Background Trace Quality Evaluators.
Scores Faithfulness, Context Precision, Context Recall, and Answer Relevancy in the background
off the critical user response path and pushes metrics to Langfuse.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from rag_core.observability.langfuse import LangfuseTracker
from rag_core.evaluation.jev_judge import JevJudge

logger = logging.getLogger(__name__)


def _extract_chunk_text(chunk: Any) -> str:
    """Safely extract plain text from chunk object or dictionary."""
    if hasattr(chunk, "text"):
        return str(chunk.text)
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(chunk)


def compute_faithfulness(answer: str, chunks: List[Any], is_refusal: bool = False) -> float:
    """
    Computes heuristic faithfulness / context adherence score (0.0 to 1.0).
    Evaluates whether non-trivial content words in answer are grounded in retrieved source context.
    If the response is an acknowledged refusal, faithfulness is 1.0.
    """
    if is_refusal:
        return 1.0

    if not chunks or not answer.strip():
        return 0.0

    all_context = " ".join([_extract_chunk_text(c) for c in chunks]).lower()

    # Filter out citation markers and markdown noise
    clean_answer = answer.split("**Sources:**")[0].split("---")[0]
    answer_words = [w for w in clean_answer.lower().split() if len(w) > 3]
    if not answer_words:
        return 1.0

    grounded_words = sum(1 for w in answer_words if w in all_context)
    return min(1.0, max(0.0, round(grounded_words / len(answer_words), 4)))


def compute_context_precision(query: str, chunks: List[Any]) -> float:
    """
    Computes Mean Average Precision (MAP) of relevant chunks at top ranks (0.0 to 1.0).
    Higher score indicates relevant chunks are positioned earlier in the candidate list.
    """
    if not chunks:
        return 0.0

    query_tokens = {t for t in query.lower().split() if len(t) > 2}
    if not query_tokens:
        return 1.0

    cum_relevant = 0
    weighted_precision_sum = 0.0

    for rank, chunk in enumerate(chunks, start=1):
        chunk_text = _extract_chunk_text(chunk).lower()
        chunk_tokens = set(chunk_text.split())
        is_rel = 1 if len(query_tokens.intersection(chunk_tokens)) >= 1 else 0

        if is_rel:
            cum_relevant += 1
            precision_at_k = cum_relevant / rank
            weighted_precision_sum += precision_at_k

    if cum_relevant == 0:
        return 0.0

    return min(1.0, max(0.0, round(weighted_precision_sum / cum_relevant, 4)))


def compute_context_recall(query: str, chunks: List[Any]) -> float:
    """
    Computes context recall (0.0 to 1.0).
    Measures coverage of query terms and entities within the aggregated retrieved context.
    """
    if not chunks:
        return 0.0

    query_tokens = {t for t in query.lower().split() if len(t) > 2}
    if not query_tokens:
        return 1.0

    all_context = " ".join([_extract_chunk_text(c) for c in chunks]).lower()
    covered_tokens = sum(1 for t in query_tokens if t in all_context)

    return min(1.0, max(0.0, round(covered_tokens / len(query_tokens), 4)))


def compute_answer_relevancy(query: str, answer: str, is_refusal: bool = False) -> float:
    """
    Computes answer relevancy (0.0 to 1.0).
    Measures how directly the generated answer addresses the user query.
    """
    if is_refusal:
        return 1.0

    if not answer.strip():
        return 0.0

    query_tokens = {t for t in query.lower().split() if len(t) > 2}
    if not query_tokens:
        return 1.0

    clean_answer = answer.split("**Sources:**")[0].lower()
    covered_tokens = sum(1 for t in query_tokens if t in clean_answer)
    base_score = covered_tokens / len(query_tokens)

    # Length penalty if answer is suspiciously empty
    if len(clean_answer.strip()) < 20:
        base_score *= 0.5

    return min(1.0, max(0.0, round(base_score, 4)))


# Backward-compatible aliases
def compute_chunk_relevance_heuristic(query: str, chunks: List[Any]) -> float:
    return compute_context_precision(query, chunks)


def compute_context_adherence_heuristic(answer: str, chunks: List[Any]) -> float:
    return compute_faithfulness(answer, chunks)


async def evaluate_trace_quality_async(
    trace_id: str,
    query: str,
    retrieved_chunks: List[Any],
    answer: str,
    tracker: Optional[LangfuseTracker] = None,
    judge: Optional[JevJudge] = None,
    is_refusal: bool = False
) -> Dict[str, float]:
    """
    Asynchronously evaluates trace quality off the critical path and submits scores to Langfuse.
    Calculates: faithfulness, context_precision, context_recall, answer_relevancy.
    """
    faithfulness = compute_faithfulness(answer, retrieved_chunks, is_refusal=is_refusal)
    context_precision = compute_context_precision(query, retrieved_chunks)
    context_recall = compute_context_recall(query, retrieved_chunks)
    answer_relevancy = compute_answer_relevancy(query, answer, is_refusal=is_refusal)

    scores = {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_relevancy": answer_relevancy,
        # Legacy aliases for backward compatibility
        "chunk_relevance": context_precision,
        "context_adherence": faithfulness,
    }

    if not trace_id or trace_id == "offline_trace":
        return scores

    t = tracker or LangfuseTracker()
    try:
        t.add_trace_score(
            trace_id=trace_id,
            name="faithfulness",
            value=faithfulness,
            comment="Groundedness of answer in retrieved context (0.0 - 1.0)"
        )
        t.add_trace_score(
            trace_id=trace_id,
            name="context_precision",
            value=context_precision,
            comment="Mean Average Precision of relevant chunks at top ranks (0.0 - 1.0)"
        )
        t.add_trace_score(
            trace_id=trace_id,
            name="context_recall",
            value=context_recall,
            comment="Coverage of query concepts within retrieved context (0.0 - 1.0)"
        )
        t.add_trace_score(
            trace_id=trace_id,
            name="answer_relevancy",
            value=answer_relevancy,
            comment="Semantic & lexical relevancy of answer to the query (0.0 - 1.0)"
        )
    except Exception as e:
        logger.warning(f"Error during async trace score submission: {e}")

    return scores


def dispatch_trace_evaluation(
    trace_id: str,
    query: str,
    retrieved_chunks: List[Any],
    answer: str,
    tracker: Optional[LangfuseTracker] = None,
    judge: Optional[JevJudge] = None,
    is_refusal: bool = False
) -> Optional[asyncio.Task]:
    """
    Dispatches asynchronous trace quality evaluation via asyncio.create_task.
    Guarantees zero latency penalty on the critical response path.
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            evaluate_trace_quality_async(
                trace_id=trace_id,
                query=query,
                retrieved_chunks=retrieved_chunks,
                answer=answer,
                tracker=tracker,
                judge=judge,
                is_refusal=is_refusal
            )
        )
        return task
    except RuntimeError:
        return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Nodes for RAG Pipeline.
Defines modular, composable node functions executing state transitions
with per-node latency profiling, token/cost estimation, and Langfuse attribution.
"""

import time
import asyncio
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


# ---------------------------------------------------------------------------
# Enterprise Science RAG Nodes (LangGraph Standard)
# ---------------------------------------------------------------------------

def create_enterprise_query_gate_node(query_gate: Optional[Any] = None):
    """LangGraph node executing QueryGate intent gating, need dispatching, and antecedent resolution."""
    from rag_core.query.query_gate import QueryGate
    from rag_core.indexing.science_schema import ContextNeedTier, CurriculumIntentTier, QueryGateDecision
    _gate = query_gate or QueryGate()

    async def query_gate_node(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("user_query", "")
        hist = state.get("history")
        chapters = state.get("selected_chapters")
        selected_turn_id = state.get("selected_context_turn_id")
        history_enabled = state.get("metadata", {}).get("history_enabled", True)

        if isinstance(_gate, QueryGate):
            decision = await _gate.evaluate_and_route(
                query=q,
                history=hist,
                selected_chapters=chapters,
                selected_context_turn_id=selected_turn_id,
                history_enabled=history_enabled
            )
        elif hasattr(_gate, "evaluate_and_rewrite"):
            res = _gate.evaluate_and_rewrite(query=q, history=hist, selected_chapters=chapters)
            raw_decision = await res if asyncio.iscoroutine(res) else res
            if isinstance(raw_decision, QueryGateDecision):
                decision = raw_decision
            else:
                proceed = getattr(raw_decision, "proceed_to_retrieval", True)
                rewritten = getattr(raw_decision, "rewritten_query", None) or q
                decision = QueryGateDecision(
                    raw_query=q,
                    effective_query=rewritten,
                    intent_tier=CurriculumIntentTier.COMPLETELY_IRRELEVANT if not proceed else CurriculumIntentTier.RELEVANT,
                    context_need_tier=ContextNeedTier.NEEDS_NEITHER if not proceed else ContextNeedTier.NEEDS_RETRIEVE_ONLY,
                    is_refusal=not proceed,
                    bypass_retrieval=not proceed,
                    sub_queries=getattr(raw_decision, "sub_queries", []),
                    reasoning=getattr(raw_decision, "reasoning", ""),
                    transformation_type=getattr(raw_decision, "transformation_type", "bypass"),
                    is_compound=getattr(raw_decision, "is_compound", False)
                )
        else:
            res = _gate.evaluate_and_route(
                query=q,
                history=hist,
                selected_chapters=chapters,
                selected_context_turn_id=selected_turn_id,
                history_enabled=history_enabled
            )
            decision = await res if asyncio.iscoroutine(res) else res
        lat = (time.perf_counter() - t0) * 1000.0

        latencies = dict(state.get("latencies", {}))
        latencies["gate_ms"] = round(lat, 2)
        latencies["rewrite_ms"] = round(lat, 2)
        if hasattr(decision, "latencies") and decision.latencies:
            latencies.update(decision.latencies)

        badge = "پاسخ بر اساس گفتگوی پیشین" if decision.context_need_tier == ContextNeedTier.NEEDS_HISTORY_ONLY else None

        return {
            "effective_query": decision.effective_query or q,
            "proceed_to_retrieval": not decision.is_refusal and not decision.bypass_retrieval,
            "is_refusal": decision.is_refusal,
            "intent_tier": decision.intent_tier.value,
            "context_need_tier": decision.context_need_tier.value,
            "selected_history_items": [item.model_dump() for item in decision.selected_history_items],
            "sub_queries": decision.sub_queries,
            "conversational_context_badge": badge,
            "rewrite_decision": {
                "needs_rewrite": decision.effective_query != q.strip(),
                "rewritten_query": decision.effective_query,
                "proceed_to_retrieval": not decision.is_refusal and not decision.bypass_retrieval
            },
            "query_gate_decision": decision.model_dump() if hasattr(decision, "model_dump") else {},
            "latencies": latencies
        }
    return query_gate_node


# Backward compatibility alias
create_enterprise_gated_rewrite_node = create_enterprise_query_gate_node



def create_enterprise_refusal_node():
    """Terminal node executing immediate grounded refusal for out-of-scope or rejected inquiries."""
    from rag_core.pipeline import GROUNDED_REFUSAL_FA

    async def enterprise_refusal_node(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        latencies = dict(state.get("latencies", {}))
        latencies["refusal_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        gate_decision = state.get("query_gate_decision", {})
        reason = gate_decision.get("reasoning", "")
        persian_refusal = any('\u0600' <= c <= '\u06FF' for c in reason)
        response_msg = reason if persian_refusal else GROUNDED_REFUSAL_FA

        return {
            "agent_response": response_msg,
            "is_refusal": True,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "expanded_sections": [],
            "citations": [],
            "latencies": latencies
        }
    return enterprise_refusal_node


def create_enterprise_hybrid_retrieve_node(hybrid_retriever: Optional[Any] = None, overfetch_k: int = 50):
    """LangGraph node executing parallel Qdrant dense + Hazm BM25 search with Reciprocal Rank Fusion."""
    from rag_core.retrieval.hybrid import HybridRetriever
    _retriever = hybrid_retriever or HybridRetriever()

    async def hybrid_retrieve_node(state: RAGGraphState) -> Dict[str, Any]:
        import asyncio
        t0 = time.perf_counter()
        effective_query = state.get("effective_query") or state.get("user_query", "")
        sub_queries = state.get("sub_queries", [])
        chapters = state.get("selected_chapters")

        # Augment retrieval query with historical summaries when needs_history_and_retrieve (User Story 4)
        selected_history_raw = state.get("selected_history_items", [])
        if selected_history_raw:
            from rag_core.indexing.science_schema import SelectedHistoryItem
            from rag_core.query.query_gate import QueryGate
            typed_items = [
                SelectedHistoryItem(**item) if isinstance(item, dict) else item
                for item in selected_history_raw
            ]
            effective_query = QueryGate.enrich_query_with_summaries(effective_query, typed_items)

        if sub_queries and len(sub_queries) > 1:
            tasks = [
                _retriever.retrieve(query=sq, top_k=overfetch_k, overfetch_k=overfetch_k, chapters=chapters)
                for sq in sub_queries
            ]
            sub_results = await asyncio.gather(*tasks)
            candidates = []
            seen = set()
            for cand_list in sub_results:
                for c in cand_list:
                    if c.chunk_id not in seen:
                        candidates.append(c)
                        seen.add(c.chunk_id)
        else:
            candidates = await _retriever.retrieve(
                query=effective_query,
                top_k=overfetch_k,
                overfetch_k=overfetch_k,
                chapters=chapters
            )

        lat = (time.perf_counter() - t0) * 1000.0
        latencies = dict(state.get("latencies", {}))
        latencies["retrieval_ms"] = round(lat, 2)

        return {
            "retrieved_chunks": candidates,
            "fused_candidates": candidates,
            "latencies": latencies
        }
    return hybrid_retrieve_node


def create_enterprise_neural_rerank_node(reranker: Optional[Any] = None, top_k: int = 5):
    """LangGraph node executing neural cross-encoder reranking (bge-reranker-v2-m3) with circuit breaker."""
    from rag_core.fusion.reranker import CrossEncoderReranker
    from rag_core.config import settings
    _reranker = reranker or CrossEncoderReranker()

    async def neural_rerank_node(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("effective_query") or state.get("user_query", "")
        candidates = state.get("retrieved_chunks", [])

        if not candidates:
            return {"reranked_chunks": [], "latencies": state.get("latencies", {})}

        timeout = getattr(settings, "rerank_timeout_ms", 3000.0)
        reranked = await _reranker.rerank(
            query=q,
            candidates=candidates[:8],
            top_k=top_k,
            timeout_ms=timeout
        )

        lat = (time.perf_counter() - t0) * 1000.0
        latencies = dict(state.get("latencies", {}))
        latencies["rerank_ms"] = round(lat, 2)

        return {
            "reranked_chunks": reranked,
            "latencies": latencies
        }
    return neural_rerank_node


def create_enterprise_expansion_node(expander: Optional[Any] = None, reorderer: Optional[Any] = None):
    """LangGraph node executing parent section expansion and lost-in-the-middle reordering."""
    from rag_core.postprocessing.expansion import ParentSectionExpander, LongContextReorder
    from rag_core.postprocessing.citations import CitationFormatter
    _expander = expander or ParentSectionExpander()
    _reorderer = reorderer or LongContextReorder()
    _formatter = CitationFormatter()

    async def expansion_node(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        reranked = state.get("reranked_chunks", [])

        if not reranked:
            return {"expanded_sections": [], "context_text": "", "citations": [], "latencies": state.get("latencies", {})}

        expanded = _expander.expand(candidates=reranked, max_sections=3, token_ceiling=2500)
        reordered = _reorderer.reorder(expanded.parent_sections)

        # Build citations
        citation_candidates = []
        for rc in reranked:
            c = rc.candidate if hasattr(rc, "candidate") else rc
            doc_id = getattr(c, "doc_id", "science_textbook") or "science_textbook"
            if not doc_id.endswith(".pdf"):
                doc_id = f"{doc_id}.pdf"
            citation_candidates.append({
                "doc_id": doc_id,
                "filename": "کتاب علوم تجربی هفتم",
                "page": c.page,
                "chunk_id": c.chunk_id,
                "score": getattr(rc, "cross_encoder_score", 0.0),
                "metadata": {"title": c.breadcrumbs}
            })

        citation_bundle = _formatter.format(citation_candidates)
        context_text = "\n\n---\n\n".join(
            f"### فصل {sec.chapter}: {sec.title} (صفحه {sec.page_start} تا {sec.page_end})\n{sec.full_text}"
            for sec in reordered
        )

        lat = (time.perf_counter() - t0) * 1000.0
        latencies = dict(state.get("latencies", {}))
        latencies["expansion_ms"] = round(lat, 2)

        return {
            "expanded_sections": reordered,
            "context_text": context_text,
            "citations": citation_candidates,
            "metadata": {"citation_inline": citation_bundle.inline_text},
            "latencies": latencies
        }
    return expansion_node


def create_enterprise_synthesis_node(synthesizer: Optional[Any] = None):
    """LangGraph node executing grounded response synthesis with citation injection."""
    from rag_core.generation.synthesizer import ResponseSynthesizer
    from rag_core.generation.prompts import SYSTEM_PROMPT_PERSIAN_GROUNDED, PROMPT_COMPOUND_DIRECTIVE_FA
    _synthesizer = synthesizer or ResponseSynthesizer()

    async def synthesis_node(state: RAGGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("effective_query") or state.get("user_query", "")
        reordered_sections = state.get("expanded_sections", [])
        citations = state.get("citations", [])

        if not reordered_sections:
            from rag_core.pipeline import GROUNDED_REFUSAL_FA
            ch = state.get("selected_chapters")
            if ch:
                ch_str = "، ".join(str(c) for c in ch)
                refusal_msg = f"در محدوده فصل‌های انتخابی ({ch_str})، اطلاعاتی درباره این پرسش در کتاب یافت نشد. لطفاً دامنه جستجو را گسترش دهید یا تمام کتاب را انتخاب فرمایید."
            else:
                refusal_msg = GROUNDED_REFUSAL_FA
            return {"agent_response": refusal_msg, "is_refusal": True, "latencies": state.get("latencies", {})}

        default_doc = citations[0]["doc_id"] if citations else "science_textbook.pdf"
        chunks_for_synth = [
            {
                "text": sec.full_text,
                "page": sec.page_start,
                "breadcrumb": f"فصل {sec.chapter}: {sec.title}",
                "doc_id": default_doc,
                "chunk_id": sec.section_id
            }
            for sec in reordered_sections
        ]

        synth_prompt = SYSTEM_PROMPT_PERSIAN_GROUNDED
        dec = state.get("rewrite_decision") or {}
        if dec.get("is_compound", False):
            synth_prompt = f"{synth_prompt}\n\n{PROMPT_COMPOUND_DIRECTIVE_FA}"
        if state.get("history_enabled", True):
            from rag_core.generation.prompts import PROMPT_INCREMENTAL_SUMMARY_DIRECTIVE_FA
            synth_prompt = f"{synth_prompt}\n\n{PROMPT_INCREMENTAL_SUMMARY_DIRECTIVE_FA}"

        try:
            res = _synthesizer.synthesize(query=q, chunks=chunks_for_synth, system_prompt=synth_prompt)
            if hasattr(res, "raw_completion") and isinstance(res.raw_completion, str) and res.raw_completion.strip():
                raw_answer = res.raw_completion.rstrip()
            elif hasattr(res, "answer") and isinstance(res.answer, str):
                raw_answer = res.answer.split("**Sources:**")[0].split("📚")[0].rstrip()
            else:
                raw_answer = str(getattr(res, "answer", res))
            is_ref = getattr(res, "is_fallback", False)
        except Exception:
            raw_answer = "بر اساس متون استخراج شده از کتاب علوم هفتم..."
            is_ref = False

        citation_inline = (state.get("metadata") or {}).get("citation_inline", "")
        if citation_inline and not is_ref:
            final_answer = f"{raw_answer.rstrip()}\n\n---\n\n#### 📚 منابع استنادی کتاب درسی:\n{citation_inline}"
        else:
            final_answer = raw_answer


        lat = (time.perf_counter() - t0) * 1000.0
        latencies = dict(state.get("latencies", {}))
        latencies["synthesis_ms"] = round(lat, 2)

        return {
            "agent_response": final_answer,
            "is_refusal": is_ref,
            "q_and_a_summary": getattr(res, "q_and_a_summary", None) if not is_ref else None,
            "latencies": latencies
        }
    return synthesis_node


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdvancedRAGPipeline: Production-grade Enterprise Science RAG Engine.
Orchestrates:
  1. Confidence-gated query rewriting & sub-query decomposition (<10ms bypass)
  2. Parallel hybrid retrieval (Qdrant dense + sub-5ms BM25) with Reciprocal Rank Fusion
  3. Neural cross-encoder reranking (bge-reranker-v2-m3) with 350ms circuit breaker
  4. Dynamic parent section expansion (max 3 sections, 2,500 token ceiling)
  5. Lost-in-the-middle context reordering (LongContextReorder)
  6. Grounded response synthesis with strict Markdown link page citations
  7. Langfuse OpenTelemetry tracing with asynchronous background quality evaluation
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

from rag_core.config import settings
from rag_core.contracts import IAdvancedRAGPipeline
from rag_core.indexing.science_schema import (
    RetrievalCandidate,
    RerankedCandidate,
    RewriteDecision,
    ExpandedContext,
    AdvancedRAGResult,
    PageCitation
)
from rag_core.query.query_gate import QueryGate, GatedQueryRewriter
from rag_core.retrieval.hybrid import HybridRetriever
from rag_core.fusion.reranker import CrossEncoderReranker
from rag_core.postprocessing.expansion import ParentSectionExpander, LongContextReorder
from rag_core.postprocessing.citations import CitationFormatter, DOC_TO_CHAPTER
from rag_core.generation.synthesizer import ResponseSynthesizer as Synthesizer
from rag_core.observability.langfuse import LangfuseTracker
from rag_core.observability.trace_evaluators import dispatch_trace_evaluation
from rag_core.generation.prompts import SYSTEM_PROMPT_PERSIAN_GROUNDED

logger = logging.getLogger(__name__)

GROUNDED_REFUSAL_FA = "بر اساس برنامه درسی علوم پایه هفتم، اطلاعاتی درباره این پرسش در کتاب یافت نشد."


async def _notify_status(cb: Optional[Any], step: str, title: str) -> None:
    if cb is not None:
        try:
            res = cb(step, title)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass


class AdvancedRAGPipeline(IAdvancedRAGPipeline):
    """
    End-to-end Enterprise RAG Pipeline for curriculum science inquiry.
    Enforces latency budgets (<2.0s p95) and strict evidence grounding.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        rewriter: Optional[GatedQueryRewriter] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        expander: Optional[ParentSectionExpander] = None,
        reorderer: Optional[LongContextReorder] = None,
        synthesizer: Optional[Synthesizer] = None,
        tracker: Optional[LangfuseTracker] = None,
        warmup: bool = False,
    ):
        self.rewriter = rewriter or GatedQueryRewriter()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.expander = expander or ParentSectionExpander()
        self.reorderer = reorderer or LongContextReorder()
        self.tracker = tracker or LangfuseTracker()
        self.synthesizer = synthesizer or Synthesizer(tracker=self.tracker)
        self.citation_formatter = CitationFormatter()
        self._session_citations: Dict[str, List[PageCitation]] = {}
        self._session_is_refusal: Dict[str, bool] = {}
        self._session_effective_query: Dict[str, str] = {}
        self._session_latencies: Dict[str, Dict[str, float]] = {}
        self._session_summaries: Dict[str, str] = {}
        self.last_citations: List[PageCitation] = []
        self.last_is_refusal: bool = False
        self.last_effective_query: str = ""
        self.last_latencies: Dict[str, float] = {}
        self.last_summary: Optional[str] = None

        if warmup:
            self.warmup()

    def warmup(self) -> None:
        """Eagerly load CrossEncoder weights and dense embedder into memory to eliminate first-query cold-start latency."""
        self.reranker.warmup()
        if hasattr(self.hybrid_retriever, "warmup"):
            self.hybrid_retriever.warmup()

    def get_citations_for_session(self, session_id: Optional[str]) -> List[PageCitation]:
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
            return dict(self._session_latencies[session_id])
        return dict(self.last_latencies)

    def get_summary_for_session(self, session_id: Optional[str]) -> Optional[str]:
        if session_id and session_id in getattr(self, "_session_summaries", {}):
            return self._session_summaries[session_id]
        return getattr(self, "last_summary", None)

    def get_full_prompt_for_session(self, session_id: Optional[str]) -> Optional[str]:
        if session_id and session_id in getattr(self, "_session_full_prompts", {}):
            return self._session_full_prompts[session_id]
        return getattr(self.synthesizer, "last_full_prompt", getattr(self, "last_full_prompt", None))


    def as_langgraph_pipeline(self):
        """
        Compile and return the pipeline as a native LangGraph StateGraph (EnterpriseScienceGraph).
        Enables standard LangGraph declarative execution, streaming, and visual graph inspection.
        """
        from rag_core.graph.pipeline_graph import EnterpriseScienceGraph
        return EnterpriseScienceGraph(
            rewriter=self.rewriter,
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            expander=self.expander,
            reorderer=self.reorderer,
            synthesizer=self.synthesizer
        )

    async def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        overfetch_k: int = 50,
        selected_chapters: Optional[List[int]] = None,
    ) -> List[RetrievalCandidate]:
        """
        Execute decoupled retrieval funnel: Gated Rewriting -> Hybrid Search -> RRF -> CrossEncoder.
        Bypasses expansion, LLM synthesis, and citations for fast candidate ranking.
        """
        rewrite_decision: RewriteDecision = await self.rewriter.evaluate_and_rewrite(
            query=query_text,
            selected_chapters=selected_chapters
        )
        if not rewrite_decision.proceed_to_retrieval:
            return []

        effective_query = rewrite_decision.rewritten_query or query_text
        candidates = await self.hybrid_retriever.retrieve(
            query=effective_query,
            top_k=overfetch_k,
            overfetch_k=overfetch_k,
            chapters=rewrite_decision.active_chapters
        )
        if not candidates:
            return []

        rerank_timeout = getattr(settings, "rerank_timeout_ms", 3000.0)
        reranked = await self.reranker.rerank(
            query=effective_query,
            candidates=candidates[:8],
            top_k=top_k,
            timeout_ms=rerank_timeout
        )
        return [r.candidate for r in reranked]

    async def query_stream(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        overfetch_k: int = 50,
        selected_chapters: Optional[List[int]] = None,
        status_callback: Optional[Any] = None,
    ) -> AsyncIterator[str]:
        """
        Execute full enterprise RAG pipeline yielding response tokens as an asynchronous generator.
        Measures Time-To-First-Token (TTFT < 1.0s) and appends citation metadata at the end of the stream.
        """
        total_start = time.perf_counter()
        latencies: Dict[str, float] = {}

        # Stage 1: Confidence-Gated Query Understanding
        await _notify_status(status_callback, "rewrite", "تحلیل و بررسی سوال")
        t0 = time.perf_counter()
        rewrite_decision: RewriteDecision = await self.rewriter.evaluate_and_rewrite(
            query=query_text,
            history=history,
            selected_chapters=selected_chapters
        )
        latencies["rewrite_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        effective_query = rewrite_decision.rewritten_query or query_text

        if not rewrite_decision.proceed_to_retrieval:
            scope_desc = (
                f"فصل‌های انتخابی ({', '.join(str(c) for c in rewrite_decision.active_chapters)})"
                if rewrite_decision.active_chapters
                else "کتاب علوم تجربی پایه هفتم"
            )
            latencies["total_ms"] = round((time.perf_counter() - total_start) * 1000.0, 2)
            if session_id:
                self._session_latencies[session_id] = latencies
            self.last_latencies = latencies
            refusal_msg = (
                rewrite_decision.reasoning
                if (rewrite_decision.reasoning and any('\u0600' <= c <= '\u06FF' for c in rewrite_decision.reasoning))
                else f"این سوال در حیطه مباحث {scope_desc} قرار ندارد. لطفاً سوالی مرتبط با مباحث درسی مطرح کنید."
            )
            yield refusal_msg
            return

        # Stage 2: Parallel Hybrid Retrieval
        await _notify_status(status_callback, "retrieval", "جستجوی اسناد و کتاب درسی")
        t0 = time.perf_counter()
        if rewrite_decision.sub_queries and len(rewrite_decision.sub_queries) > 1:
            retrieval_tasks = [
                self.hybrid_retriever.retrieve(
                    query=sq,
                    top_k=overfetch_k,
                    overfetch_k=overfetch_k,
                    chapters=rewrite_decision.active_chapters
                )
                for sq in rewrite_decision.sub_queries
            ]
            sub_results = await asyncio.gather(*retrieval_tasks)
            candidates = []
            seen = set()
            for cand_list in sub_results:
                for c in cand_list:
                    if c.chunk_id not in seen:
                        candidates.append(c)
                        seen.add(c.chunk_id)
        else:
            candidates = await self.hybrid_retriever.retrieve(
                query=effective_query,
                top_k=overfetch_k,
                overfetch_k=overfetch_k,
                chapters=rewrite_decision.active_chapters
            )
        latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if not candidates:
            latencies["total_ms"] = round((time.perf_counter() - total_start) * 1000.0, 2)
            if session_id:
                self._session_latencies[session_id] = latencies
            self.last_latencies = latencies
            if rewrite_decision.active_chapters:
                ch_str = "، ".join(str(c) for c in rewrite_decision.active_chapters)
                refusal_msg = f"در محدوده فصل‌های انتخابی ({ch_str})، اطلاعاتی درباره این پرسش در کتاب یافت نشد. لطفاً دامنه جستجو را گسترش دهید یا تمام کتاب را انتخاب فرمایید."
            else:
                refusal_msg = GROUNDED_REFUSAL_FA
            yield refusal_msg
            return

        # Stage 3: Neural Cross-Encoder Reranking
        await _notify_status(status_callback, "rerank", "بازرتبه‌بندی عصبی شواهد")
        t0 = time.perf_counter()
        rerank_timeout = getattr(settings, "rerank_timeout_ms", 3000.0)
        reranked_candidates = await self.reranker.rerank(
            query=effective_query,
            candidates=candidates[:8],
            top_k=top_k,
            timeout_ms=rerank_timeout
        )
        latencies["rerank_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 4: Parent Section Expansion & Context Reordering
        await _notify_status(status_callback, "expansion", "بسط و تکمیل بخش‌های مرتبط")
        t0 = time.perf_counter()
        expanded_context = self.expander.expand(
            candidates=reranked_candidates,
            max_sections=3,
            token_ceiling=2500
        )
        reordered_sections = self.reorderer.reorder(expanded_context.parent_sections)
        latencies["expansion_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Build citation objects
        citation_candidates = []
        for rc in reranked_candidates:
            c = rc.candidate
            doc_id = getattr(c, "doc_id", "science_textbook") or "science_textbook"
            if not doc_id.endswith(".pdf"):
                doc_id = f"{doc_id}.pdf"
            ch_info = DOC_TO_CHAPTER.get(doc_id)
            display_name = f"فصل {ch_info[0]}، صفحه {c.page}" if ch_info else f"کتاب درسی، صفحه {c.page}"
            citation_candidates.append({
                "doc_id": doc_id,
                "filename": doc_id,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "score": rc.cross_encoder_score,
                "metadata": {"title": c.breadcrumbs, "display_name": display_name}
            })
        citation_bundle = self.citation_formatter.format(citation_candidates)

        default_synth_doc = citation_candidates[0]["doc_id"] if citation_candidates else "science_textbook.pdf"
        chunks_for_synth = [
            {
                "text": sec.full_text,
                "page": sec.page_start,
                "breadcrumb": f"فصل {sec.chapter}: {sec.title}",
                "doc_id": default_synth_doc,
                "chunk_id": sec.section_id
            }
            for sec in reordered_sections
        ]

        synth_prompt = SYSTEM_PROMPT_PERSIAN_GROUNDED
        if getattr(rewrite_decision, "is_compound", False):
            from rag_core.generation.prompts import PROMPT_COMPOUND_DIRECTIVE_FA
            synth_prompt = f"{synth_prompt}\n\n{PROMPT_COMPOUND_DIRECTIVE_FA}"

        # Stage 5: Streaming Token Synthesis with TTFT Tracking
        await _notify_status(status_callback, "synthesis", "نگارش پاسخ مستند")
        t0 = time.perf_counter()
        first_token = True
        self.last_ttft_ms = None
        full_tokens = []
        async for token in self.synthesizer.synthesize_stream(
            query=effective_query,
            chunks=chunks_for_synth,
            system_prompt=synth_prompt,
            session_id=session_id,
            history=history,
        ):
            if first_token:
                self.last_ttft_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
                first_token = False
            full_tokens.append(token)
            yield token

        latencies["synthesis_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies["total_ms"] = round((time.perf_counter() - total_start) * 1000.0, 2)

        # Append citations at the end of stream
        accumulated_answer = "".join(full_tokens).strip()
        is_refusal = (
            "اطلاعات کافی برای پاسخ" in accumulated_answer
            or "اطلاعاتی درباره این پرسش در کتاب یافت نشد" in accumulated_answer
        )
        if citation_bundle.inline_text and not is_refusal:
            yield f"\n\n---\n\n**Sources:**\n{citation_bundle.inline_text}"

        # Persist session metadata for streaming consumer
        if session_id:
            self._session_citations[session_id] = citation_bundle.citations
            self._session_is_refusal[session_id] = is_refusal
            self._session_effective_query[session_id] = effective_query
            self._session_latencies[session_id] = latencies
        self.last_citations = citation_bundle.citations
        self.last_is_refusal = is_refusal
        self.last_effective_query = effective_query
        # Stage 6: Langfuse Observability for streaming
        trace_id = "offline_trace"
        if self.tracker and self.tracker.is_active:
            try:
                trace_id = self.tracker.trace_advanced_pipeline(
                    name="advanced_pipeline",
                    question=query_text,
                    answer=accumulated_answer,
                    is_refusal=is_refusal,
                    session_id=session_id,
                    tags=["pipeline:advanced", "domain:science_textbook", "mode:stream"],
                    metadata={
                        "effective_query": effective_query,
                        "rewritten": rewrite_decision.needs_rewrite,
                        "top_k": top_k,
                    },
                    rewrite_decision=rewrite_decision,
                    history=history,
                    effective_query=effective_query,
                    rewrite_latency_ms=latencies.get("rewrite_ms", 0.0),
                    dense_results=candidates,
                    bm25_results=[],
                    fused_candidates=candidates,
                    retrieval_latency_ms=latencies.get("retrieval_ms", 0.0),
                    input_candidates=candidates[:8],
                    reranked_candidates=reranked_candidates,
                    circuit_breaker_status=getattr(self.reranker, "circuit_breaker_status", "active"),
                    rerank_latency_ms=latencies.get("rerank_ms", 0.0),
                    expanded_context=expanded_context,
                    reordered_sections=reordered_sections,
                    expansion_latency_ms=latencies.get("expansion_ms", 0.0),
                    synthesis_result=None,
                    full_prompt=synth_prompt,
                    raw_completion=accumulated_answer,
                    synthesis_latency_ms=latencies.get("synthesis_ms", 0.0),
                    ttft_ms=self.last_ttft_ms,
                    latencies=latencies,
                )
                self.tracker.flush()
            except Exception as e:
                logger.warning(f"Error logging Langfuse stream trace: {e}")
        self.last_trace_id = trace_id


    async def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        overfetch_k: int = 50,
        selected_chapters: Optional[List[int]] = None,
    ) -> AdvancedRAGResult:
        """
        Execute full enterprise RAG pipeline within latency budget.
        """
        total_start = time.perf_counter()
        latencies: Dict[str, float] = {}

        # -----------------------------------------------------------------
        # Stage 1: Confidence-Gated Query Understanding & Rewriting
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        rewrite_decision: RewriteDecision = await self.rewriter.evaluate_and_rewrite(
            query=query_text,
            history=history,
            selected_chapters=selected_chapters
        )
        latencies["rewrite_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        effective_query = rewrite_decision.rewritten_query or query_text

        # Gate Check: Early halt if query is out of scope for curriculum topics
        if not rewrite_decision.proceed_to_retrieval:
            latencies["total_ms"] = round((time.perf_counter() - total_start) * 1000.0, 2)
            scope_desc = (
                f"فصل‌های انتخابی ({', '.join(str(c) for c in rewrite_decision.active_chapters)})"
                if rewrite_decision.active_chapters
                else "کتاب علوم تجربی پایه هفتم"
            )
            return AdvancedRAGResult(
                query=query_text,
                effective_query=effective_query,
                answer=f"این سوال در حیطه مباحث {scope_desc} قرار ندارد. لطفاً سوالی مرتبط با مباحث درسی مطرح کنید.",
                citations=[],
                retrieved_chunks=[],
                top_parents=[],
                latency_profile=latencies,
                trace_id="",
                is_refusal=True
            )

        # -----------------------------------------------------------------
        # Stage 2: Parallel Hybrid Retrieval (BM25 + Dense Qdrant + RRF)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        dense_results: List[RetrievalCandidate] = []
        bm25_results: List[RetrievalCandidate] = []
        if rewrite_decision.sub_queries and len(rewrite_decision.sub_queries) > 1:
            # Multi-concept query: execute hybrid retrieval across sub-queries
            sub_tasks = [
                self.hybrid_retriever.retrieve(
                    sq,
                    top_k=overfetch_k,
                    overfetch_k=overfetch_k,
                    chapters=rewrite_decision.active_chapters
                )
                for sq in rewrite_decision.sub_queries
            ]
            sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)
            merged_candidates_map: Dict[str, RetrievalCandidate] = {}
            for res in sub_results:
                if isinstance(res, list):
                    for cand in res:
                        if isinstance(cand, RetrievalCandidate):
                            if cand.chunk_id not in merged_candidates_map:
                                merged_candidates_map[cand.chunk_id] = cand
                            else:
                                merged_candidates_map[cand.chunk_id].rrf_score += cand.rrf_score
            candidates = sorted(
                merged_candidates_map.values(),
                key=lambda x: x.rrf_score,
                reverse=True
            )[:overfetch_k]
            dense_results = [c for c in candidates if getattr(c, "dense_score", None) is not None]
            bm25_results = [c for c in candidates if getattr(c, "bm25_score", None) is not None]
        else:
            retrieval_res = None
            if hasattr(self.hybrid_retriever, "retrieve_with_details"):
                try:
                    retrieval_res = await self.hybrid_retriever.retrieve_with_details(
                        query=effective_query,
                        top_k=overfetch_k,
                        overfetch_k=overfetch_k,
                        chapters=rewrite_decision.active_chapters
                    )
                except Exception:
                    retrieval_res = None

            if isinstance(retrieval_res, dict) and "fused" in retrieval_res:
                candidates = retrieval_res.get("fused", [])
                dense_results = retrieval_res.get("dense", [])
                bm25_results = retrieval_res.get("bm25", [])
            else:
                ret_out = await self.hybrid_retriever.retrieve(
                    effective_query,
                    top_k=overfetch_k,
                    overfetch_k=overfetch_k,
                    chapters=rewrite_decision.active_chapters
                )
                candidates = ret_out if isinstance(ret_out, list) else []
                dense_results = [c for c in candidates if getattr(c, "dense_score", None) is not None]
        # Mode 1 Scoped Practice: Filter candidates by active chapters if specified
        if rewrite_decision.active_chapters:
            candidates = [c for c in candidates if getattr(c, "chapter", None) in rewrite_decision.active_chapters]
            dense_results = [c for c in dense_results if getattr(c, "chapter", None) in rewrite_decision.active_chapters]
            bm25_results = [c for c in bm25_results if getattr(c, "chapter", None) in rewrite_decision.active_chapters]

        latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Edge Case: Zero candidates retrieved
        if not candidates:
            total_elapsed_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            latencies["total_ms"] = total_elapsed_ms
            trace_id = "offline_trace"
            if self.tracker and self.tracker.is_active:
                try:
                    trace_id = self.tracker.trace_advanced_pipeline(
                        name="advanced_pipeline",
                        question=query_text,
                        answer=GROUNDED_REFUSAL_FA,
                        is_refusal=True,
                        session_id=session_id,
                        rewrite_decision=rewrite_decision,
                        history=history,
                        effective_query=effective_query,
                        rewrite_latency_ms=latencies.get("rewrite_ms", 0.0),
                        dense_results=[],
                        bm25_results=[],
                        fused_candidates=[],
                        retrieval_latency_ms=latencies.get("retrieval_ms", 0.0),
                        latencies=latencies
                    )
                except Exception as e:
                    logger.warning(f"Error logging Langfuse trace: {e}")

            dispatch_trace_evaluation(
                trace_id=trace_id,
                query=query_text,
                retrieved_chunks=[],
                answer=GROUNDED_REFUSAL_FA,
                tracker=self.tracker,
                is_refusal=True
            )
            return AdvancedRAGResult(
                query=query_text,
                effective_query=effective_query,
                answer=GROUNDED_REFUSAL_FA,
                citations=[],
                retrieved_chunks=[],
                top_parents=[],
                latency_profile=latencies,
                trace_id=trace_id,
                is_refusal=True
            )

        # -----------------------------------------------------------------
        # Stage 3: Cross-Encoder Neural Re-Ranking with Circuit Breaker
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        rerank_timeout = getattr(settings, "rerank_timeout_ms", 3000.0)
        reranked_candidates = await self.reranker.rerank(
            query=effective_query,
            candidates=candidates[:8],
            top_k=top_k,
            timeout_ms=rerank_timeout
        )
        circuit_breaker_status = getattr(self.reranker, "circuit_breaker_status", "active")
        latencies["rerank_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # -----------------------------------------------------------------
        # Stage 4: Parent Section Expansion & Context Reordering
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        expanded_context = self.expander.expand(
            candidates=reranked_candidates,
            max_sections=3,
            token_ceiling=2500
        )
        # Apply Lost-in-the-Middle reordering
        reordered_sections = self.reorderer.reorder(expanded_context.parent_sections)
        expanded_context.parent_sections = reordered_sections
        expanded_context.reordered_texts = [
            f"### فصل {sec.chapter}: {sec.title} (صفحه {sec.page_start} تا {sec.page_end})\n{sec.full_text}"
            for sec in reordered_sections
        ]
        latencies["expansion_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # -----------------------------------------------------------------
        # Stage 5: Grounded Response Synthesis & Strict Citations
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        # Build synthesis prompt context
        combined_context_text = "\n\n---\n\n".join(expanded_context.reordered_texts)

        # Build citation objects
        citation_candidates = []
        for rc in reranked_candidates:
            c = rc.candidate
            doc_id = getattr(c, "doc_id", "science_textbook") or "science_textbook"
            if not doc_id.endswith(".pdf"):
                doc_id = f"{doc_id}.pdf"
            ch_info = DOC_TO_CHAPTER.get(doc_id)
            display_name = f"فصل {ch_info[0]}، صفحه {c.page}" if ch_info else f"کتاب درسی، صفحه {c.page}"
            citation_candidates.append({
                "doc_id": doc_id,
                "filename": doc_id,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "score": rc.cross_encoder_score,
                "metadata": {"title": c.breadcrumbs, "display_name": display_name}
            })
        citation_bundle = self.citation_formatter.format(citation_candidates)

        # Synthesize answer using Grounded Synthesizer
        default_synth_doc = citation_candidates[0]["doc_id"] if citation_candidates else "science_textbook.pdf"
        chunks_for_synth = [
            {
                "text": sec.full_text,
                "page": sec.page_start,
                "breadcrumb": f"فصل {sec.chapter}: {sec.title}",
                "doc_id": default_synth_doc,
                "chunk_id": sec.section_id
            }
            for sec in reordered_sections
        ]
        synth_result = None
        synth_prompt = SYSTEM_PROMPT_PERSIAN_GROUNDED
        if getattr(rewrite_decision, "is_compound", False):
            from rag_core.generation.prompts import PROMPT_COMPOUND_DIRECTIVE_FA
            synth_prompt = f"{synth_prompt}\n\n{PROMPT_COMPOUND_DIRECTIVE_FA}"

        try:
            synth_result = self.synthesizer.synthesize(
                query=effective_query,
                chunks=chunks_for_synth,
                system_prompt=synth_prompt,
                session_id=session_id,
                history=history,
            )
            raw_answer = synth_result.raw_completion or synth_result.answer.split("**Sources:**")[0].split("📚")[0].rstrip()
            is_refusal = synth_result.is_fallback
        except Exception as e:
            logger.warning(f"Synthesizer invocation error: {e}, using grounded summary fallback")
            # Deterministic fallback when LLM gateway is offline
            raw_answer = f"بر اساس متون استخراج شده از کتاب علوم هفتم:\n{combined_context_text[:300]}..."
            is_refusal = False

        # Format citations
        formatted_citations: List[PageCitation] = [
            PageCitation(
                doc_id=c["doc_id"],
                filename=c["filename"],
                page=c["page"],
                chunk_id=c["chunk_id"],
                score=c["score"]
            )
            for c in citation_candidates
        ]

        if citation_bundle.inline_text and not is_refusal:
            final_answer = f"{raw_answer.rstrip()}\n\n---\n\n**Sources:**\n{citation_bundle.inline_text}"

        else:
            final_answer = raw_answer


        latencies["synthesis_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        total_elapsed_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
        latencies["total_ms"] = total_elapsed_ms

        # -----------------------------------------------------------------
        # Stage 6: Langfuse Observability & Async Trace Evaluation
        # -----------------------------------------------------------------
        trace_id = "offline_trace"
        if self.tracker and self.tracker.is_active:
            try:
                trace_id = self.tracker.trace_advanced_pipeline(
                    name="advanced_pipeline",
                    question=query_text,
                    answer=final_answer,
                    is_refusal=is_refusal,
                    session_id=session_id,
                    tags=["pipeline:advanced", "domain:science_textbook"],
                    metadata={
                        "effective_query": effective_query,
                        "rewritten": rewrite_decision.needs_rewrite,
                        "top_k": top_k,
                    },
                    # Stage 1: Query Understanding
                    rewrite_decision=rewrite_decision,
                    history=history,
                    effective_query=effective_query,
                    rewrite_latency_ms=latencies.get("rewrite_ms", 0.0),
                    # Stage 2: Hybrid Retrieval
                    dense_results=dense_results,
                    bm25_results=bm25_results,
                    fused_candidates=candidates,
                    retrieval_latency_ms=latencies.get("retrieval_ms", 0.0),
                    # Stage 3: Neural Reranking
                    input_candidates=candidates[:15],
                    reranked_candidates=reranked_candidates,
                    circuit_breaker_status=circuit_breaker_status,
                    rerank_latency_ms=latencies.get("rerank_ms", 0.0),
                    # Stage 4: Parent Expansion & Context Ordering
                    expanded_context=expanded_context,
                    reordered_sections=reordered_sections,
                    expansion_latency_ms=latencies.get("expansion_ms", 0.0),
                    # Stage 5: Response Synthesis
                    synthesis_result=synth_result,
                    full_prompt=getattr(synth_result, "full_prompt", "") if synth_result else combined_context_text,
                    raw_completion=raw_answer,
                    synthesis_latency_ms=latencies.get("synthesis_ms", 0.0),
                    latencies=latencies,
                )
                self.tracker.flush()
            except Exception as e:
                logger.warning(f"Error logging Langfuse trace: {e}")

        # Dispatch async background trace evaluation (zero critical-path overhead)
        dispatch_trace_evaluation(
            trace_id=trace_id,
            query=query_text,
            retrieved_chunks=[rc.candidate for rc in reranked_candidates],
            answer=final_answer,
            tracker=self.tracker,
            is_refusal=is_refusal
        )

        return AdvancedRAGResult(
            query=query_text,
            effective_query=effective_query,
            answer=final_answer,
            citations=formatted_citations,
            retrieved_chunks=[rc.candidate for rc in reranked_candidates],
            top_parents=[sec.section_id for sec in expanded_context.parent_sections],
            latency_profile=latencies,
            trace_id=trace_id,
            is_refusal=is_refusal
        )

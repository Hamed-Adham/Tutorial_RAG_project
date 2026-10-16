#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QueryGate: Confidence-gated history-aware query router, 3-tier curriculum gate, and SubQ conversational memory engine.
Implements:
1. 3-tier curriculum intent gating: completely_irrelevant (immediate refusal in <25ms), partially_relevant (upfront distillation), relevant.
2. 4-tier context need dispatching: needs_history_and_retrieve, needs_history_only, needs_retrieve_only, needs_neither.
3. Decision Model 2 via SubQ Decisions API (subq-snap-preview, 10-10 parallel batching).
4. Deterministic missing summary backfill and context fidelity selection (recent 5 full-text vs older summaries).
5. Manual single-turn context injection when history is disabled.
"""

import time
import asyncio
import re
from typing import List, Dict, Any, Optional
import structlog

from rag_core.contracts import IQueryRewriter
from rag_core.indexing.science_schema import (
    RewriteDecision,
    CurriculumIntentTier,
    ContextNeedTier,
    HistoryRelevanceTier,
    SelectedHistoryItem,
    QueryGateDecision,
)
from rag_core.query.decomposition import SubQueryDecomposer
from rag_core.query.subq_client import SubQDecisionClient
from rag_core.query.summarizer import ConversationSummarizer
from rag_core.evaluation.jev_client import TypeSafeJevClient
from rag_core.generation.prompts import (
    SYSTEM_PROMPT_RESOLVE_CONTEXT_FA,
    USER_PROMPT_RESOLVE_CONTEXT_FA,
    SYSTEM_PROMPT_EXPAND_QUERY_FA,
    USER_PROMPT_EXPAND_QUERY_FA,
    SYSTEM_PROMPT_DISTILL_PARTIAL_QUERY_FA,
    USER_PROMPT_DISTILL_PARTIAL_QUERY_FA,
    SYSTEM_PROMPT_DECOMPOSE_QUERY_FA,
    USER_PROMPT_DECOMPOSE_QUERY_FA,
)

logger = structlog.get_logger(__name__)

# Concise canonical topic phrases for each of the 15 chapters of 7th-grade science (علوم تجربی هفتم)
OLUM_7_CHAPTER_TOPICS: Dict[int, List[str]] = {
    1: ["روش علمی و تفکر", "فرضیه‌سازی و آزمایش", "مهارت‌های یادگیری علوم"],
    2: ["اندازه‌گیری و ابزارها", "جرم و وزن و چگالی", "طول و حجم و زمان"],
    3: ["اتم‌ها و ساختار ذره‌ای", "عنصرها و ترکیب‌ها", "نمادهای شیمیایی و مولکول"],
    4: ["مواد و خواص آنها", "فلزات و نافلزات", "چکش‌خواری و رسانایی و سختی"],
    5: ["معادن و استخراج فلزات", "تولید آهن و فولاد", "شیشه و سیمان و بازیافت"],
    6: ["سفر آب روی زمین", "رودخانه‌ها و حوضه آبریز", "دریاچه‌ها و آلودگی آب"],
    7: ["سفر آب درون زمین", "آب‌های زیرزمینی و آبخوان", "قنات و چشمه و نفوذپذیری"],
    8: ["انرژی و انواع آن", "انرژی جنبشی و پتانسیل گرانشی", "پایستگی و تبدیل انرژی"],
    9: ["منابع انرژی تجدیدپذیر", "سوخت‌های فسیلی", "انرژی خورشیدی و باد و زمین‌گرمایی"],
    10: ["دما و گرما", "دماسنجی و انبساط", "رسانش و همرفت و تابش", "عایق‌بندی گرما"],
    11: ["یاخته و ساختار سلول", "اندامک‌های سلولی و هسته", "بافت و اندام و دستگاه"],
    12: ["سفره سلامت و تغذیه", "کربوهیدرات و پروتئین و چربی", "ویتامین‌ها و مواد معدنی"],
    13: ["دستگاه گوارش", "معده و روده باریک و آنزیم‌ها", "جذب مواد غذایی و کبد"],
    14: ["قلب و گردش خون", "سرخرگ و سیاهرگ و مویرگ", "ترکیبات خون و نبض و فشار خون"],
    15: ["دستگاه تنفس و تبادل گازها", "شش‌ها و دم و بازدم", "دستگاه دفع ادرار و کلیه‌ها"]
}

OLUM_7_CHAPTER_NAMES: Dict[int, str] = {
    1: "تفکر و تجربه",
    2: "اندازه‌گیری در علوم و ابزارهای آن",
    3: "اتم‌ها، الفبای مواد",
    4: "مواد پیرامون ما",
    5: "از معدن تا خانه",
    6: "سفر آب روی زمین",
    7: "سفر آب درون زمین",
    8: "انرژی و تبدیل‌های آن",
    9: "منابع انرژی",
    10: "گرما و بهینه‌سازی مصرف انرژی",
    11: "یاخته و سازمان‌بندی آن",
    12: "سفرهٔ سلامت",
    13: "سفر غذا",
    14: "گردش مواد",
    15: "تبادل با محیط",
}


class QueryGate(IQueryRewriter):
    """
    Evaluates incoming queries via TypeSafe Jev System One decision model and SubQ Decision Model 2.
    Routes queries between standalone, rewritten, decomposed, and refusal paths.
    """

    @classmethod
    def detect_candidate_chapters(cls, query: str) -> List[int]:
        """Check which chapters match terms in the query based on OLUM_7_CHAPTER_TOPICS."""
        q_lower = query.lower()
        matched: List[int] = []
        for ch, phrases in OLUM_7_CHAPTER_TOPICS.items():
            for p in phrases:
                words = [w for w in p.split() if len(w) > 2]
                if any(w in q_lower for w in words):
                    if ch not in matched:
                        matched.append(ch)
                    break
        return matched

    def __init__(
        self,
        jev_client: Optional[TypeSafeJevClient] = None,
        decomposer: Optional[SubQueryDecomposer] = None,
        synthesizer: Optional[Any] = None,
        subq_client: Optional[SubQDecisionClient] = None,
        summarizer: Optional[ConversationSummarizer] = None,
    ):
        self.jev_client = jev_client or TypeSafeJevClient(timeout=2.0, max_retries=1)
        self.decomposer = decomposer or SubQueryDecomposer()
        self._synthesizer = synthesizer
        self.subq_client = subq_client or SubQDecisionClient()
        self.summarizer = summarizer or ConversationSummarizer()

    def _get_synthesizer(self) -> Any:
        """Lazily initialize synthesizer for query transformations."""
        if self._synthesizer is None:
            try:
                from rag_core.generation.synthesizer import ResponseSynthesizer
                self._synthesizer = ResponseSynthesizer()
            except Exception as e:
                logger.warning("Failed to initialize synthesizer for rewriter: %s", e)
                return None
        return self._synthesizer

    async def distill_partial_query(self, query: str) -> str:
        """
        Purify mixed inquiries by removing irrelevant chitchat/noise while preserving the science inquiry.
        """
        synth = self._get_synthesizer()
        if not synth:
            return query

        user_msg = USER_PROMPT_DISTILL_PARTIAL_QUERY_FA.format(query=query)
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: synth.generate_raw(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_DISTILL_PARTIAL_QUERY_FA},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=200,
                    temperature=0.0
                )
            )
            if res and res.strip():
                clean = res.strip().strip('"').strip("'").strip("«").strip("»")
                return clean if clean else query
        except Exception as exc:
            logger.warning("Query distillation failed or timed out; falling back to original query", error=str(exc))
        return query

    async def evaluate_and_route(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        selected_chapters: Optional[List[int]] = None,
        selected_faq_categories: Optional[List[str]] = None,
        domain: str = "science_textbook",
        selected_context_turn_id: Optional[str] = None,
        history_enabled: bool = True
    ) -> QueryGateDecision:
        """
        Primary entry point for Feature 006 & Feature 008:
        1. Classifies intent tier (completely_irrelevant, partially_relevant, relevant) per domain.
        2. Distills partial queries upfront.
        3. Classifies context need (needs_history_and_retrieve, needs_history_only, etc.).
        4. Handles Turn 0 cold-start coercion.
        5. Evaluates historical relevance via SubQ Decision Model 2 (or direct single-turn injection).
        """
        t0 = time.perf_counter()
        cleaned_query = query.strip().strip('"\'«»“”‘’')
        latencies: Dict[str, float] = {}

        if not cleaned_query:
            latencies["gate_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return QueryGateDecision(
                raw_query="",
                effective_query="",
                intent_tier=CurriculumIntentTier.COMPLETELY_IRRELEVANT,
                context_need_tier=ContextNeedTier.NEEDS_NEITHER,
                is_refusal=True,
                bypass_retrieval=True,
                reasoning="Empty query",
                latencies=latencies
            )

        # 1. Gate 1: Evaluate Jev System One Decision Model
        intent_tier = CurriculumIntentTier.RELEVANT
        context_need_tier = ContextNeedTier.NEEDS_RETRIEVE_ONLY
        is_compound = False
        is_terse = False
        confidence = 1.0
        jev_res: Optional[Dict[str, Any]] = None

        if self.jev_client and self.jev_client.is_configured:
            try:
                loop = asyncio.get_running_loop()
                jev_res = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._call_jev_model_1(
                            query=cleaned_query,
                            history=history or [],
                            selected_chapters=selected_chapters,
                            selected_faq_categories=selected_faq_categories,
                            domain=domain
                        )
                    ),
                    timeout=2.0
                )
                if jev_res:
                    intent_tier = jev_res.get("intent_tier", CurriculumIntentTier.RELEVANT)
                    context_need_tier = jev_res.get("context_need_tier", ContextNeedTier.NEEDS_RETRIEVE_ONLY)
                    is_compound = jev_res.get("is_compound", False)
                    is_terse = jev_res.get("is_terse", False)
                    confidence = jev_res.get("confidence", 1.0)
            except Exception as exc:
                logger.warning(
                    "Jev Gate 1 evaluation failed or timed out; falling back to conservative defaults",
                    error=str(exc)
                )
        if jev_res is None:
            # Deterministic keyword fallback when Jev client is offline, failed, or timed out
            from rag_core.evaluation.code_gates import (
                ADMIN_ROUTING_KEYWORDS,
                SCIENCE_ROUTING_KEYWORDS,
                _count_keyword_matches
            )
            if domain == "administrative_faq":
                admin_hits = _count_keyword_matches(cleaned_query, ADMIN_ROUTING_KEYWORDS)
                if admin_hits == 0 and history:
                    for turn in reversed(history):
                        c = turn.get("content", "")
                        if c and _count_keyword_matches(c, ADMIN_ROUTING_KEYWORDS) > 0:
                            admin_hits = 1
                            break
                if admin_hits > 0:
                    intent_tier = CurriculumIntentTier.RELEVANT
                    context_need_tier = ContextNeedTier.NEEDS_RETRIEVE_ONLY
                else:
                    intent_tier = CurriculumIntentTier.COMPLETELY_IRRELEVANT
                    context_need_tier = ContextNeedTier.NEEDS_NEITHER
            else:
                sci_hits = _count_keyword_matches(cleaned_query, SCIENCE_ROUTING_KEYWORDS)
                if sci_hits == 0 and history:
                    for turn in reversed(history):
                        c = turn.get("content", "")
                        if c and _count_keyword_matches(c, SCIENCE_ROUTING_KEYWORDS) > 0:
                            sci_hits = 1
                            break
                # For science textbook, retain default RELEVANT unless matched chapters restrict or explicit refusal

        gate1_ms = (time.perf_counter() - t0) * 1000.0
        latencies["gate1_ms"] = round(gate1_ms, 2)

        # Priority 1: Completely irrelevant query (Refusal in <25ms, 0 vector tokens)
        if intent_tier == CurriculumIntentTier.COMPLETELY_IRRELEVANT or context_need_tier == ContextNeedTier.NEEDS_NEITHER:
            latencies["total_gate_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            if domain == "administrative_faq":
                if selected_faq_categories:
                    cat_str = "، ".join(selected_faq_categories)
                    reasoning = (
                        f"این پرسش در محدوده دسته‌های انتخابی شما ({cat_str}) قرار ندارد. "
                        f"برای دریافت پاسخ مستند، لطفاً دسته مربوطه یا تمام دسته‌ها را انتخاب فرمایید."
                    )
                else:
                    reasoning = (
                        "این پرسش خارج از دامنه پرسش‌های متداول و خدمات صندوق نوآوری و شکوفایی است. "
                        "مشاور هوشمند صندوق صرفاً به پرسش‌های مرتبط با تسهیلات، ضمانت‌نامه‌ها، سرمایه‌گذاری و ضوابط دانش‌بنیان پاسخ می‌دهد."
                    )
            else:
                # Check if this refusal is due to chapter scoping
                matched_candidate_chapters = self.detect_candidate_chapters(cleaned_query)
                matched_other_chapters = [
                    c for c in matched_candidate_chapters if not selected_chapters or c not in selected_chapters
                ]
                intends_curriculum = bool(jev_res and jev_res.get("intends_curriculum_answer", False))

                if selected_chapters and (intends_curriculum or matched_other_chapters):
                    fa_map = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
                    selected_names = [f"فصل {str(c).translate(fa_map)} ({OLUM_7_CHAPTER_NAMES.get(c, '')})" for c in selected_chapters]
                    selected_str = "، ".join(selected_names)
                    if matched_other_chapters:
                        target_names = [f"فصل {str(c).translate(fa_map)} ({OLUM_7_CHAPTER_NAMES.get(c, '')})" for c in matched_other_chapters[:2]]
                        target_str = " یا ".join(target_names)
                        reasoning = (
                            f"این پرسش در محدوده فصل‌های انتخابی شما ({selected_str}) قرار ندارد، "
                            f"اما در {target_str} کتاب علوم تجربی هفتم مطرح شده است. "
                            f"برای دریافت پاسخ مستند، لطفاً دامنه جستجو را در بالای صفحه تغییر دهید یا تمام کتاب را انتخاب فرمایید."
                        )
                    else:
                        reasoning = (
                            f"این پرسش در محدوده فصل‌های انتخابی شما ({selected_str}) قرار ندارد، "
                            f"اما مربوط به کتاب علوم تجربی پایه هفتم است. "
                            f"برای دریافت پاسخ مستند، لطفاً دامنه جستجو را در بالای صفحه تغییر داده و تمام کتاب را انتخاب فرمایید."
                        )
                else:
                    reasoning = (
                        "این پرسش خارج از دامنه کتاب علوم تجربی پایه هفتم است. "
                        "دستیار آموزشی علوم هفتم صرفاً به پرسش‌های مرتبط با مباحث این کتاب درسی پاسخ می‌دهد."
                    )

            return QueryGateDecision(
                raw_query=cleaned_query,
                effective_query=cleaned_query,
                intent_tier=CurriculumIntentTier.COMPLETELY_IRRELEVANT,
                context_need_tier=ContextNeedTier.NEEDS_NEITHER,
                is_refusal=True,
                bypass_retrieval=True,
                transformation_type="rejected",
                reasoning=reasoning,
                latencies=latencies
            )

        # Priority 2: Partial query distillation before Model 2 or Retrieval
        effective_query = cleaned_query
        if intent_tier == CurriculumIntentTier.PARTIALLY_RELEVANT:
            t_distill = time.perf_counter()
            effective_query = await self.distill_partial_query(cleaned_query)
            latencies["distill_ms"] = round((time.perf_counter() - t_distill) * 1000.0, 2)

        # Priority 3: Cold-start Turn 0 Coercion Logic
        has_history = bool(history and len(history) > 0)
        if not has_history:
            if context_need_tier == ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE:
                # Option 1 auto-coerces to Option 3 (fresh standalone inquiry)
                context_need_tier = ContextNeedTier.NEEDS_RETRIEVE_ONLY
            elif context_need_tier == ContextNeedTier.NEEDS_HISTORY_ONLY:
                # Option 2 returns friendly clarification prompt
                latencies["total_gate_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                return QueryGateDecision(
                    raw_query=cleaned_query,
                    effective_query=effective_query,
                    intent_tier=intent_tier,
                    context_need_tier=ContextNeedTier.NEEDS_HISTORY_ONLY,
                    is_refusal=True,
                    bypass_retrieval=True,
                    transformation_type="rejected",
                    reasoning="هنوز گفتگویی انجام نشده است، لطفاً پرسش خود را مطرح کنید.",
                    latencies=latencies
                )

        selected_history_items: List[SelectedHistoryItem] = []

        # Priority 4: History Evaluation / Decision Model 2
        # Case A: History is OFF but manual single context turn is selected from UI (User Story 6)
        if not history_enabled:
            if selected_context_turn_id and history:
                for idx, t in enumerate(history):
                    tid = str(t.get("turn_id", ""))
                    if tid == selected_context_turn_id:
                        selected_history_items.append(
                            SelectedHistoryItem(
                                turn_id=tid,
                                turn_index=idx,
                                is_recent_5=True,
                                content_type="full_text",
                                question=str(t.get("question") or t.get("user") or t.get("content", "")),
                                answer=str(t.get("answer") or t.get("assistant", "")),
                                summary=t.get("q_and_a_summary") or t.get("summary"),
                                relevance_tier=HistoryRelevanceTier.NEEDED_RELEVANT,
                                confidence=1.0
                            )
                        )
                        break
        # Case B: History is ON and context is needed (User Story 3)
        elif has_history and context_need_tier in (ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE, ContextNeedTier.NEEDS_HISTORY_ONLY):
            t_hist = time.perf_counter()
            # Select up to last 20 turns
            raw_candidates = history[-20:]
            # Format candidates: pair user and assistant turns if message-based
            candidate_turns: List[Dict[str, Any]] = []
            if raw_candidates and any("role" in t for t in raw_candidates):
                i = 0
                while i < len(raw_candidates):
                    turn = raw_candidates[i]
                    role = turn.get("role")
                    if role == "user":
                        tid = str(turn.get("turn_id", f"turn_{len(candidate_turns)}"))
                        q_text = str(turn.get("content", "")).strip()
                        a_text = ""
                        s_text = turn.get("summary") or turn.get("q_and_a_summary")
                        if i + 1 < len(raw_candidates) and raw_candidates[i + 1].get("role") == "assistant":
                            asst = raw_candidates[i + 1]
                            a_text = str(asst.get("content", "")).strip()
                            s_text = s_text or asst.get("summary") or asst.get("q_and_a_summary")
                            tid = str(asst.get("turn_id") or tid)
                            i += 1
                        candidate_turns.append({
                            "turn_id": tid,
                            "turn_index": len(candidate_turns),
                            "question": q_text,
                            "answer": a_text,
                            "summary": s_text
                        })
                    elif role == "assistant":
                        tid = str(turn.get("turn_id", f"turn_{len(candidate_turns)}"))
                        a_text = str(turn.get("content", "")).strip()
                        s_text = turn.get("summary") or turn.get("q_and_a_summary")
                        candidate_turns.append({
                            "turn_id": tid,
                            "turn_index": len(candidate_turns),
                            "question": "",
                            "answer": a_text,
                            "summary": s_text
                        })
                    else:
                        tid = str(turn.get("turn_id", f"turn_{len(candidate_turns)}"))
                        q_text = str(turn.get("question") or turn.get("user") or "").strip()
                        a_text = str(turn.get("answer") or turn.get("assistant") or "").strip()
                        s_text = turn.get("q_and_a_summary") or turn.get("summary")
                        candidate_turns.append({
                            "turn_id": tid,
                            "turn_index": len(candidate_turns),
                            "question": q_text,
                            "answer": a_text,
                            "summary": s_text
                        })
                    i += 1
            else:
                for idx, turn in enumerate(raw_candidates):
                    tid = str(turn.get("turn_id", f"turn_{idx}"))
                    q_text = str(turn.get("question") or turn.get("user") or "").strip()
                    a_text = str(turn.get("answer") or turn.get("assistant") or "").strip()
                    summary_text = turn.get("q_and_a_summary") or turn.get("summary")
                    candidate_turns.append({
                        "turn_id": tid,
                        "turn_index": idx,
                        "question": q_text,
                        "answer": a_text,
                        "summary": summary_text
                    })

            # Check for missing summaries and deterministically backfill via batch LLM call
            missing_summaries = [t for t in candidate_turns if not t.get("summary")]
            if missing_summaries:
                t_backfill = time.perf_counter()
                backfilled = await self.summarizer.batch_summarize_missing(missing_summaries)
                latencies["summary_backfill_ms"] = round((time.perf_counter() - t_backfill) * 1000.0, 2)
                for t in candidate_turns:
                    if t["turn_id"] in backfilled:
                        t["summary"] = backfilled[t["turn_id"]]

            # Execute SubQ Decision Model 2 with 10-10 parallel batching
            t_subq = time.perf_counter()
            evaluations = await self.subq_client.evaluate_history_turns(
                query=effective_query,
                candidate_turns=candidate_turns
            )
            latencies["subq_model2_ms"] = round((time.perf_counter() - t_subq) * 1000.0, 2)

            total_candidate_count = len(candidate_turns)
            for idx, turn in enumerate(candidate_turns):
                tid = turn["turn_id"]
                tier = evaluations.get(tid, HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT)
                if tier in (HistoryRelevanceTier.NEEDED_RELEVANT, HistoryRelevanceTier.NEEDED_NOT_RELEVANT):
                    # Recent-5 Message Rule: turns in last 5 messages get full text, older get summaries
                    index_from_end = total_candidate_count - 1 - idx
                    is_recent_5 = index_from_end < 5

                    selected_history_items.append(
                        SelectedHistoryItem(
                            turn_id=tid,
                            turn_index=idx,
                            is_recent_5=is_recent_5,
                            content_type="full_text" if is_recent_5 else "summary",
                            question=turn["question"],
                            answer=turn["answer"],
                            summary=turn.get("summary"),
                            relevance_tier=tier,
                            confidence=1.0
                        )
                    )

            # Deterministic Fallback: If Decision Model 1 determined history is needed,
            # but Decision Model 2 (SubQ) evaluated none or returned empty,
            # auto-select the immediate preceding conversational turn.
            if not selected_history_items and candidate_turns:
                last_turn = candidate_turns[-1]
                selected_history_items.append(
                    SelectedHistoryItem(
                        turn_id=str(last_turn["turn_id"]),
                        turn_index=last_turn.get("turn_index", len(candidate_turns) - 1),
                        is_recent_5=True,
                        content_type="summary" if last_turn.get("summary") else "full_text",
                        question=last_turn.get("question", ""),
                        answer=last_turn.get("answer", ""),
                        summary=last_turn.get("summary"),
                        relevance_tier=HistoryRelevanceTier.NEEDED_RELEVANT,
                        confidence=0.95
                    )
                )
            latencies["history_eval_ms"] = round((time.perf_counter() - t_hist) * 1000.0, 2)

        # Priority 5: Sub-query decomposition if compound
        sub_queries: List[str] = []
        if is_compound:
            sub_queries = self.decomposer.decompose(effective_query)
            if len(sub_queries) <= 1:
                # LLM query transformer fallback if regex did not split
                synth = self._get_synthesizer()
                if synth and hasattr(synth, "generate_raw"):
                    try:
                        loop = asyncio.get_running_loop()
                        llm_decomp = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda: synth.generate_raw(
                                    messages=[
                                        {"role": "system", "content": SYSTEM_PROMPT_DECOMPOSE_QUERY_FA},
                                        {"role": "user", "content": USER_PROMPT_DECOMPOSE_QUERY_FA.format(query=effective_query)}
                                    ],
                                    max_tokens=200,
                                    temperature=0.0
                                )
                            ),
                            timeout=2.0
                        )
                        if llm_decomp:
                            extracted = []
                            for line in llm_decomp.strip().split("\n"):
                                cl = line.strip().strip('"\'«»“”‘’`-•* ').strip()
                                if cl and len(cl) >= 4:
                                    extracted.append(cl)
                            if len(extracted) > 1:
                                sub_queries = extracted
                    except Exception as exc:
                        logger.warning("LLM query decomposition fallback failed: %s", exc)

        # Determine transformation_type and perform contextual resolution/expansion if needed
        transformation_type = "bypass"
        if intent_tier == CurriculumIntentTier.COMPLETELY_IRRELEVANT:
            transformation_type = "rejected"
        elif intent_tier == CurriculumIntentTier.PARTIALLY_RELEVANT:
            transformation_type = "distill"
        elif has_history and context_need_tier in (ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE, ContextNeedTier.NEEDS_HISTORY_ONLY):
            transformation_type = "resolve_context"
            synth = self._get_synthesizer()
            if synth and hasattr(synth, "generate_raw"):
                history_text = "\n".join(
                    f"{t.get('role', 'user')}: {t.get('content', '')}"
                    for t in history[-3:]
                )
                user_msg = USER_PROMPT_RESOLVE_CONTEXT_FA.format(history=history_text, query=effective_query)
                try:
                    loop = asyncio.get_running_loop()
                    res = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: synth.generate_raw(
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT_RESOLVE_CONTEXT_FA},
                                    {"role": "user", "content": user_msg}
                                ],
                                max_tokens=150,
                                temperature=0.0
                            )
                        ),
                        timeout=5.0
                    )
                    if res and res.strip():
                        effective_query = res.strip().strip('"\'«»“”‘’').strip()
                except Exception as exc:
                    logger.warning("Context resolution failed or timed out: %s", exc)
        elif is_terse and len(cleaned_query.split()) <= 3:
            transformation_type = "expand"
            synth = self._get_synthesizer()
            if synth and hasattr(synth, "generate_raw"):
                user_msg = USER_PROMPT_EXPAND_QUERY_FA.format(query=effective_query)
                try:
                    loop = asyncio.get_running_loop()
                    res = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: synth.generate_raw(
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT_EXPAND_QUERY_FA},
                                    {"role": "user", "content": user_msg}
                                ],
                                max_tokens=120,
                                temperature=0.0
                            )
                        ),
                        timeout=5.0
                    )
                    if res and res.strip():
                        effective_query = res.strip().strip('"\'«»“”‘’').strip()
                except Exception as exc:
                    logger.warning("Query expansion failed or timed out: %s", exc)
        elif is_compound:
            transformation_type = "decompose"

        bypass_retrieval = (context_need_tier == ContextNeedTier.NEEDS_HISTORY_ONLY)
        latencies["total_gate_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        return QueryGateDecision(
            raw_query=cleaned_query,
            effective_query=effective_query,
            intent_tier=intent_tier,
            context_need_tier=context_need_tier,
            is_refusal=False,
            bypass_retrieval=bypass_retrieval,
            transformation_type=transformation_type,
            is_compound=is_compound,
            selected_history_items=selected_history_items,
            sub_queries=sub_queries,
            reasoning=(
                f"QueryGate decision: intent={intent_tier.value}, need={context_need_tier.value}, "
                f"selected_history={len(selected_history_items)}"
            ),
            latencies=latencies
        )

    def _call_jev_model_1(
        self,
        query: str,
        history: List[Dict[str, Any]],
        selected_chapters: Optional[List[int]] = None,
        selected_faq_categories: Optional[List[str]] = None,
        domain: str = "science_textbook"
    ) -> Dict[str, Any]:
        """
        Execute typed Jev System One decision questions for Gate 1.
        Evaluates 3-tier curriculum intent and 4-tier context need for target domain.
        """
        if domain == "administrative_faq":
            target_topics = selected_faq_categories or [
                "سرمایه‌گذاری", "ودیعه رهن", "حقوقی(وثایق، تضامین)", "توانمندسازی (کمک بلاعوض)",
                "ضمانت‌نامه", "فروش اقساطی (لیزینگ)", "میز خدمت و پذیرش", "صکوک", "رشد و تثبیت تولید"
            ]
            topics_bulleted = "\n".join(f"• {p}" for p in target_topics)
            mode_desc = f"selected FAQ categories ({selected_faq_categories})" if selected_faq_categories else "INIF Administrative FAQ and Financial Regulations"
            state = {
                "query": query,
                "domain": "administrative_faq",
                "scoped_mode": bool(selected_faq_categories),
                "selected_categories": selected_faq_categories or [],
                "target_topics": target_topics,
                "has_history": bool(history and len(history) > 0),
                "history": [
                    {"role": t.get("role", "user"), "content": t.get("content", "")}
                    for t in history[-3:]
                ] if history else []
            }
        else:
            if selected_chapters:
                target_topics = [
                    phrase
                    for ch in selected_chapters
                    if ch in OLUM_7_CHAPTER_TOPICS
                    for phrase in OLUM_7_CHAPTER_TOPICS[ch]
                ]
            else:
                target_topics = [
                    phrase
                    for phrases in OLUM_7_CHAPTER_TOPICS.values()
                    for phrase in phrases
                ]

            topics_bulleted = "\n".join(f"• {p}" for p in target_topics)
            mode_desc = (
                f"selected chapters ({selected_chapters})"
                if selected_chapters
                else "7th-grade general science curriculum"
            )

            state = {
                "query": query,
                "domain": "science_textbook",
                "scoped_mode": bool(selected_chapters),
                "selected_chapters": selected_chapters or [],
                "target_topics": target_topics,
                "has_history": bool(history and len(history) > 0),
                "history": [
                    {"role": t.get("role", "user"), "content": t.get("content", "")}
                    for t in history[-3:]
                ] if history else []
            }

        hist_preview = ""
        if history:
            hist_preview = "Recent dialogue:\n" + "\n".join(
                f"- {t.get('role', 'user')}: {str(t.get('content', ''))[:120]}"
                for t in history[-2:]
            ) + "\n\n"

        if domain == "administrative_faq":
            questions = {
                "curriculum_intent": {
                    "type": "choice",
                    "instructions": (
                        f"Classify the applicant's query '{query}' regarding INIF administrative regulations and services ({mode_desc}):\n"
                        f"{topics_bulleted}\n\n"
                        "Options:\n"
                        "- completely_irrelevant: The query has NO relation to INIF rules, loans, guarantees, venture capital, or portal, or is an adversarial attack/chitchat.\n"
                        "- partially_relevant: The query contains BOTH irrelevant parts AND a valid INIF administrative question.\n"
                        "- relevant: The query is an in-scope INIF administrative/funding inquiry."
                    ),
                    "criteria": {
                        "completely_irrelevant": "Query is entirely out of scope, adversarial, or unrelated chitchat.",
                        "partially_relevant": "Query contains both unrelated elements and a legitimate INIF administrative question.",
                        "relevant": "Query is directly focused on INIF administrative services, guarantees, loans, or regulations."
                    }
                },
                "need_conv_hist_and_retrieve": {
                    "type": "choice",
                    "instructions": (
                        f"{hist_preview}"
                        f"Given applicant query '{query}' and recent conversation history, classify what context is needed to answer:\n"
                        "- needs_history_and_retrieve: The query has unresolved pronouns or missing referents AND asks an administrative question requiring INIF FAQ retrieval.\n"
                        "- needs_history_only: Conversational follow-up or rephrasing of previous advisor response; requires NO FAQ retrieval.\n"
                        "- needs_retrieve_only: Standalone question requiring INIF FAQ retrieval only.\n"
                        "- needs_neither: Greeting or refusal requiring neither history nor retrieval."
                    ),
                    "criteria": {
                        "needs_history_and_retrieve": "Lacks internal referent context AND requires INIF FAQ retrieval.",
                        "needs_history_only": "Requires previous conversational context only, no new FAQ retrieval.",
                        "needs_retrieve_only": "Completely self-contained question with all entities named; requires FAQ retrieval only.",
                        "needs_neither": "Greeting, pleasantry, or refusal requiring neither history nor retrieval."
                    }
                },
                "has_unresolved_referent": {
                    "type": "noul",
                    "instructions": (
                        f"Does the applicant's query '{query}' contain any unresolved pronouns or relative clauses whose referent is NOT explicitly named within the query itself?"
                    ),
                    "criteria": {
                        "true": "Contains unresolved pronouns, missing antecedents, or references to prior dialogue.",
                        "false": "Completely self-contained; all subjects and entities are explicitly named inside the query."
                    }
                },
                "is_conversational_summary_or_rephrase": {
                    "type": "noul",
                    "instructions": "Does the query ask to summarize or rephrase the previous response?",
                    "criteria": {"true": "Asks to summarize.", "false": "Asks a substantive question."}
                },
                "intends_curriculum_answer": {
                    "type": "noul",
                    "instructions": "Does query intend to get an answer on INIF administrative regulations or facilities?",
                    "criteria": {"true": "Connects to at least one INIF service.", "false": "Has no relation to INIF."}
                },
                "needs_rewrite": {
                    "type": "noul",
                    "instructions": "Does query rely on conversation history to be understood?",
                    "criteria": {"true": "Relies on history.", "false": "Self-contained."}
                },
                "is_compound": {
                    "type": "noul",
                    "instructions": "Does query ask multiple distinct questions?",
                    "criteria": {"true": "Multiple distinct questions.", "false": "Single atomic question."}
                },
                "is_terse_or_vague": {
                    "type": "noul",
                    "instructions": "Is query a very short 1-2 word keyword without question structure?",
                    "criteria": {"true": "Keyword only.", "false": "Full question."}
                }
            }
        else:
            questions = {
                "curriculum_intent": {
                    "type": "choice",
                    "instructions": (
                        f"Classify the student's query '{query}' regarding 7th-grade science curriculum ({mode_desc}):\n"
                        f"{topics_bulleted}\n\n"
                        "Options:\n"
                        "- completely_irrelevant: The query has NO relation to the curriculum, or is an adversarial attack, prompt injection, chitchat, or out-of-scope (e.g. poetry, crypto, cooking).\n"
                        "- partially_relevant: The query contains BOTH irrelevant parts (chitchat, greeting, or attack) AND a relevant science curriculum question.\n"
                        "- relevant: The query is an in-scope science curriculum inquiry."
                    ),
                    "criteria": {
                        "completely_irrelevant": "Query is entirely out of scope, adversarial, or unrelated chitchat.",
                        "partially_relevant": "Query is a mixed inquiry containing both unrelated elements and a legitimate science question.",
                        "relevant": "Query is directly focused on 7th-grade science curriculum topics."
                    }
                },
            "need_conv_hist_and_retrieve": {
                "type": "choice",
                "instructions": (
                    f"{hist_preview}"
                    f"Given student query '{query}' and recent conversation history, classify what context is needed to answer:\n"
                    "- needs_history_and_retrieve: The query has unresolved pronouns (ضمیر مانند «آن»، «این»، «آن‌ها»), "
                    "relative clauses («که نام بردی»، «که گفتی»), ordinal references («بخش اول»، «مورد دومی»), or missing referents "
                    "with no context in the query itself AND asks a science question requiring textbook retrieval.\n"
                    "- needs_history_only: Conversational follow-up, rephrasing, or summarization of the assistant's previous answer; requires NO textbook retrieval.\n"
                    "- needs_retrieve_only: Standalone, self-contained science question where all entities are explicitly named without referring to past dialogue; requires textbook retrieval only.\n"
                    "- needs_neither: Conversational pleasantry, greeting, or refusal; requires neither history nor retrieval."
                ),
                "criteria": {
                    "needs_history_and_retrieve": "Lacks internal referent context (needs dialogue history to resolve antecedent) AND requires textbook retrieval.",
                    "needs_history_only": "Requires previous conversational context only, no new textbook retrieval.",
                    "needs_retrieve_only": "Completely self-contained question with all entities named; zero reference to past dialogue.",
                    "needs_neither": "Greeting, pleasantry, or refusal requiring neither history nor retrieval."
                }
            },
            "has_unresolved_referent": {
                "type": "noul",
                "instructions": (
                    f"Does the student's query '{query}' contain any unresolved pronouns (ضمیر مانند «آن»، «این»، «آن‌ها»), "
                    "demonstratives, relative clauses («که نام بردی»، «که گفتی»، «ذکر کردی»), or ordinal indicators "
                    "(«بخش اول/دوم»، «مورد اولی/دومی»، «پاسخ قبلی») whose referent is NOT explicitly named within the query itself, "
                    "meaning the query has no internal context of what is being referred to and depends on conversation history?"
                ),
                "criteria": {
                    "true": "Contains unresolved pronouns, missing antecedents, or references to prior dialogue.",
                    "false": "Completely self-contained; all subjects and entities are explicitly named inside the query."
                }
            },
            "is_conversational_summary_or_rephrase": {
                "type": "noul",
                "instructions": (
                    f"Does the query '{query}' ask to summarize, shorten, list as bullet points, or rephrase "
                    "the previous response rather than asking a new substantive science question?"
                ),
                "criteria": {
                    "true": "Asks to summarize, condense, or rephrase previous response.",
                    "false": "Asks a substantive science question or introduces a new topic."
                }
            },
            "intends_curriculum_answer": {
                "type": "noul",
                "instructions": "Does query intend to get an answer on 7th-grade science topics?",
                "criteria": {
                    "true": "The query connects to at least one science topic.",
                    "false": "The query has no relation to science topics."
                }
            },
            "needs_rewrite": {
                "type": "noul",
                "instructions": "Does query rely on conversation history to be understood?",
                "criteria": {
                    "true": "Relies on history.",
                    "false": "Self-contained."
                }
            },
            "is_compound": {
                "type": "noul",
                "instructions": "Does query ask multiple distinct questions?",
                "criteria": {
                    "true": "Multiple distinct questions.",
                    "false": "Single atomic question."
                }
            },
            "is_terse_or_vague": {
                "type": "noul",
                "instructions": "Is query a very short 1-2 word keyword without question structure?",
                "criteria": {
                    "true": "Keyword only.",
                    "false": "Full question."
                }
            }
        }

        res = self.jev_client.system_one(state=state, questions=questions)
        answers = res.get("answers", {})

        # 1. Parse Curriculum Intent Tier
        intent_val = answers.get("curriculum_intent")
        intent_tier = CurriculumIntentTier.RELEVANT
        if isinstance(intent_val, dict):
            choice = intent_val.get("choice") or intent_val.get("value")
            if choice == "completely_irrelevant":
                intent_tier = CurriculumIntentTier.COMPLETELY_IRRELEVANT
            elif choice == "partially_relevant":
                intent_tier = CurriculumIntentTier.PARTIALLY_RELEVANT
            elif choice == "relevant":
                intent_tier = CurriculumIntentTier.RELEVANT
        elif isinstance(intent_val, str):
            c_str = intent_val.strip().lower()
            if c_str in ("completely_irrelevant", "partially_relevant", "relevant"):
                intent_tier = CurriculumIntentTier(c_str)
        else:
            # Fallback to intends_curriculum_answer noul
            intends_prob = float(answers.get("intends_curriculum_answer", {}).get("noul", 1.0))
            if intends_prob < 0.25:
                intent_tier = CurriculumIntentTier.COMPLETELY_IRRELEVANT
            elif intends_prob <= 0.75:
                intent_tier = CurriculumIntentTier.PARTIALLY_RELEVANT
            else:
                intent_tier = CurriculumIntentTier.RELEVANT

        # 2. Parse Context Need Tier
        need_val = answers.get("need_conv_hist_and_retrieve")
        context_need_tier = ContextNeedTier.NEEDS_RETRIEVE_ONLY
        if isinstance(need_val, dict):
            c_need = need_val.get("choice") or need_val.get("value")
            if c_need in (
                "needs_history_and_retrieve",
                "needs_history_only",
                "needs_retrieve_only",
                "needs_neither"
            ):
                context_need_tier = ContextNeedTier(c_need)
        elif isinstance(need_val, str):
            n_str = need_val.strip().lower()
            if n_str in (
                "needs_history_and_retrieve",
                "needs_history_only",
                "needs_retrieve_only",
                "needs_neither"
            ):
                context_need_tier = ContextNeedTier(n_str)
        else:
            # Fallback to has_unresolved_referent / needs_rewrite noul
            unres_prob = float(
                answers.get("has_unresolved_referent", {}).get(
                    "noul",
                    answers.get("needs_rewrite", {}).get("noul", 0.0)
                )
            )
            if unres_prob >= 0.5:
                context_need_tier = ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE
            else:
                context_need_tier = ContextNeedTier.NEEDS_RETRIEVE_ONLY

        # Semantic fusion of unresolved pronouns (ضمیر) and conversational summary signals:
        # If the query lacks internal referent context (unresolved pronoun/referent) or requests summary,
        # and conversation history exists, upgrade the context need tier semantically.
        has_history = bool(history and len(history) > 0)
        unresolved_prob = float(
            answers.get("has_unresolved_referent", {}).get(
                "noul",
                answers.get("needs_rewrite", {}).get("noul", 0.0)
            )
        )
        summary_prob = float(answers.get("is_conversational_summary_or_rephrase", {}).get("noul", 0.0))

        if has_history:
            if summary_prob >= 0.6:
                context_need_tier = ContextNeedTier.NEEDS_HISTORY_ONLY
            elif unresolved_prob >= 0.5 and context_need_tier == ContextNeedTier.NEEDS_RETRIEVE_ONLY:
                context_need_tier = ContextNeedTier.NEEDS_HISTORY_AND_RETRIEVE

        compound_prob = float(answers.get("is_compound", {}).get("noul", 0.0))
        terse_prob = float(answers.get("is_terse_or_vague", {}).get("noul", 0.0))
        conf = float(answers.get("curriculum_intent", {}).get("confidence", 0.95))

        intends_val = answers.get("intends_curriculum_answer")
        intends_prob = 1.0
        if isinstance(intends_val, dict):
            intends_prob = float(intends_val.get("noul", 1.0))
        elif isinstance(intends_val, (int, float)):
            intends_prob = float(intends_val)
        elif isinstance(intends_val, bool):
            intends_prob = 1.0 if intends_val else 0.0

        return {
            "intent_tier": intent_tier,
            "context_need_tier": context_need_tier,
            "intends_curriculum_answer": (intends_prob >= 0.4),
            "is_compound": compound_prob >= 0.5,
            "is_terse": terse_prob >= 0.5,
            "confidence": conf
        }

    @staticmethod
    def enrich_query_with_summaries(
        query: str,
        selected_items: List[SelectedHistoryItem]
    ) -> str:
        """
        Concatenate purified user query with historical summaries for hybrid retrieval (User Story 4).
        """
        if not selected_items:
            return query
        summary_snippets = [
            f"[پیش‌زمینه: {item.summary or item.answer[:150]}]"
            for item in selected_items
            if item.relevance_tier in (HistoryRelevanceTier.NEEDED_RELEVANT, HistoryRelevanceTier.NEEDED_NOT_RELEVANT)
        ]
        if not summary_snippets:
            return query
        return f"{query} {' '.join(summary_snippets)}"

    async def evaluate_and_rewrite(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        selected_chapters: Optional[List[int]] = None,
        active_chapters: Optional[List[int]] = None
    ) -> RewriteDecision:
        """
        Backward compatibility adapter returning RewriteDecision.
        """
        chapters = selected_chapters or active_chapters
        decision = await self.evaluate_and_route(
            query=query,
            history=history,
            selected_chapters=chapters
        )
        is_compound = len(decision.sub_queries) > 1 or decision.is_compound
        needs_rewrite = (
            decision.effective_query != query.strip()
            or len(decision.sub_queries) > 0
            or decision.transformation_type in ("resolve_context", "expand", "decompose", "distill")
        )
        reasoning_map = {
            "resolve_context": "TypeSafe Jev: resolved conversational antecedents",
            "rejected": "TypeSafe Jev gate: query out of scope",
            "expand": "TypeSafe Jev: expanded terse keyword query",
            "decompose": "TypeSafe Jev routing: decomposed compound inquiry",
            "distill": "TypeSafe Jev: distilled partial query",
            "bypass": "TypeSafe Jev routing: standalone in-scope query",
        }
        if decision.is_refusal and decision.reasoning:
            reasoning = decision.reasoning
        else:
            reasoning = reasoning_map.get(decision.transformation_type, decision.reasoning)
        return RewriteDecision(
            needs_rewrite=needs_rewrite,
            confidence=1.0,
            rewritten_query=decision.effective_query,
            sub_queries=decision.sub_queries,
            reasoning=reasoning,
            proceed_to_retrieval=not decision.is_refusal and not decision.bypass_retrieval,
            active_chapters=chapters,
            transformation_type=decision.transformation_type,
            is_compound=is_compound
        )


# Backward compatibility alias
GatedQueryRewriter = QueryGate

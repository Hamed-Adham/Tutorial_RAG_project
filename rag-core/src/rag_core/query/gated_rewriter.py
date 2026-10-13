#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GatedQueryRewriter: Confidence-gated history-aware query rewriter and curriculum topic router.
Supports:
1. Mode 1 (Scoped Practice): Evaluates query against user-selected chapters.
2. Mode 2 (Global Open Inquiries): Evaluates binary relevance against all 15 curriculum topic phrases.
If Jev fails or is unconfigured, gracefully passes the query directly to the next stage in the pipeline.
"""

import time
from typing import List, Dict, Any, Optional
import structlog

from rag_core.contracts import IQueryRewriter
from rag_core.indexing.science_schema import RewriteDecision
from rag_core.query.decomposition import SubQueryDecomposer
from rag_core.evaluation.jev_client import TypeSafeJevClient
from rag_core.generation.prompts import (
    SYSTEM_PROMPT_RESOLVE_CONTEXT_FA,
    USER_PROMPT_RESOLVE_CONTEXT_FA,
    SYSTEM_PROMPT_EXPAND_QUERY_FA,
    USER_PROMPT_EXPAND_QUERY_FA,
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


class GatedQueryRewriter(IQueryRewriter):
    """
    Evaluates incoming queries via TypeSafe Jev System One decision model.
    Routes queries between standalone, rewritten, and decomposed paths,
    and performs a binary relevance gate against curriculum topic phrases.
    """

    def __init__(
        self,
        jev_client: Optional[TypeSafeJevClient] = None,
        decomposer: Optional[SubQueryDecomposer] = None,
        synthesizer: Optional[Any] = None
    ):
        self.jev_client = jev_client or TypeSafeJevClient()
        self.decomposer = decomposer or SubQueryDecomposer()
        self._synthesizer = synthesizer

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

    async def evaluate_and_rewrite(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        selected_chapters: Optional[List[int]] = None
    ) -> RewriteDecision:
        """
        Evaluate and rewrite the query using TypeSafe Jev System One.
        Supports Mode 1 (scoped to selected_chapters) and Mode 2 (global open search).
        If Jev fails or is unconfigured, directly pass the query to the next pipeline step.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RewriteDecision(
                needs_rewrite=False,
                confidence=1.0,
                rewritten_query="",
                sub_queries=[],
                reasoning="Empty query",
                proceed_to_retrieval=False,
                active_chapters=selected_chapters
            )

        # 1. Primary: Evaluate via TypeSafe Jev System One Decision Model
        if self.jev_client and self.jev_client.is_configured:
            try:
                decision = self._call_jev_decision_model(
                    query=cleaned_query,
                    history=history or [],
                    selected_chapters=selected_chapters
                )
                if decision is not None:
                    return decision
            except Exception as exc:
                logger.warning(
                    "Jev decision model call failed; passing query directly to next pipeline step",
                    error=str(exc)
                )

        # 2. Direct Pass-Through Fallback:
        # If Jev call fails or is unconfigured, pass query directly to the next pipeline step.
        return RewriteDecision(
            needs_rewrite=False,
            confidence=1.0,
            rewritten_query=cleaned_query,
            sub_queries=[],
            reasoning="Jev model call bypassed/failed; passed directly to next step",
            proceed_to_retrieval=True,
            active_chapters=selected_chapters,
            transformation_type="fallback",
            is_compound=False
        )

    def _call_jev_decision_model(
        self,
        query: str,
        history: List[Dict[str, str]],
        selected_chapters: Optional[List[int]] = None
    ) -> Optional[RewriteDecision]:
        """
        Execute typed Jev System One decision questions:
        1. intends_curriculum_answer: binary noul check whether query connects to target topic phrases.
        2. needs_rewrite: noul judgment on whether query has pronoun/contextual dependence.
        3. is_compound: noul judgment on whether query contains multiple distinct sub-questions.
        """
        # Determine target topic phrases based on selected mode
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

        state = {
            "query": query,
            "scoped_mode": bool(selected_chapters),
            "selected_chapters": selected_chapters or [],
            "target_topics": target_topics,
            "target_topics_formatted": topics_bulleted,
            "has_history": bool(history and len(history) > 0),
            "history": [
                {"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in history[-3:]
            ] if history else []
        }

        mode_desc = (
            f"selected chapters ({selected_chapters})"
            if selected_chapters
            else "7th-grade general science curriculum"
        )

        questions = {
            "intends_curriculum_answer": {
                "type": "noul",
                "instructions": (
                    f"Does `query` intend to get an answer, explanation, or information based on ANY of the "
                    f"following curriculum topics for {mode_desc}?\n"
                    f"{topics_bulleted}\n\n"
                    f"Score TRUE if the query connects to, asks about, or can be answered using concepts from any of "
                    f"these topics, even if phrased casually, informally, or curiously. "
                    f"Score FALSE only if the query is unrelated to all of these topics (e.g. general chit-chat, "
                    f"computer programming, pop culture, history, finances, or gibberish)."
                ),
                "criteria": {
                    "true": "The query connects to at least one of the target curriculum topics.",
                    "false": "The query has no relation to any of the target curriculum topics."
                }
            },
            "needs_rewrite": {
                "type": "noul",
                "instructions": (
                    "Does `query` rely on conversation `history` to be understood, or contain "
                    "ambiguous references/pronouns (such as آن, این, او)?"
                ),
                "criteria": {
                    "true": "The query cannot be understood without previous context.",
                    "false": "The query is completely self-contained."
                }
            },
            "is_compound": {
                "type": "noul",
                "instructions": (
                    "Does `query` ask multiple distinct questions, comparisons, or tasks that "
                    "should be split into independent sub-queries?"
                ),
                "criteria": {
                    "true": "The query asks about two or more distinct concepts or questions.",
                    "false": "The query is a single atomic question."
                }
            },
            "is_terse_or_vague": {
                "type": "noul",
                "instructions": (
                    "Is `query` extremely short (1 to 2 words), keyword-only without an explicit question verb, "
                    "or so terse that dense vector retrieval would suffer from ambiguity?"
                ),
                "criteria": {
                    "true": "The query is an isolated keyword or very short phrase without question context.",
                    "false": "The query is a full, articulated sentence or question."
                }
            }
        }

        res = self.jev_client.system_one(state=state, questions=questions)
        answers = res.get("answers", {})

        intends_ans = answers.get("intends_curriculum_answer", {})
        intends_prob = float(intends_ans.get("noul", 1.0))

        noul_ans = answers.get("needs_rewrite", {})
        rewrite_prob = float(noul_ans.get("noul", 0.0))
        conf = float(noul_ans.get("confidence", 0.90))

        compound_ans = answers.get("is_compound", {})
        compound_prob = float(compound_ans.get("noul", 0.0))

        terse_ans = answers.get("is_terse_or_vague", {})
        terse_prob = float(terse_ans.get("noul", 0.0))

        proceed_to_retrieval = intends_prob >= 0.5
        is_compound = compound_prob >= 0.5
        needs_context = rewrite_prob >= 0.5
        is_terse = terse_prob >= 0.5

        # Priority 1: Out-of-scope query: gate halts retrieval
        if not proceed_to_retrieval:
            return RewriteDecision(
                needs_rewrite=False,
                confidence=conf,
                rewritten_query=query,
                sub_queries=[],
                reasoning=(
                    f"TypeSafe Jev gate: query out of scope for "
                    f"{'selected chapters' if selected_chapters else 'curriculum topics'} "
                    f"(intends={intends_prob:.2f})"
                ),
                proceed_to_retrieval=False,
                active_chapters=selected_chapters,
                transformation_type="rejected",
                is_compound=False
            )

        # Priority 2: Standalone, atomic in-scope query: bypass gate (<10ms, 0 tokens)
        if not needs_context and not is_compound and not is_terse:
            return RewriteDecision(
                needs_rewrite=False,
                confidence=conf,
                rewritten_query=query,
                sub_queries=[],
                reasoning=(
                    f"TypeSafe Jev routing: standalone in-scope query "
                    f"(intends={intends_prob:.2f}, rewrite={rewrite_prob:.2f}, compound={compound_prob:.2f})"
                ),
                proceed_to_retrieval=True,
                active_chapters=selected_chapters,
                transformation_type="bypass",
                is_compound=False
            )

        # Priority 3: Conversational pronoun / context resolution
        if needs_context and history:
            rewritten_text = query
            synth = self._get_synthesizer()
            if synth and hasattr(synth, "generate_raw"):
                history_text = "\n".join(
                    f"{t.get('role', 'user')}: {t.get('content', '')}"
                    for t in history[-3:]
                )
                user_msg = USER_PROMPT_RESOLVE_CONTEXT_FA.format(history=history_text, query=query)
                res = synth.generate_raw(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_RESOLVE_CONTEXT_FA},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=150,
                    temperature=0.0
                )
                if res and res.strip():
                    rewritten_text = res.strip().strip('"').strip("'")

            return RewriteDecision(
                needs_rewrite=True,
                confidence=conf,
                rewritten_query=rewritten_text,
                sub_queries=[],
                reasoning=f"TypeSafe Jev: resolved conversational antecedents (rewrite={rewrite_prob:.2f})",
                proceed_to_retrieval=True,
                active_chapters=selected_chapters,
                transformation_type="resolve_context",
                is_compound=False
            )

        # Priority 4: Terse keyword expansion
        if is_terse and len(query.split()) <= 3:
            expanded_text = query
            synth = self._get_synthesizer()
            if synth and hasattr(synth, "generate_raw"):
                user_msg = USER_PROMPT_EXPAND_QUERY_FA.format(query=query)
                res = synth.generate_raw(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_EXPAND_QUERY_FA},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=120,
                    temperature=0.0
                )
                if res and res.strip():
                    expanded_text = res.strip().strip('"').strip("'")

            return RewriteDecision(
                needs_rewrite=True,
                confidence=conf,
                rewritten_query=expanded_text,
                sub_queries=[],
                reasoning=f"TypeSafe Jev: expanded terse keyword query (terse={terse_prob:.2f})",
                proceed_to_retrieval=True,
                active_chapters=selected_chapters,
                transformation_type="expand",
                is_compound=False
            )

        # Priority 5: Compound multi-concept query decomposition
        sub_queries = []
        if is_compound:
            sub_queries = self.decomposer.decompose(query)

        return RewriteDecision(
            needs_rewrite=True,
            confidence=conf,
            rewritten_query=query,
            sub_queries=sub_queries,
            reasoning=(
                f"TypeSafe Jev routing: decomposed compound inquiry "
                f"(compound={compound_prob:.2f}, parts={len(sub_queries)})"
            ),
            proceed_to_retrieval=True,
            active_chapters=selected_chapters,
            transformation_type="decompose",
            is_compound=True
        )

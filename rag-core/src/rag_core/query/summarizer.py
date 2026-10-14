# -*- coding: utf-8 -*-
"""
Conversation Summarizer for Feature 006: QueryGate & Conversational Memory Routing.
Provides:
1. Batch summarization of historical Q&A turns missing summaries.
2. 2.0s strict fallback guard using first 250 characters of answer if LLM fails or times out.
3. Single turn summarizer helper for incremental caching in synthesis.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
import structlog

from rag_core.config import settings

logger = structlog.get_logger(__name__)


class ConversationSummarizer:
    """
    Summarizes conversational turns into concise Persian summaries (<= 512 tokens / ~6 lines).
    Enforces a strict 2.0s fallback guard for batch backfills.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 2.0
    ):
        self._base_url = base_url or settings.nine_router_base_url or "http://localhost:20128/v1"
        self._api_key = api_key if (api_key and api_key != "none") else (settings.nine_router_api_key or "none")
        self._model = model or settings.llm_model or "RAG-synthesizer"
        self.timeout = timeout
        self._async_client = None

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._async_client

    async def batch_summarize_missing(
        self,
        missing_turns: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Backfill summaries for all turns in a single batch LLM call.
        Enforces a 2.0s timeout guard; if exceeded, safely falls back to answer[:250].

        Args:
            missing_turns: List of turns needing summaries [{'turn_id': str, 'question': str, 'answer': str}]

        Returns:
            Dict mapping turn_id to summary text.
        """
        if not missing_turns:
            return {}

        fallback_summaries = {
            str(t.get("turn_id", f"turn_{i}")): str(t.get("answer", ""))[:250].strip()
            for i, t in enumerate(missing_turns)
        }

        try:
            return await asyncio.wait_for(
                self._execute_batch_llm_call(missing_turns),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Batch summary backfill timed out (>2.0s); falling back to 250-char answers",
                turn_count=len(missing_turns)
            )
            return fallback_summaries
        except Exception as exc:
            logger.warning(
                "Batch summary backfill failed; falling back to 250-char answers",
                error=str(exc),
                turn_count=len(missing_turns)
            )
            return fallback_summaries

    async def _execute_batch_llm_call(
        self,
        turns: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Call LLM with structured JSON output schema."""
        client = self._get_async_client()

        turns_payload = [
            {
                "turn_id": str(t.get("turn_id", f"turn_{i}")),
                "question": str(t.get("question", "")).strip(),
                "answer": str(t.get("answer", "")).strip()[:1000]
            }
            for i, t in enumerate(turns)
        ]

        prompt = (
            "شما یک دستیار خلاصه‌سازی آموزشی علوم پایه هفتم هستید. "
            "وظیفه شما این است که برای هر یک از جفت‌های پرسش و پاسخ زیر، یک خلاصه مستقل، فشرده و به زبان فارسی "
            "(حداکثر ۱ بند، تا ۶ سطر، بدون مقدمه و کلمات اضافی) تولید کنید که نکات علمی و مفاهیم اصلی پاسخ را پوشش دهد.\n\n"
            f"داده‌های ورودی:\n{json.dumps(turns_payload, ensure_ascii=False, indent=2)}\n\n"
            "پاسخ را دقیقاً در قالب یک شیء JSON با ساختار زیر بازگردانید:\n"
            "{\n"
            '  "summaries": {\n'
            '    "<turn_id>": "<متن خلاصه فارسی>"\n'
            "  }\n"
            "}"
        )

        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a concise educational summarizer. Respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=len(turns) * 200,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        raw_summaries = parsed.get("summaries", parsed)

        results: Dict[str, str] = {}
        for i, t in enumerate(turns):
            tid = str(t.get("turn_id", f"turn_{i}"))
            summary = raw_summaries.get(tid)
            if summary and isinstance(summary, str) and summary.strip():
                results[tid] = summary.strip()
            else:
                results[tid] = str(t.get("answer", ""))[:250].strip()

        return results

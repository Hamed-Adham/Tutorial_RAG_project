#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naive synthesizer using local LLM endpoint (OpenAI-compatible /chat/completions).
Implements baseline single-shot prompting.
"""

import time
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from rag_core.naive.chunker import Chunk

NAIVE_SYSTEM_PROMPT = """شما یک دستیار هوشمند و صادق هستید. وظیفه شما پاسخ دقیق به پرسش کاربر صرفاً بر اساس متون ارائه شده در بخش «متن مستندات» است.
قوانین اکید:
۱. فقط و فقط از اطلاعات موجود در متن مستندات استفاده کنید.
۲. اگر پاسخ به طور کامل در متن وجود ندارد یا سوال خارج از متن است، صراحتاً بگویید: «اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد.»
۳. از حدس زدن، فرضیه‌سازی و تولید اطلاعات خارج از متن جداً خودداری کنید."""

NAIVE_USER_TEMPLATE = """متن مستندات:
---
{context}
---

پرسش کاربر:
{query}

پاسخ:"""

class NaiveSynthesizer:
    """Synthesizer that calls local LLM via HTTP POST."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ):
        from rag_core.config import settings
        self.base_url = (base_url or settings.nine_router_base_url).rstrip("/")
        self.api_key = api_key or settings.nine_router_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def format_context(self, retrieved_chunks: List[Any]) -> str:
        """Format retrieved chunks into a single concatenated context string."""
        if not retrieved_chunks:
            return "هیچ متنی بازیابی نشد."

        contexts = []
        for i, item in enumerate(retrieved_chunks, 1):
            if isinstance(item, tuple):
                chunk, score = item
                contexts.append(f"[مستند {i} | منبع: {chunk.source_name} | امتیاز: {score:.3f}]\n{chunk.text}")
            else:
                contexts.append(f"[مستند {i}]\n{getattr(item, 'text', str(item))}")

        return "\n\n".join(contexts)

    def synthesize(
        self,
        query: str,
        retrieved_chunks: List[Any],
        timeout: int = 90
    ) -> str:
        """Generate response given query and retrieved context."""
        context_str = self.format_context(retrieved_chunks)
        user_content = NAIVE_USER_TEMPLATE.format(context=context_str, query=query)

        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        from rag_core.generation.synthesizer import sanitize_math_markup

        last_error = None
        for attempt in range(1, 3):
            try:
                # Try OpenAI client first if available, else urllib
                try:
                    from openai import OpenAI
                    client = OpenAI(base_url=self.base_url, api_key=self.api_key or "none", timeout=timeout)
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=payload["messages"],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = resp.choices[0].message.content or ""
                    return sanitize_math_markup(content.strip())
                except ImportError:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        content = data["choices"][0]["message"]["content"] or ""
                        return sanitize_math_markup(content.strip())
            except Exception as e:
                last_error = e
                print(f"  [NaiveSynthesizer warning] Attempt {attempt} failed ({e}).")
                time.sleep(1)

        # Fallback to extractive summary from highest-ranked chunk to prevent ASGI crash
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            top_text = getattr(top_chunk, "text", "") or ""
            return sanitize_math_markup(f"بر اساس منبع بازیابی شده:\n{top_text[:400]}...")
        return "اطلاعاتی درباره این پرسش در منابع در دسترس یافت نشد."

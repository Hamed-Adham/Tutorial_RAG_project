"""
Stage 9: Grounded LLM Response Synthesizer

Produces grounded answers from retrieved context using a configurable LLM
(GPT-4o via OpenAI-compatible gateway, or local Llama-3.1-8B).

Key behaviours:
- **Grounding constraint**: The system prompt forbids the LLM from
  using information outside the provided context.
- **"I don't know" fallback**: If context is empty or the LLM cannot
  answer from context, returns a polite refusal.
- **Citation injection**: Appends page/bbox citations assembled by
  ``CitationFormatter`` to every grounded answer.
- **Offline fallback**: Returns a mock answer if no LLM is configured,
  so unit tests work without API keys.

Constitution v1.2.0:
- Three-layer decoupled: no FastAPI/Next.js imports.
- Local Nine Router gateway (http://localhost:20128/v1) is the default.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from rag_core.postprocessing.citations import CitationFormatter

logger = logging.getLogger(__name__)


from rag_core.generation.prompts import (
    SYSTEM_PROMPT_GROUNDED,
    ANSWER_NOT_FOUND_EN as ANSWER_NOT_FOUND,
)

# ---------------------------------------------------------------------------
# Synthesis result
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    """Output of the response synthesizer."""

    answer: str
    """Generated answer text (with inline citations appended)."""

    is_fallback: bool
    """True if the synthesizer returned the "I don't know" fallback."""

    latency_seconds: float
    """Wall-clock time for the LLM call."""

    model: str = ""
    """LLM model identifier used for generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    citations_text: str = ""
    """Formatted citation block appended after the answer."""

    raw_completion: str = ""
    """Raw text completion returned by the LLM."""

    full_prompt: str = ""
    """Full assembled prompt sent to the LLM (system + user with context)."""

    system_prompt: str = ""
    """System prompt instruction applied during synthesis."""

    context_block: str = ""
    """Context block embedded within the user message."""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _build_context_block(chunks: List[Any], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into a numbered <context> block for the prompt.

    Each chunk is prefixed with its breadcrumb and page number.
    Context is truncated at *max_chars* to stay within the LLM context window.
    """
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            page = chunk.get("page", 0)
            breadcrumb = chunk.get("breadcrumb", "")
        else:
            text = getattr(chunk, "text", "")
            page = getattr(chunk, "page", 0)
            breadcrumb = getattr(chunk, "breadcrumb", "")

        entry = f"[{i}] (p.{page}) {breadcrumb}\n{text}" if breadcrumb else f"[{i}] (p.{page})\n{text}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# ResponseSynthesizer
# ---------------------------------------------------------------------------


class ResponseSynthesizer:
    """
    Grounded LLM response synthesizer.

    Connects to an OpenAI-compatible REST endpoint (local or remote).
    Uses the ``openai`` SDK with a configurable ``base_url`` so it works
    with the Nine Router local gateway (``http://localhost:20128/v1``)
    as well as direct OpenAI/Azure endpoints.

    Args:
        model:          LLM model name (default ``"gpt-4o-mini"``).
        base_url:       OpenAI-compatible API base URL.
        api_key:        API key (can be ``"none"`` for local gateways).
        max_tokens:     Maximum tokens for the completion.
        temperature:    Sampling temperature (0.0 = deterministic).
        context_max_chars: Max characters to include from context.
        tracker:        Optional ``LangfuseTracker`` for span instrumentation.

    Example::

        synthesizer = ResponseSynthesizer(
            model="llama-3.1-8b",
            base_url="http://localhost:20128/v1",
            api_key="none",
        )
        result = synthesizer.synthesize(query="What is ATP?", chunks=retrieved_chunks)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        context_max_chars: int = 6000,
        tracker: Optional[Any] = None,
    ) -> None:
        from rag_core.config import Settings
        settings = Settings()

        self._model = model or settings.llm_model or "open-code"
        self._base_url = base_url or settings.nine_router_base_url or "http://localhost:20128/v1"
        self._api_key = api_key if (api_key and api_key != "none") else (settings.nine_router_api_key or "none")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._context_max_chars = context_max_chars
        self._tracker = tracker
        self._citation_formatter = CitationFormatter()
        self._client: Optional[Any] = None  # Lazy-init openai.OpenAI
        self._async_client: Optional[Any] = None  # Lazy-init openai.AsyncOpenAI

    def _get_async_client(self) -> Any:
        """Lazily initialise the AsyncOpenAI client."""
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._async_client

    async def synthesize_stream(
        self,
        query: str,
        chunks: List[Any],
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream the grounded response token by token as an asynchronous generator.
        Yields individual text tokens as they arrive from the LLM.
        """
        # Handle empty context — immediate fallback
        if not chunks:
            yield ANSWER_NOT_FOUND
            return

        context_block = _build_context_block(chunks, self._context_max_chars)
        sys_prompt = system_prompt or SYSTEM_PROMPT_GROUNDED
        if any("؀" <= ch <= "ۿ" for ch in query):
            user_message = f"<context>\n{context_block}\n</context>\n\nپرسش: {query}\nپاسخ مستند به زبان فارسی:"
        else:
            user_message = f"<context>\n{context_block}\n</context>\n\nQuestion: {query}"

        try:
            client = self._get_async_client()
            stream = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.warning("Streaming synthesis failed: %s, falling back to static text", exc)
            yield f"بر اساس متون استخراج شده:\n{context_block[:300]}..."

    def synthesize(
        self,
        query: str,
        chunks: List[Any],
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Generate a grounded answer for *query* using *chunks* as context.

        Args:
            query:         User's natural language question.
            chunks:        Retrieved ``RetrievedChunk``-compatible objects or dicts.
            system_prompt: Override the default grounding system prompt.
            session_id:    Optional session ID for Langfuse tracing.

        Returns:
            ``SynthesisResult`` with answer text, token counts, and latency.
        """
        t_start = time.perf_counter()

        # Build citations
        citation_bundle = self._citation_formatter.format(chunks)

        # Handle empty context — immediate fallback
        if not chunks:
            return SynthesisResult(
                answer=ANSWER_NOT_FOUND,
                is_fallback=True,
                latency_seconds=time.perf_counter() - t_start,
                model=self._model,
                citations_text="",
                raw_completion=ANSWER_NOT_FOUND,
                full_prompt="",
                system_prompt=system_prompt or SYSTEM_PROMPT_GROUNDED,
                context_block="",
            )

        context_block = _build_context_block(chunks, self._context_max_chars)
        sys_prompt = system_prompt or SYSTEM_PROMPT_GROUNDED
        if any("\u0600" <= ch <= "\u06ff" for ch in query):
            user_message = f"<context>\n{context_block}\n</context>\n\nپرسش: {query}\nپاسخ مستند به زبان فارسی:"
        else:
            user_message = f"<context>\n{context_block}\n</context>\n\nQuestion: {query}"

        full_prompt = f"System:\n{sys_prompt}\n\nUser:\n{user_message}"

        # Attempt LLM call
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            answer_text = response.choices[0].message.content.strip()
            usage = response.usage

            latency = time.perf_counter() - t_start

            # Detect "I don't know" style fallback in model output (English or Persian)
            is_fallback = (
                ANSWER_NOT_FOUND.lower()[:30] in answer_text.lower()
                or "اطلاعات کافی برای پاسخ" in answer_text
                or "اطلاعاتی درباره این پرسش در کتاب یافت نشد" in answer_text
                or "اطلاعات کافی در بافت ارائه نشده" in answer_text
            )

            # Append citations to the answer
            full_answer = answer_text
            if citation_bundle.inline_text and not is_fallback:
                full_answer = f"{answer_text}\n\n**Sources:**\n{citation_bundle.inline_text}"

            logger.info(
                "Synthesizer: model=%s latency=%.2fs tokens=%s fallback=%s",
                self._model,
                latency,
                getattr(usage, "total_tokens", "?"),
                is_fallback,
            )

            return SynthesisResult(
                answer=full_answer,
                is_fallback=is_fallback,
                latency_seconds=latency,
                model=self._model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                citations_text=citation_bundle.inline_text,
                raw_completion=answer_text,
                full_prompt=full_prompt,
                system_prompt=sys_prompt,
                context_block=context_block,
            )

        except ImportError:
            logger.warning(
                "openai package not installed — returning mock synthesis result."
            )
            return self._mock_result(
                query=query,
                citations_text=citation_bundle.inline_text,
                t_start=t_start,
                sys_prompt=sys_prompt,
                user_msg=user_message,
                ctx_block=context_block,
            )

        except Exception as exc:
            logger.error("Synthesizer LLM call failed: %s", exc, exc_info=True)
            # Graceful fallback: return "I don't know" rather than crashing
            latency = time.perf_counter() - t_start
            return SynthesisResult(
                answer=ANSWER_NOT_FOUND,
                is_fallback=True,
                latency_seconds=latency,
                model=self._model,
                raw_completion=ANSWER_NOT_FOUND,
                full_prompt=full_prompt,
                system_prompt=sys_prompt,
                context_block=context_block,
            )

    def _get_client(self) -> Any:
        """Lazily initialise the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def generate_raw(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Generate a short raw text completion for query rewriting, expansion, or decomposition."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("generate_raw failed: %s", exc)
            return None

    def _mock_result(
        self,
        query: str,
        citations_text: str,
        t_start: float,
        sys_prompt: str = "",
        user_msg: str = "",
        ctx_block: str = "",
    ) -> SynthesisResult:
        """Return a deterministic mock result for tests without an LLM."""
        raw_ans = f"[MOCK ANSWER] Based on the retrieved context, here is a synthesised response for: '{query}'"
        answer = f"{raw_ans}\n\n**Sources:**\n{citations_text}" if citations_text else raw_ans
        return SynthesisResult(
            answer=answer,
            is_fallback=False,
            latency_seconds=time.perf_counter() - t_start,
            model="mock",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            citations_text=citations_text,
            raw_completion=raw_ans,
            full_prompt=f"System:\n{sys_prompt}\n\nUser:\n{user_msg}" if sys_prompt else query,
            system_prompt=sys_prompt,
            context_block=ctx_block,
        )

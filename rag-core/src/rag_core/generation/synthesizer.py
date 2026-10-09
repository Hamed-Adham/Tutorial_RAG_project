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
from typing import Any, Dict, List, Optional

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
        model: str = "gpt-4o-mini",
        base_url: str = "http://localhost:20128/v1",
        api_key: str = "none",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        context_max_chars: int = 6000,
        tracker: Optional[Any] = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._context_max_chars = context_max_chars
        self._tracker = tracker
        self._citation_formatter = CitationFormatter()
        self._client: Optional[Any] = None  # Lazy-init openai.OpenAI

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
            )

        context_block = _build_context_block(chunks, self._context_max_chars)
        sys_prompt = system_prompt or SYSTEM_PROMPT_GROUNDED
        user_message = f"<context>\n{context_block}\n</context>\n\nQuestion: {query}"

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

            # Detect "I don't know" style fallback in model output
            is_fallback = ANSWER_NOT_FOUND.lower()[:30] in answer_text.lower()

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
            )

        except ImportError:
            logger.warning(
                "openai package not installed — returning mock synthesis result."
            )
            return self._mock_result(query, citation_bundle.inline_text, t_start)

        except Exception as exc:
            logger.error("Synthesizer LLM call failed: %s", exc, exc_info=True)
            # Graceful fallback: return "I don't know" rather than crashing
            latency = time.perf_counter() - t_start
            return SynthesisResult(
                answer=ANSWER_NOT_FOUND,
                is_fallback=True,
                latency_seconds=latency,
                model=self._model,
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

    def _mock_result(
        self, query: str, citations_text: str, t_start: float
    ) -> SynthesisResult:
        """Return a deterministic mock result for tests without an LLM."""
        answer = f"[MOCK ANSWER] Based on the retrieved context, here is a synthesised response for: '{query}'"
        if citations_text:
            answer += f"\n\n**Sources:**\n{citations_text}"
        return SynthesisResult(
            answer=answer,
            is_fallback=False,
            latency_seconds=time.perf_counter() - t_start,
            model="mock",
            citations_text=citations_text,
        )

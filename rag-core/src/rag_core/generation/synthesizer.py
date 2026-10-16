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
import os
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

    q_and_a_summary: Optional[str] = None
    """Structured summary of the Q&A turn for conversational memory (Feature 006)."""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _build_context_block(chunks: List[Any], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into a numbered <context> block for the prompt.

    Each chunk is prefixed with its clean Persian chapter, page number, and breadcrumb.
    Context is truncated at *max_chars* to stay within the LLM context window.
    """
    from rag_core.postprocessing.citations import DOC_TO_CHAPTER, _extract_start_page_from_filename

    parts = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            page = chunk.get("page", 0)
            breadcrumb = chunk.get("breadcrumb", "")
            doc_id = chunk.get("doc_id", "") or chunk.get("filename", "")
        else:
            text = getattr(chunk, "text", "")
            page = getattr(chunk, "page", 0)
            breadcrumb = getattr(chunk, "breadcrumb", "")
            doc_id = getattr(chunk, "doc_id", "") or getattr(chunk, "filename", "")

        category = chunk.get("category", "") if isinstance(chunk, dict) else getattr(chunk, "category", "")
        row = chunk.get("row", 0) if isinstance(chunk, dict) else getattr(chunk, "row", 0)
        source_domain = chunk.get("source_domain", "") if isinstance(chunk, dict) else getattr(chunk, "source_domain", "")
        if not source_domain and (doc_id == "administrative_faq" or category or row):
            source_domain = "administrative_faq"
        chapter = chunk.get("chapter", 0) if isinstance(chunk, dict) else getattr(chunk, "chapter", 0)

        fa_map = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

        if source_domain == "administrative_faq" or category:
            if category and row:
                loc_label = f"صندوق نوآوری – دسته: {category} – ردیف: {row}"
            elif category:
                loc_label = f"صندوق نوآوری – دسته: {category}"
            elif row:
                loc_label = f"صندوق نوآوری – ردیف: {row}"
            else:
                loc_label = "پرسش‌های متداول صندوق نوآوری"
        else:
            pdf_key = doc_id if doc_id.endswith(".pdf") else f"{doc_id}.pdf"
            ch_info = DOC_TO_CHAPTER.get(pdf_key)
            start_page = _extract_start_page_from_filename(pdf_key)
            if ch_info:
                ch_num, ch_title = ch_info
                book_p = start_page + page - 1 if page else start_page
                ch_fa = str(ch_num).translate(fa_map)
                book_p_fa = str(book_p).translate(fa_map)
                loc_label = f"فصل {ch_fa} ({ch_title})، صفحه {book_p_fa}" if page else f"فصل {ch_fa} ({ch_title})"
            elif chapter:
                ch_fa = str(chapter).translate(fa_map)
                p_fa = str(page).translate(fa_map) if page else ""
                loc_label = f"فصل {ch_fa}، صفحه {p_fa}" if page else f"فصل {ch_fa}"
            else:
                loc_label = f"صفحه {page}" if page else "کتاب درسی"

        header = f"[{i}] ({loc_label}) - {breadcrumb}" if breadcrumb else f"[{i}] ({loc_label})"
        entry = f"{header}\n{text}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n\n---\n\n".join(parts)

def sanitize_math_markup(text: str) -> str:
    """
    Strip and convert raw LaTeX syntax and messy formula artifacts into clean,
    readable plain text, unicode symbols, and bold markdown.
    """
    if not text:
        return ""
    import re

    # 1. Convert LaTeX text/style wrappers
    t = re.sub(r"\\(?:text|mathrm|operatorname)\{([^}]+)\}", r"\1", text)
    t = re.sub(r"\\mathbf\{([^}]+)\}", r"**\1**", t)
    t = re.sub(r"\\mathit\{([^}]+)\}", r"*\1*", t)

    # 2. Fractions \frac{A}{B} -> A / B
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1 / \2)", t)
    t = re.sub(r"=\s*\(([^\/()]+)\s*\/\s*([^\/()]+)\)", r"= \1 / \2", t)

    # 3. Greek and scientific symbols
    symbols = {
        r"\\rho\b": "ρ",
        r"\\Delta\b": "Δ",
        r"\\delta\b": "δ",
        r"\\pi\b": "π",
        r"\\alpha\b": "α",
        r"\\beta\b": "β",
        r"\\gamma\b": "γ",
        r"\\theta\b": "θ",
        r"\\lambda\b": "λ",
        r"\\mu\b": "μ",
        r"\\sigma\b": "σ",
        r"\\omega\b": "ω",
        r"\\times\b": "×",
        r"\\cdot\b": "·",
        r"\\approx\b": "≈",
        r"\\pm\b": "±",
        r"\\leq?\b": "≤",
        r"\\geq?\b": "≥",
        r"\\neq?\b": "≠",
        r"\\degree\b": "°",
    }
    for pat, rep in symbols.items():
        t = re.sub(pat, rep, t)

    # 4. Superscripts
    t = t.replace("^2", "²").replace("^3", "³").replace("^{2}", "²").replace("^{3}", "³")

    # 5. Block math $$...$$
    t = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: f" **{m.group(1).strip()}** " if m.group(1).strip() else "", t)

    # 6. Inline math $...$ or $...\$
    def _clean_inline(m: re.Match) -> str:
        c = m.group(1).replace("\\", "").strip()
        if c == "rho":
            c = "ρ"
        elif c == "pi":
            c = "π"
        return f"**{c}**" if c else ""

    t = re.sub(r"(?:\$|\\\$)+([^\n$]+?)(?:\\*\$)+", _clean_inline, t)

    # 7. Stray backslashes
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: "ρ" if m.group(1) == "rho" else ("π" if m.group(1) == "pi" else m.group(1)), t)
    t = re.sub(r"\\([$()\[\]{}*\/=+\-])", r"\1", t)

    # 8. Fix floating lone bullets
    t = re.sub(r"(?:^|\n)\s*[•\-\*]\s*\n+(?=[^\n])", r"\n• ", t)
    t = re.sub(r"(?:^|\n)\s*[•\-\*]\s*(?=\n|$)", r"", t)

    # 9. Normalize spaces around division slashes between digits and clean excessive spaces
    t = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1 / \2", t)
    t = re.sub(r"[ \t]{2,}", " ", t)

    return t


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
        max_tokens: int = 3072,
        temperature: float = 0.0,
        context_max_chars: int = 6000,
        tracker: Optional[Any] = None,
    ) -> None:
        from rag_core.config import Settings
        settings = Settings()

        self._model = model or settings.llm_model or "RAG-synthesizer"
        self._base_url = base_url or settings.nine_router_base_url or "http://localhost:20128/v1"
        self._api_key = api_key if (api_key and api_key != "none") else (settings.nine_router_api_key or "none")
        self._max_tokens = int(os.getenv("SYNTHESIZER_MAX_TOKENS", str(max_tokens)))
        self._temperature = temperature
        self._context_max_chars = context_max_chars
        self._tracker = tracker
        self.last_full_prompt: str = ""
        self.last_context_block: str = ""
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
                timeout=30.0,
                max_retries=1,
            )
        return self._async_client

    async def synthesize_stream(
        self,
        query: str,
        chunks: List[Any],
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream the grounded response token by token as an asynchronous generator.
        Yields individual text tokens as they arrive from the LLM.
        """
        import asyncio

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

        messages = [{"role": "system", "content": sys_prompt}]
        has_history_in_chunks = any(
            isinstance(c, dict) and (c.get("breadcrumb") == "گفتگوی پیشین" or c.get("doc_id") == "conversation_history")
            for c in chunks
        )
        if history and not has_history_in_chunks:
            for turn in history[-4:]:
                r = turn.get("role", "user")
                c = turn.get("content", "")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_message})

        self.last_context_block = context_block
        self.last_full_prompt = "\n\n".join([f"[{m['role']}]:\n{m['content']}" for m in messages])

        try:
            client = self._get_async_client()
            stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error("Streaming synthesis failed: %s", exc, exc_info=True)
            yield f"⚠️ متأسفانه در فرآیند تولید پاسخ هوشمند خطایی رخ داد ({type(exc).__name__}: {exc}). لطفاً ارتباط با مدل لوکال را بررسی فرمایید."



    @staticmethod
    def parse_answer_and_summary(raw_text: str) -> tuple[str, Optional[str]]:
        """
        Extract clean answer and structured memory summary tag content if present.
        Validates structure using Pydantic TurnMemoryPayload.
        """
        import re
        from rag_core.indexing.science_schema import TurnMemoryPayload
        if not raw_text:
            return "", None

        # 1. Primary explicit tag: <<<MEMORY_SUMMARY_START>>> ... <<<MEMORY_SUMMARY_END>>>
        match_custom = re.search(r"<<<MEMORY_SUMMARY_START>>>([\s\S]*?)(?:<<<MEMORY_SUMMARY_END>>>|$)", raw_text, re.IGNORECASE)
        if match_custom:
            summary = match_custom.group(1).strip()
            summary = re.sub(r"دستورالعمل ثبت خلاصه[^\n]*\n+", "", summary, flags=re.IGNORECASE).strip()
            summary = re.sub(r"توجه بسیار مهم:\s*از خلاصه‌سازی[\s\S]*$", "", summary, flags=re.IGNORECASE).strip()
            clean_answer = re.sub(r"(?:دستورالعمل ثبت خلاصه[^\n]*\n+)?<<<MEMORY_SUMMARY_START>>>[\s\S]*?(?:<<<MEMORY_SUMMARY_END>>>|$)(?:[\s\S]*$)?", "", raw_text, flags=re.IGNORECASE).strip()
            clean_answer = re.sub(r"دستورالعمل ثبت خلاصه[\s\S]*$", "", clean_answer, flags=re.IGNORECASE).strip()
            clean_answer = re.sub(r"توجه بسیار مهم[\s\S]*$", "", clean_answer, flags=re.IGNORECASE).strip()
            try:
                payload = TurnMemoryPayload(answer=clean_answer, summary=summary, has_explicit_summary=True)
                return payload.answer, payload.summary
            except Exception:
                return clean_answer, summary

        # 2. Standard HTML tag: <!-- SUMMARY --> or <!-- TURN_SUMMARY -->
        match_html = re.search(r"<!--\s*(?:TURN_)?SUMMARY\s*-->([\s\S]*?)<!--\s*/(?:TURN_)?SUMMARY\s*-->", raw_text, re.IGNORECASE)
        if match_html:
            summary = match_html.group(1).strip()
            summary = re.sub(r"دستورالعمل ثبت خلاصه[^\n]*\n+", "", summary, flags=re.IGNORECASE).strip()
            summary = re.sub(r"توجه بسیار مهم:\s*از خلاصه‌سازی[\s\S]*$", "", summary, flags=re.IGNORECASE).strip()
            clean_answer = re.sub(r"(?:دستورالعمل ثبت خلاصه[^\n]*\n+)?<!--\s*(?:TURN_)?SUMMARY\s*-->[\s\S]*?<!--\s*/(?:TURN_)?SUMMARY\s*-->[\s\S]*$", "", raw_text, flags=re.IGNORECASE).strip()
            clean_answer = re.sub(r"دستورالعمل ثبت خلاصه[\s\S]*$", "", clean_answer, flags=re.IGNORECASE).strip()
            clean_answer = re.sub(r"توجه بسیار مهم[\s\S]*$", "", clean_answer, flags=re.IGNORECASE).strip()
            try:
                payload = TurnMemoryPayload(answer=clean_answer, summary=summary, has_explicit_summary=True)
                return payload.answer, payload.summary
            except Exception:
                return clean_answer, summary

        # 3. Alternative Persian marker: ### خلاصه / خلاصه گفت‌وگو:
        match_fa = re.search(r"(?:###|---)\s*(?:خلاصه|خلاصه پرسش و پاسخ|خلاصه گفت‌وگو|خلاصه گفتگو)\s*[:\s]*(?:###|---)?\s*[\n]([\s\S]*)$", raw_text)
        if match_fa:
            summary = match_fa.group(1).strip()
            clean_answer = raw_text[:match_fa.start()].strip()
            clean_answer = re.sub(r"دستورالعمل ثبت خلاصه[\s\S]*$", "", clean_answer, flags=re.IGNORECASE).strip()
            clean_answer = re.sub(r"توجه بسیار مهم[\s\S]*$", "", clean_answer, flags=re.IGNORECASE).strip()
            try:
                payload = TurnMemoryPayload(answer=clean_answer, summary=summary, has_explicit_summary=True)
                return payload.answer, payload.summary
            except Exception:
                return clean_answer, summary

        # 4. No tags found: return unmodified raw text and None
        return raw_text, None

    def synthesize(
        self,
        query: str,
        chunks: List[Any],
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> SynthesisResult:
        """
        Generate a grounded answer for *query* using *chunks* as context.

        Args:
            query:         User's natural language question.
            chunks:        Retrieved ``RetrievedChunk``-compatible objects or dicts.
            system_prompt: Override the default grounding system prompt.
            session_id:    Optional session ID for Langfuse tracing.
            history:       Optional prior dialogue turns for context memory.

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
                q_and_a_summary=None,
            )

        context_block = _build_context_block(chunks, self._context_max_chars)
        sys_prompt = system_prompt or SYSTEM_PROMPT_GROUNDED
        if any("\u0600" <= ch <= "\u06ff" for ch in query):
            user_message = f"<context>\n{context_block}\n</context>\n\nپرسش: {query}\nپاسخ مستند به زبان فارسی:"
        else:
            user_message = f"<context>\n{context_block}\n</context>\n\nQuestion: {query}"

        full_prompt = f"System:\n{sys_prompt}\n\nUser:\n{user_message}"

        messages = [{"role": "system", "content": sys_prompt}]
        has_history_in_chunks = any(
            isinstance(c, dict) and (c.get("breadcrumb") == "گفتگوی پیشین" or c.get("doc_id") == "conversation_history")
            for c in chunks
        )
        if history and not has_history_in_chunks:
            for turn in history[-4:]:
                r = turn.get("role", "user")
                c = turn.get("content", "")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_message})

        self.last_context_block = context_block
        self.last_full_prompt = "\n\n".join([f"[{m['role']}]:\n{m['content']}" for m in messages])

        # Attempt LLM call
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            raw_text = response.choices[0].message.content.strip()
            clean_content, parsed_summary = self.parse_answer_and_summary(raw_text)
            answer_text = sanitize_math_markup(clean_content)
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

            summary_val = parsed_summary or (answer_text[:250].strip() if not is_fallback else None)

            return SynthesisResult(
                answer=full_answer,
                is_fallback=is_fallback,
                latency_seconds=latency,
                model=self._model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                citations_text=citation_bundle.inline_text,
                raw_completion=raw_text,
                full_prompt=full_prompt,
                system_prompt=sys_prompt,
                context_block=context_block,
                q_and_a_summary=summary_val,
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
                timeout=30.0,
                max_retries=1,
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

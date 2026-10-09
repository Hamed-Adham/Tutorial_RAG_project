"""
Stage 2: Tiered Parsing Probe Router

Routes each incoming document to the appropriate parsing tier based on
content-type probing:

    Tier 0 (layout-aware) — text-native PDFs: PyMuPDF / Docling
    Tier 1 (OCR)           — scanned/image PDFs: Textract / PaddleOCR  [stub]
    Tier 2 (VLM)           — figures / slides:   Vision LLM captions    [stub]

The router always starts with Tier 0.  If PyMuPDF reports zero text blocks
(i.e., the PDF is scanned), it falls back to Tier 1 automatically.

Constitution v1.2.0 / Law of Parsimony:
- Simplest viable path is taken first (Tier 0 → 1 → 2).
- Tier 1 and Tier 2 stubs raise ``NotImplementedError`` in this milestone.
- Router is unit-testable without any real PDF or LLM.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Optional, Union

from rag_core.parsing.schema import ParsedDocument
from rag_core.parsing.tier0_layout import PlainTextExtractor, PyMuPDFExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier stubs (Tier 1 & Tier 2 placeholders)
# ---------------------------------------------------------------------------


class Tier1OCRExtractor:
    """
    Tier 1 stub: OCR-based extractor for scanned/image PDFs.

    Will integrate Textract / PaddleOCR in a future milestone.
    Raises ``NotImplementedError`` if called directly.
    """

    def parse(
        self,
        source: Union[str, Path, bytes],
        doc_id: str,
        filename: str,
    ) -> ParsedDocument:
        raise NotImplementedError(
            "Tier 1 OCR extractor is not yet implemented. "
            "Use the ParsingRouter which auto-selects the best available tier."
        )


class Tier2VLMExtractor:
    """
    Tier 2 stub: Vision-Language Model captioning for figures and slides.

    Will integrate a multimodal LLM in a future milestone.
    Raises ``NotImplementedError`` if called directly.
    """

    def parse(
        self,
        source: Union[str, Path, bytes],
        doc_id: str,
        filename: str,
    ) -> ParsedDocument:
        raise NotImplementedError(
            "Tier 2 VLM extractor is not yet implemented. "
            "Use the ParsingRouter which auto-selects the best available tier."
        )


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------


def _probe_mime_type(source: Union[str, Path, bytes], filename: str) -> str:
    """
    Detect the MIME type of *source*.

    Uses ``mimetypes`` for file-path sources and magic-byte detection
    for raw bytes. Returns ``"application/octet-stream"`` as fallback.
    """
    if isinstance(source, (str, Path)):
        mime, _ = mimetypes.guess_type(str(source))
        return mime or "application/octet-stream"

    # Raw bytes — check PDF magic bytes
    if isinstance(source, bytes) and source[:4] == b"%PDF":
        return "application/pdf"

    # Guess from filename extension
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _is_text_native_pdf(parsed: ParsedDocument, min_text_blocks: int = 3) -> bool:
    """
    Return True if a parsed PDF document contains enough text blocks
    to be considered text-native (not a scanned image PDF).

    Args:
        parsed: A ``ParsedDocument`` from an initial Tier 0 pass.
        min_text_blocks: Minimum number of content blocks to classify
            the document as text-native (default: 3).
    """
    return len(parsed.content_blocks) >= min_text_blocks


# ---------------------------------------------------------------------------
# Parsing Router
# ---------------------------------------------------------------------------


class ParsingRouter:
    """
    Content-aware parsing probe router.

    Selects the most appropriate parsing tier for each document:

    1. Probe the MIME type from filename or magic bytes.
    2. Attempt Tier 0 (PyMuPDF for PDFs, PlainTextExtractor for txt/md).
    3. If Tier 0 produces fewer than ``min_text_blocks`` text blocks
       (likely a scanned PDF), fall through to Tier 1 (OCR stub).
    4. VLM captions (Tier 2) are appended as separate ``FIGURE`` blocks
       in a future milestone when Tier 2 is implemented.

    Args:
        min_text_blocks: Threshold for text-native PDF detection (default 3).
        force_tier: If set (0, 1, or 2), bypass probing and always use
            the specified tier (useful for testing).

    Example::

        router = ParsingRouter()
        doc = router.parse(source="invoice.pdf", doc_id="doc-001", filename="invoice.pdf")
        print(f"Parsed {len(doc.blocks)} blocks from tier {doc.parser_tier}")
    """

    def __init__(
        self,
        min_text_blocks: int = 3,
        force_tier: Optional[int] = None,
    ) -> None:
        self._min_text_blocks = min_text_blocks
        self._force_tier = force_tier

        self._tier0_pdf = PyMuPDFExtractor()
        self._tier0_text = PlainTextExtractor()
        self._tier1 = Tier1OCRExtractor()
        self._tier2 = Tier2VLMExtractor()

    def parse(
        self,
        source: Union[str, Path, bytes],
        doc_id: str,
        filename: str,
    ) -> ParsedDocument:
        """
        Route *source* to the appropriate parsing tier and return a
        ``ParsedDocument`` with Canonical IR blocks.

        Args:
            source: File path, raw bytes, or pre-read text.
            doc_id: Stable doc_id from the manifest catalog.
            filename: Original filename (used for MIME probing and metadata).

        Returns:
            ``ParsedDocument`` with blocks ordered by reading position.
        """
        if self._force_tier is not None:
            return self._dispatch_tier(self._force_tier, source, doc_id, filename)

        mime = _probe_mime_type(source, filename)
        logger.debug("ParsingRouter: mime=%s filename=%s", mime, filename)

        # --- PDF ---
        if mime == "application/pdf" or filename.lower().endswith(".pdf"):
            try:
                parsed = self._tier0_pdf.parse(source, doc_id, filename)
                if _is_text_native_pdf(parsed, self._min_text_blocks):
                    logger.info(
                        "Tier 0 (PyMuPDF): doc_id=%s blocks=%d pages=%d",
                        doc_id,
                        len(parsed.blocks),
                        parsed.page_count,
                    )
                    return parsed
                else:
                    logger.info(
                        "Tier 0 returned %d blocks — likely scanned PDF. "
                        "Tier 1 OCR not yet implemented; returning Tier 0 result.",
                        len(parsed.blocks),
                    )
                    return parsed
            except ImportError:
                logger.warning(
                    "PyMuPDF not installed — falling back to PlainTextExtractor"
                )
                return self._tier0_text.parse(source, doc_id, filename)

        # --- Plain text / Markdown ---
        if mime in (
            "text/plain",
            "text/markdown",
            "text/html",
            "text/csv",
        ) or filename.lower().endswith((".txt", ".md", ".rst", ".csv")):
            parsed = self._tier0_text.parse(source, doc_id, filename)
            logger.info(
                "Tier 0 (PlainText): doc_id=%s blocks=%d",
                doc_id,
                len(parsed.blocks),
            )
            return parsed

        # --- Unknown: attempt PlainTextExtractor as a last resort ---
        logger.warning(
            "Unrecognised MIME type '%s' for '%s'. "
            "Attempting plain-text extraction.",
            mime,
            filename,
        )
        return self._tier0_text.parse(source, doc_id, filename)

    def _dispatch_tier(
        self,
        tier: int,
        source: Union[str, Path, bytes],
        doc_id: str,
        filename: str,
    ) -> ParsedDocument:
        """Directly dispatch to the specified tier (for testing)."""
        if tier == 0:
            mime = _probe_mime_type(source, filename)
            if mime == "application/pdf" or filename.lower().endswith(".pdf"):
                return self._tier0_pdf.parse(source, doc_id, filename)
            return self._tier0_text.parse(source, doc_id, filename)
        elif tier == 1:
            return self._tier1.parse(source, doc_id, filename)
        elif tier == 2:
            return self._tier2.parse(source, doc_id, filename)
        else:
            raise ValueError(f"Invalid tier: {tier}. Must be 0, 1, or 2.")

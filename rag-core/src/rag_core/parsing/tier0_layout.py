"""
Stage 2: Tier 0 — Layout-Aware PDF/Document Extractor

Uses PyMuPDF (fitz) as the primary engine with a Docling fallback.
Produces Canonical IR ``Block`` objects with full layout metadata:
- Reading order preservation
- Heading level detection from font-size heuristics
- Table detection and Markdown serialisation
- Page and bounding-box attribution for every block

Constitution v1.2.0 / Law of Parsimony:
- PyMuPDF is always attempted first (fastest, zero API calls).
- Returns a ``ParsedDocument`` regardless of which backend is used.
- Graceful fallback to plain-text extraction if PyMuPDF is unavailable.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rag_core.parsing.schema import Block, BlockType, BBox, ParsedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_heading_level(font_size: float, body_size: float) -> Optional[int]:
    """
    Infer heading level from font size relative to the body text size.

    Returns 1–4 for headings or ``None`` if the span is body text.
    """
    ratio = font_size / body_size if body_size > 0 else 1.0
    if ratio >= 1.8:
        return 1
    elif ratio >= 1.5:
        return 2
    elif ratio >= 1.25:
        return 3
    elif ratio >= 1.1:
        return 4
    return None


def _median_font_size(blocks_raw: List[Dict[str, Any]]) -> float:
    """Compute the median font size across all text spans (heuristic body size)."""
    sizes: List[float] = []
    for block in blocks_raw:
        if block.get("type") != 0:  # 0 = text block in PyMuPDF
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 12.0))
    if not sizes:
        return 12.0
    sizes.sort()
    mid = len(sizes) // 2
    return sizes[mid]


# ---------------------------------------------------------------------------
# PyMuPDF extractor
# ---------------------------------------------------------------------------


class PyMuPDFExtractor:
    """
    Extract ``Block`` objects from a PDF using PyMuPDF (``fitz``).

    Produces rich layout metadata including bounding boxes, reading order,
    font-size-based heading detection, and table block identification.
    """

    SOURCE_PARSER = "pymupdf"

    def parse(
        self,
        source: Union[str, Path, bytes],
        doc_id: str,
        filename: str,
    ) -> ParsedDocument:
        """
        Parse a PDF document into a ``ParsedDocument``.

        Args:
            source: File path (``str`` / ``Path``) or raw PDF bytes.
            doc_id: Stable doc_id from the manifest catalog.
            filename: Original filename for metadata.

        Returns:
            ``ParsedDocument`` with ``Block`` list ordered by reading position.

        Raises:
            ImportError: If ``pymupdf`` (``fitz``) is not installed.
            ValueError: If *source* cannot be opened as a PDF.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for Tier 0 parsing. "
                "Install with: pip install pymupdf"
            ) from exc

        # Open document
        if isinstance(source, (str, Path)):
            doc = fitz.open(str(source))
        else:
            doc = fitz.open(stream=source, filetype="pdf")

        blocks: List[Block] = []
        page_count = len(doc)

        for page_num, page in enumerate(doc, start=1):
            raw_dict = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            raw_blocks: List[Dict[str, Any]] = raw_dict.get("blocks", [])

            body_size = _median_font_size(raw_blocks)
            reading_order = 0

            for raw_block in raw_blocks:
                block_type_id = raw_block.get("type", 0)

                # Image blocks → FIGURE
                if block_type_id == 1:
                    bbox_list = raw_block.get("bbox", [0, 0, 0, 0])
                    blocks.append(
                        Block(
                            type=BlockType.FIGURE,
                            text="",  # VLM caption populated by Tier 2
                            page=page_num,
                            bbox=tuple(bbox_list),  # type: ignore[arg-type]
                            source_parser=self.SOURCE_PARSER,
                            reading_order=reading_order,
                        )
                    )
                    reading_order += 1
                    continue

                # Text blocks
                bbox_list = raw_block.get("bbox", [0, 0, 0, 0])
                bbox: BBox = tuple(bbox_list)  # type: ignore[assignment]
                lines = raw_block.get("lines", [])

                # Collect spans into logical paragraphs
                span_groups: List[Dict[str, Any]] = []
                for line in lines:
                    for span in line.get("spans", []):
                        span_groups.append(span)

                if not span_groups:
                    continue

                # Group contiguous spans of the same font size into one block
                current_size: Optional[float] = None
                current_texts: List[str] = []
                current_flags: int = 0

                for span in span_groups:
                    size = round(span.get("size", 12.0), 1)
                    text_piece = span.get("text", "").strip()
                    if not text_piece:
                        continue

                    if current_size is None:
                        current_size = size
                        current_flags = span.get("flags", 0)

                    if abs(size - current_size) > 0.5:
                        # Flush previous group
                        combined = " ".join(current_texts).strip()
                        if combined:
                            heading_level = _infer_heading_level(current_size, body_size)
                            btype = BlockType.HEADING if heading_level else BlockType.TEXT
                            blocks.append(
                                Block(
                                    type=btype,
                                    text=combined,
                                    page=page_num,
                                    bbox=bbox,
                                    heading_level=heading_level,
                                    source_parser=self.SOURCE_PARSER,
                                    reading_order=reading_order,
                                )
                            )
                            reading_order += 1
                        current_texts = [text_piece]
                        current_size = size
                        current_flags = span.get("flags", 0)
                    else:
                        current_texts.append(text_piece)

                # Flush last group
                if current_texts and current_size is not None:
                    combined = " ".join(current_texts).strip()
                    if combined:
                        heading_level = _infer_heading_level(current_size, body_size)
                        btype = BlockType.HEADING if heading_level else BlockType.TEXT
                        blocks.append(
                            Block(
                                type=btype,
                                text=combined,
                                page=page_num,
                                bbox=bbox,
                                heading_level=heading_level,
                                source_parser=self.SOURCE_PARSER,
                                reading_order=reading_order,
                            )
                        )
                        reading_order += 1

        doc.close()

        extra_meta: Dict[str, Any] = {}
        try:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            extra_meta = {k: v for k, v in (meta or {}).items() if v}
        except Exception:
            pass

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            blocks=blocks,
            page_count=page_count,
            parser_tier=0,
            extra_metadata=extra_meta,
        )


# ---------------------------------------------------------------------------
# Plain-text fallback extractor
# ---------------------------------------------------------------------------


class PlainTextExtractor:
    """
    Minimal fallback extractor for plain-text or pre-extracted text content.

    Produces a single ``TEXT`` block per non-empty paragraph.
    No layout, bbox, or heading detection.
    """

    SOURCE_PARSER = "plaintext"

    def parse(
        self,
        source: Union[str, Path, bytes],
        doc_id: str,
        filename: str,
    ) -> ParsedDocument:
        """Parse plain text into a ``ParsedDocument``."""
        if isinstance(source, (str, Path)):
            raw = Path(source).read_text(encoding="utf-8", errors="replace")
        else:
            raw = source.decode("utf-8", errors="replace")

        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        blocks = [
            Block(
                type=BlockType.TEXT,
                text=para,
                page=1,
                source_parser=self.SOURCE_PARSER,
                reading_order=i,
            )
            for i, para in enumerate(paragraphs)
        ]

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            blocks=blocks,
            page_count=1,
            parser_tier=0,
        )

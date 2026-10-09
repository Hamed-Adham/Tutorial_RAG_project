"""
Stage 3: Structure-Aware Hierarchical Chunking

Splits a ``ParsedDocument`` into retrieval chunks by walking the heading
hierarchy and prepending a breadcrumb trail to every chunk:

    "Doc > Section > Subsection (page N): <chunk text>"

This gives every chunk full context about its location in the document,
dramatically improving retrieval precision for deeply nested sections.

Constitution v1.2.0:
- Zero external dependencies (pure Python dataclasses).
- Breadcrumb prepending is mandatory per spec FR-007.
- Chunk size and overlap are configurable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rag_core.parsing.schema import Block, BlockType, ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """
    A single retrieval unit produced by any chunking strategy.

    Common schema shared across hierarchical, parent-child, semantic,
    and contextual chunkers.
    """

    chunk_id: str
    """Deterministic ID: ``{doc_id}_c{index:04d}``."""

    doc_id: str
    """Stable doc_id from the manifest catalog."""

    text: str
    """
    Full chunk text ready for embedding.

    For hierarchical chunks includes the breadcrumb prefix:
    ``"Doc > Section > Sub (page N): <body>"``
    """

    page: int
    """Source page of the first block in this chunk."""

    chunk_index: int
    """0-based position of this chunk within the document."""

    breadcrumb: str = ""
    """Human-readable heading path: ``"Doc > Section > Subsection"``."""

    heading_level: int = 0
    """Depth of the section heading that opens this chunk (0 = document root)."""

    parent_chunk_id: Optional[str] = None
    """Reference to the parent (larger) chunk (used in Parent-Child chunking)."""

    metadata: Dict[str, str] = field(default_factory=dict)
    """Arbitrary string metadata (filename, suite, tenant, etc.)."""

    def to_dict(self) -> Dict:
        """Serialise to a plain dict."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "breadcrumb": self.breadcrumb,
            "heading_level": self.heading_level,
            "parent_chunk_id": self.parent_chunk_id,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Breadcrumb helpers
# ---------------------------------------------------------------------------


def _build_breadcrumb(heading_stack: List[str]) -> str:
    """Join the heading stack into a breadcrumb string."""
    return " > ".join(h for h in heading_stack if h)


def _format_chunk_text(breadcrumb: str, body: str, page: int) -> str:
    """
    Prepend breadcrumb to chunk body.

    Format: ``"<breadcrumb> (page N): <body>"``
    """
    if breadcrumb:
        return f"{breadcrumb} (page {page}): {body}"
    return body


# ---------------------------------------------------------------------------
# HierarchicalChunker
# ---------------------------------------------------------------------------


class HierarchicalChunker:
    """
    Structure-aware chunker that follows the document heading hierarchy.

    Algorithm:
    1. Walk blocks in reading order.
    2. Maintain a heading stack (depth-tracked).
    3. When a HEADING block is encountered, flush the current buffer as
       a chunk and update the heading stack.
    4. Accumulate TEXT/LIST_ITEM/TABLE blocks under the current heading.
    5. If the accumulated text exceeds ``max_chunk_chars``, split mid-section
       (soft split at sentence boundary if possible).
    6. Prepend the breadcrumb trail to every chunk.

    Args:
        max_chunk_chars: Maximum character count per chunk (default 1500).
        overlap_chars:   Character overlap between consecutive chunks in
                         the same section (default 150).
        filename:        Optional document title used as the root breadcrumb.

    Example::

        chunker = HierarchicalChunker(max_chunk_chars=1500)
        chunks = chunker.chunk(parsed_doc)
    """

    def __init__(
        self,
        max_chunk_chars: int = 1500,
        overlap_chars: int = 150,
        filename: Optional[str] = None,
    ) -> None:
        self._max = max_chunk_chars
        self._overlap = overlap_chars
        self._filename = filename

    def chunk(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """
        Chunk *parsed_doc* into retrieval-ready ``Chunk`` objects.

        Args:
            parsed_doc: A ``ParsedDocument`` from any parsing tier.

        Returns:
            Ordered list of ``Chunk`` objects.
        """
        blocks = parsed_doc.sorted_blocks()
        doc_title = self._filename or parsed_doc.filename or "Document"

        # heading_stack[i] = title of heading at depth i+1
        heading_stack: List[str] = [doc_title]
        current_heading_level: int = 0

        buffer_texts: List[str] = []
        buffer_page: int = 1
        chunks: List[Chunk] = []
        chunk_index: int = 0

        def flush(page: int) -> None:
            nonlocal chunk_index
            body = " ".join(buffer_texts).strip()
            if not body:
                return
            breadcrumb = _build_breadcrumb(heading_stack)
            full_text = _format_chunk_text(breadcrumb, body, page)
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed_doc.doc_id}_c{chunk_index:04d}",
                    doc_id=parsed_doc.doc_id,
                    text=full_text,
                    page=page,
                    chunk_index=chunk_index,
                    breadcrumb=breadcrumb,
                    heading_level=current_heading_level,
                    metadata={"filename": parsed_doc.filename},
                )
            )
            chunk_index += 1
            buffer_texts.clear()

        for block in blocks:
            if block.type == BlockType.HEADING and block.text.strip():
                # Flush current buffer before starting new section
                flush(buffer_page)

                level = block.heading_level or 1
                heading_text = block.text.strip()

                # Trim stack to current depth
                while len(heading_stack) > level:
                    heading_stack.pop()
                while len(heading_stack) < level:
                    heading_stack.append("")  # pad missing levels

                if len(heading_stack) == level:
                    heading_stack.append(heading_text)
                else:
                    heading_stack[level] = heading_text

                current_heading_level = level  # type: ignore[assignment]
                buffer_page = block.page
                continue

            if not block.is_content:
                continue

            text_piece = block.text.strip()
            if not text_piece:
                continue

            # Record page of first block in buffer
            if not buffer_texts:
                buffer_page = block.page

            # Mid-section split if buffer is getting too large
            if (
                sum(len(t) for t in buffer_texts) + len(text_piece)
                > self._max
                and buffer_texts
            ):
                flush(buffer_page)
                # Overlap: carry last N chars into new buffer
                if self._overlap and chunks:
                    overlap_text = chunks[-1].text[-self._overlap :]
                    buffer_texts.append(overlap_text)
                buffer_page = block.page

            buffer_texts.append(text_piece)

        # Final flush
        flush(buffer_page)

        logger.info(
            "HierarchicalChunker: doc_id=%s produced %d chunks",
            parsed_doc.doc_id,
            len(chunks),
        )
        return chunks

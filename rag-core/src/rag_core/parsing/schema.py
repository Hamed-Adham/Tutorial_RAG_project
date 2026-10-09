"""
Stage 2: Canonical IR Block Schema

Defines the interchange format produced by every parsing tier.
All downstream stages (chunking, indexing, citation assembly)
work exclusively against this schema — parsers are swappable
without touching any other component.

Key design decisions (Constitution v1.2.0 / Law of Parsimony):
- Plain Python dataclasses: zero external dependencies.
- All fields Optional where layout information may be unavailable
  (e.g. plain-text extraction vs. full PDF layout).
- `bbox` is a 4-tuple (x0, y0, x1, y1) in PDF user-space units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Block type taxonomy
# ---------------------------------------------------------------------------


class BlockType(str, Enum):
    """
    Semantic type of a parsed content block.

    Parsers assign one of these types so downstream stages can apply
    type-specific logic (e.g. table chunking vs. heading hierarchy).
    """

    TEXT = "text"
    """Flowing narrative paragraph."""

    HEADING = "heading"
    """Section heading with an associated level (h1=1, h2=2, …)."""

    TABLE = "table"
    """Tabular data (serialised as Markdown or CSV text in ``text`` field)."""

    FIGURE = "figure"
    """Figure / image block; ``text`` holds an optional caption or VLM output."""

    LIST_ITEM = "list_item"
    """A single item from a bulleted or numbered list."""

    CAPTION = "caption"
    """Caption associated with a figure or table."""

    FOOTER = "footer"
    """Page footer — typically excluded from retrieval chunks."""

    HEADER = "header"
    """Page header — typically excluded from retrieval chunks."""

    CODE = "code"
    """Source code block."""

    EQUATION = "equation"
    """Mathematical equation (LaTeX or plain-text representation)."""

    UNKNOWN = "unknown"
    """Fallback for unrecognised content."""


# ---------------------------------------------------------------------------
# Bounding box type alias
# ---------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]
"""
PDF bounding box as ``(x0, y0, x1, y1)`` in PDF user-space units (points).

Origin is at the **bottom-left** of the page (PDF convention).
Convert to pixel coordinates for UI rendering by multiplying with DPI/72.
"""


# ---------------------------------------------------------------------------
# Canonical IR Block
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """
    The fundamental unit of the Canonical Intermediate Representation.

    Every parsing tier (Tier 0 layout, Tier 1 OCR, Tier 2 VLM) produces
    a list of ``Block`` objects.  Downstream stages consume only ``Block``
    lists — they never call parser code directly.

    Fields labelled **Required** must always be set by parsers.
    Fields labelled **Optional** may be ``None`` when unavailable.
    """

    # ------------------------------------------------------------------ #
    # Required fields
    # ------------------------------------------------------------------ #

    type: BlockType
    """Semantic type of this content block. Required."""

    text: str
    """
    Textual content of the block.

    For ``TABLE`` blocks, this is a Markdown/CSV serialisation.
    For ``FIGURE`` blocks, this is the caption or VLM-generated description.
    Required (use empty string ``""`` if genuinely empty).
    """

    page: int
    """
    1-indexed page number where this block originates.

    For non-paginated sources (HTML, plain text) use ``1``.
    Required.
    """

    # ------------------------------------------------------------------ #
    # Optional fields
    # ------------------------------------------------------------------ #

    bbox: Optional[BBox] = None
    """
    Bounding box ``(x0, y0, x1, y1)`` in PDF user-space points.

    Available from Tier 0 (PyMuPDF / Docling) and Tier 1 (OCR).
    ``None`` for sources without spatial layout (plain text, HTML).
    """

    heading_level: Optional[int] = None
    """
    Heading depth for ``BlockType.HEADING`` blocks (1=h1, 2=h2, …).

    ``None`` for all other block types.
    """

    confidence: Optional[float] = None
    """
    OCR confidence score (0.0–1.0) from Tier 1 parsers.

    ``None`` for native-text (Tier 0) or VLM (Tier 2) sources.
    """

    language: Optional[str] = None
    """
    Detected language code (e.g. ``"fa"`` for Persian, ``"en"`` for English).

    ``None`` when language detection was not performed.
    """

    reading_order: Optional[int] = None
    """
    0-indexed position of this block within the page's reading order.

    Used to sort blocks when layout analysis produces them out of order.
    ``None`` when reading order is already implied by list position.
    """

    source_parser: str = "unknown"
    """
    Identifier of the parser tier that produced this block.

    Convention:
    - ``"pymupdf"`` / ``"docling"`` for Tier 0.
    - ``"textract"`` / ``"paddleocr"`` for Tier 1.
    - ``"vlm"`` for Tier 2.
    - ``"naive"`` for the naive baseline text extractor.
    """

    extra: Dict[str, Any] = field(default_factory=dict)
    """
    Arbitrary key/value bag for parser-specific metadata that does not
    fit the standard schema (e.g. table column headers, figure alt-text).
    """

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def is_structural(self) -> bool:
        """Return True for block types that define document structure."""
        return self.type in (
            BlockType.HEADING,
            BlockType.HEADER,
            BlockType.FOOTER,
        )

    @property
    def is_content(self) -> bool:
        """Return True for block types that carry retrievable content."""
        return self.type in (
            BlockType.TEXT,
            BlockType.TABLE,
            BlockType.FIGURE,
            BlockType.LIST_ITEM,
            BlockType.CAPTION,
            BlockType.CODE,
            BlockType.EQUATION,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "type": self.type.value,
            "text": self.text,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
            "heading_level": self.heading_level,
            "confidence": self.confidence,
            "language": self.language,
            "reading_order": self.reading_order,
            "source_parser": self.source_parser,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        """Deserialise from a plain dict."""
        bbox_raw = data.get("bbox")
        bbox: Optional[BBox] = tuple(bbox_raw) if bbox_raw else None  # type: ignore[assignment]
        return cls(
            type=BlockType(data["type"]),
            text=data["text"],
            page=data["page"],
            bbox=bbox,
            heading_level=data.get("heading_level"),
            confidence=data.get("confidence"),
            language=data.get("language"),
            reading_order=data.get("reading_order"),
            source_parser=data.get("source_parser", "unknown"),
            extra=data.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# Parsed Document (collection of Blocks)
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """
    The complete Canonical IR output for a single source document.

    Wraps a list of ``Block`` objects with document-level metadata.
    Produced by every parser tier and consumed by the chunking stage.
    """

    doc_id: str
    """Stable doc_id from the manifest catalog."""

    filename: str
    """Original source filename."""

    blocks: List[Block] = field(default_factory=list)
    """Ordered list of parsed content blocks."""

    page_count: int = 0
    """Total number of pages (0 for non-paginated sources)."""

    parser_tier: int = 0
    """Parser tier that produced this document (0=layout, 1=OCR, 2=VLM)."""

    language: Optional[str] = None
    """Dominant language detected across the document."""

    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    """Document-level metadata (title, author, creation date, etc.)."""

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    @property
    def content_blocks(self) -> List[Block]:
        """Return only blocks that carry retrievable content."""
        return [b for b in self.blocks if b.is_content]

    @property
    def headings(self) -> List[Block]:
        """Return all heading blocks ordered by reading position."""
        return [b for b in self.blocks if b.type == BlockType.HEADING]

    def blocks_on_page(self, page: int) -> List[Block]:
        """Return all blocks on the given 1-indexed page number."""
        return [b for b in self.blocks if b.page == page]

    def sorted_blocks(self) -> List[Block]:
        """
        Return blocks sorted by (page, reading_order).

        Blocks without ``reading_order`` are sorted last within their page.
        """
        return sorted(
            self.blocks,
            key=lambda b: (b.page, b.reading_order if b.reading_order is not None else 9999),
        )

    def total_text_length(self) -> int:
        """Return the total character count across all content blocks."""
        return sum(len(b.text) for b in self.content_blocks)

    def to_plain_text(self, separator: str = "\n\n") -> str:
        """Concatenate all content block texts into a single string."""
        return separator.join(b.text for b in self.content_blocks if b.text.strip())

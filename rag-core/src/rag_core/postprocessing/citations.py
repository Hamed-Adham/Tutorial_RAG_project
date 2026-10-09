"""
Stage 8: Page and Bounding-Box Source Citation Formatter

Assembles citations from retrieved chunks, grouping them by document
and page number.  Each citation carries the source filename, page, and
optionally the bounding-box coordinates for UI highlighting.

Citation format appended to synthesised answers::

    [1] hr_handbook_2026.pdf – page 14
    [2] physics_textbook.pdf – pages 3, 7

Constitution v1.2.0 / Law of Parsimony:
- Pure Python dataclasses; zero external dependencies.
- Compatible with any retrieval result that provides ``doc_id``,
  ``page``, and optional ``bbox`` fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Citation types
# ---------------------------------------------------------------------------


@dataclass
class PageCitation:
    """A single page-level citation extracted from a retrieved chunk."""

    doc_id: str
    """Stable doc_id from the manifest catalog."""

    filename: str
    """Original document filename (for display)."""

    page: int
    """1-indexed source page number."""

    bbox: Optional[Tuple[float, float, float, float]] = None
    """
    Optional bounding box ``(x0, y0, x1, y1)`` in PDF user-space points.
    Present when the retrieval payload includes layout metadata.
    """

    chunk_id: str = ""
    """Reference to the originating chunk."""

    score: float = 0.0
    """Retrieval similarity score of the source chunk."""


@dataclass
class CitationBundle:
    """
    Complete citation set for a synthesised answer.

    Groups ``PageCitation`` objects by document and assembles a
    human-readable inline citation string.
    """

    citations: List[PageCitation] = field(default_factory=list)
    """All individual page citations, ordered by retrieval rank."""

    inline_text: str = ""
    """
    Ready-to-append citation block, e.g.::

        [1] hr_handbook_2026.pdf – page 14
        [2] physics_textbook.pdf – pages 3, 7
    """

    def __post_init__(self) -> None:
        if self.citations and not self.inline_text:
            self.inline_text = _format_inline(self.citations)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_inline(citations: List[PageCitation]) -> str:
    """
    Build a numbered citation block grouped by document using strict Markdown links.

    Documents are listed in the order their first citation appears.
    Pages within a document are deduplicated and sorted.
    """
    # Group pages by filename (preserving order)
    seen: Dict[str, List[int]] = {}
    order: List[str] = []

    for c in citations:
        key = c.filename or c.doc_id
        if key not in seen:
            seen[key] = []
            order.append(key)
        if c.page not in seen[key]:
            seen[key].append(c.page)

    lines = []
    for i, key in enumerate(order, start=1):
        pages = sorted(seen[key])
        if len(pages) == 1:
            p = pages[0]
            lines.append(f"[{i}] [{key} – page {p}]({key}#page={p})")
        else:
            page_links = ", ".join(f"[{p}]({key}#page={p})" for p in pages)
            lines.append(f"[{i}] [{key}]({key}#page={pages[0]}) – pages {page_links}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CitationFormatter
# ---------------------------------------------------------------------------


class CitationFormatter:
    """
    Assemble source citations from a list of retrieved chunks.

    Accepts any list of objects (or dicts) that expose:
    - ``doc_id`` / ``"doc_id"``
    - ``page`` / ``"page"``
    - ``chunk_id`` / ``"chunk_id"``
    - ``score`` / ``"score"``
    - ``metadata`` / ``"metadata"`` (optional dict with ``"filename"`` key)
    - ``bbox`` / ``"bbox"`` (optional bounding box)

    Example::

        formatter = CitationFormatter()
        bundle = formatter.format(retrieved_chunks)
        answer_with_citations = answer + "\\n\\n**Sources:**\\n" + bundle.inline_text
    """

    def format(
        self,
        chunks: List[Any],
        max_citations: int = 5,
    ) -> CitationBundle:
        """
        Build a ``CitationBundle`` from retrieved chunk objects or dicts.

        Args:
            chunks:          List of ``RetrievedChunk``-compatible objects or dicts.
            max_citations:   Maximum number of citations to include (default 5).

        Returns:
            ``CitationBundle`` with individual citations and formatted inline text.
        """
        citations: List[PageCitation] = []

        for chunk in chunks[:max_citations]:
            doc_id, page, chunk_id, score, filename, bbox = self._extract_fields(chunk)
            citations.append(
                PageCitation(
                    doc_id=doc_id,
                    filename=filename,
                    page=page,
                    bbox=bbox,
                    chunk_id=chunk_id,
                    score=score,
                )
            )

        inline = _format_inline(citations) if citations else ""
        logger.debug("CitationFormatter: built %d citations", len(citations))
        return CitationBundle(citations=citations, inline_text=inline)

    def format_for_prompt(
        self,
        chunks: List[Any],
        max_citations: int = 5,
    ) -> str:
        """
        Return the citation block as a plain string for prompt injection.

        Convenience wrapper around ``format()`` that returns only the
        ``inline_text`` field.
        """
        return self.format(chunks, max_citations=max_citations).inline_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_fields(
        chunk: Any,
    ) -> Tuple[str, int, str, float, str, Optional[Tuple]]:
        """Extract citation fields from a chunk object or dict."""
        if isinstance(chunk, dict):
            doc_id = chunk.get("doc_id", "")
            page = int(chunk.get("page", 0))
            chunk_id = chunk.get("chunk_id", "")
            score = float(chunk.get("score", 0.0))
            meta = chunk.get("metadata", {}) or {}
            bbox_raw = chunk.get("bbox")
        else:
            doc_id = getattr(chunk, "doc_id", "")
            page = int(getattr(chunk, "page", 0))
            chunk_id = getattr(chunk, "chunk_id", "")
            score = float(getattr(chunk, "score", 0.0))
            meta = getattr(chunk, "metadata", {}) or {}
            bbox_raw = getattr(chunk, "bbox", None)

        filename = meta.get("filename", "") if isinstance(meta, dict) else ""
        if not filename:
            filename = doc_id  # fall back to doc_id if filename not in metadata

        bbox: Optional[Tuple] = tuple(bbox_raw) if bbox_raw else None
        return doc_id, page, chunk_id, score, filename, bbox


# ---------------------------------------------------------------------------
# Citation Provenance & Verification (US4)
# ---------------------------------------------------------------------------

import re


@dataclass
class CitationAuditResult:
    """Audit outcome for source citations against gold reference chunks."""
    precision: float
    recall: float
    hallucinated_citations: List[str] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)
    matched_citations: List[str] = field(default_factory=list)


def parse_citations_from_text(text: str) -> List[str]:
    """
    Extracts citation chunk identifiers from answer text.
    Strictly requires Markdown link format per Clarification Q-3:
    - '[Document Title](filename.pdf#page=4)' -> 'filename.pdf#page=4'
    - '[Document Title](file:///path/to/doc.pdf#page=5)' -> 'doc.pdf#page=5'
    - '[FAQ Title](faq_qa_pairs.csv#row=42)' -> 'faq_qa_pairs.csv#row=42'
    - '[FAQ Title](faq_042)' -> 'faq_042'

    Plain bracketed references (e.g. '[1]', '[faq_042]', '[منبع: ...]')
    and bare unlinked URIs are treated as unverified and excluded.
    """
    if not text:
        return []

    citations: List[str] = []

    # Match all Markdown links: [anchor](target)
    link_pattern = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
    for _anchor, target in link_pattern:
        target_clean = target.strip().replace("\\", "/")
        # 1. Target with #page=N or #row=N fragment
        m_frag = re.search(r"([^/\s)]+?)#(page=\d+|row=\d+)", target_clean)
        if m_frag:
            doc_file = m_frag.group(1).strip()
            fragment = m_frag.group(2).strip()
            citation_id = f"{doc_file}#{fragment}"
            if citation_id not in citations:
                citations.append(citation_id)
            continue

        # 2. Target with catalog ID token (e.g. faq_042 or CH01-SEC-0001)
        m_cat = re.search(r"\b(faq_\d{3}|CH\d{2}-SEC-\d{4})\b", target_clean)
        if m_cat:
            cat_id = m_cat.group(1).strip()
            if cat_id not in citations:
                citations.append(cat_id)

    return citations


def _chunk_matches(a: str, b: str) -> bool:
    """Helper to check if cited identifier matches gold reference chunk."""
    if not a or not b:
        return False
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if a_norm == b_norm:
        return True
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return False


def evaluate_citation_provenance(
    cited_chunks: Sequence[str],
    gold_chunks: Sequence[str]
) -> CitationAuditResult:
    """
    Computes Citation Precision and Citation Recall.
    - Precision: proportion of cited chunks that are valid gold references.
    - Recall: proportion of gold reference chunks that were cited.
    """
    cited_list = [c.strip() for c in cited_chunks if c.strip()]
    gold_list = [g.strip() for g in gold_chunks if g.strip()]

    if not cited_list and not gold_list:
        return CitationAuditResult(precision=1.0, recall=1.0)
    if not cited_list and gold_list:
        return CitationAuditResult(
            precision=0.0,
            recall=0.0,
            missing_citations=gold_list
        )
    if cited_list and not gold_list:
        return CitationAuditResult(
            precision=0.0,
            recall=1.0,
            hallucinated_citations=cited_list
        )

    matched_cited = []
    hallucinated = []
    for c in cited_list:
        if any(_chunk_matches(c, g) for g in gold_list):
            matched_cited.append(c)
        else:
            hallucinated.append(c)

    matched_gold = []
    missing_gold = []
    for g in gold_list:
        if any(_chunk_matches(c, g) for c in cited_list):
            matched_gold.append(g)
        else:
            missing_gold.append(g)

    precision = len(matched_cited) / len(cited_list) if cited_list else 1.0
    recall = len(matched_gold) / len(gold_list) if gold_list else 1.0

    return CitationAuditResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        hallucinated_citations=hallucinated,
        missing_citations=missing_gold,
        matched_citations=matched_cited
    )


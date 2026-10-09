"""
Stage 3: Parent-Child Chunking

Indexes small **child** chunks for high-precision retrieval, but stores
a reference to their **parent** chunk (a larger surrounding context window).

At synthesis time the retriever returns child chunks, then the postprocessor
expands each child reference to its parent for richer context.

This gives the best of both worlds:
- Precise retrieval (small child chunks → better embedding alignment).
- Rich synthesis context (parent chunks → more complete answer grounding).

Constitution v1.2.0 / Law of Parsimony:
- Zero external dependencies.
- Parent chunks are built from the hierarchical chunker output.
- Parent-child relationships are stored as ``parent_chunk_id`` on each Chunk.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from rag_core.chunking.hierarchical import Chunk, HierarchicalChunker
from rag_core.parsing.schema import ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParentChildChunker
# ---------------------------------------------------------------------------


class ParentChildChunker:
    """
    Two-level chunking: large **parent** chunks + small **child** chunks.

    The parent chunker uses a large window (``parent_chunk_chars``) to
    create contextually rich parent segments.  The child chunker then
    splits each parent into smaller, more embedding-friendly children.

    Retrieval indexes **child chunks** only.
    At synthesis time, child → parent expansion gives full context.

    Args:
        parent_chunk_chars: Maximum character count for parent chunks (default 3000).
        child_chunk_chars:  Maximum character count for child chunks (default 600).
        overlap_chars:      Overlap between siblings within the same parent (default 80).

    Example::

        chunker = ParentChildChunker()
        parents, children = chunker.chunk(parsed_doc)
        # Index only `children` in Qdrant
        # At query time expand child hits to parents for synthesis
    """

    def __init__(
        self,
        parent_chunk_chars: int = 3000,
        child_chunk_chars: int = 600,
        overlap_chars: int = 80,
    ) -> None:
        self._parent_chunker = HierarchicalChunker(
            max_chunk_chars=parent_chunk_chars,
            overlap_chars=0,  # no overlap between parents
        )
        self._child_max = child_chunk_chars
        self._overlap = overlap_chars

    def chunk(
        self, parsed_doc: ParsedDocument
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Produce parent and child chunk lists.

        Args:
            parsed_doc: Canonical IR document.

        Returns:
            ``(parents, children)`` tuple where every child's
            ``parent_chunk_id`` points to a parent in *parents*.
        """
        parents = self._parent_chunker.chunk(parsed_doc)
        children: List[Chunk] = []
        child_index = 0

        for parent in parents:
            parent_text = parent.text
            child_chunks = self._split_to_children(
                parent_text=parent_text,
                parent=parent,
                child_index_start=child_index,
                doc_id=parsed_doc.doc_id,
            )
            children.extend(child_chunks)
            child_index += len(child_chunks)

        logger.info(
            "ParentChildChunker: doc_id=%s → %d parents, %d children",
            parsed_doc.doc_id,
            len(parents),
            len(children),
        )
        return parents, children

    def _split_to_children(
        self,
        parent_text: str,
        parent: Chunk,
        child_index_start: int,
        doc_id: str,
    ) -> List[Chunk]:
        """
        Split *parent_text* into child chunks of at most ``child_chunk_chars``.

        Tries to split at sentence boundaries (``". "`` or ``"\n"``).
        Falls back to hard character split if no boundary is found.
        """
        children: List[Chunk] = []
        text = parent_text
        idx = child_index_start

        while len(text) > self._child_max:
            # Try sentence boundary split
            split_at = self._find_split_boundary(text, self._child_max)
            child_text = text[:split_at].strip()
            if child_text:
                children.append(
                    Chunk(
                        chunk_id=f"{doc_id}_child_{idx:04d}",
                        doc_id=doc_id,
                        text=child_text,
                        page=parent.page,
                        chunk_index=idx,
                        breadcrumb=parent.breadcrumb,
                        heading_level=parent.heading_level,
                        parent_chunk_id=parent.chunk_id,
                        metadata={**parent.metadata, "chunk_type": "child"},
                    )
                )
                idx += 1

            # Overlap: cap overlap to at most half of split_at to ensure forward progress
            effective_overlap = min(self._overlap, split_at // 2)
            advance = max(1, split_at - effective_overlap)
            text = text[advance:]

        # Remainder
        if text.strip():
            children.append(
                Chunk(
                    chunk_id=f"{doc_id}_child_{idx:04d}",
                    doc_id=doc_id,
                    text=text.strip(),
                    page=parent.page,
                    chunk_index=idx,
                    breadcrumb=parent.breadcrumb,
                    heading_level=parent.heading_level,
                    parent_chunk_id=parent.chunk_id,
                    metadata={**parent.metadata, "chunk_type": "child"},
                )
            )

        return children

    @staticmethod
    def _find_split_boundary(text: str, max_chars: int) -> int:
        """
        Find the best split position at or before *max_chars*.

        Prefers splitting at ``". "`` (sentence end) or ``"\n"`` (paragraph).
        Falls back to ``max_chars`` (hard split) if no boundary is found.
        """
        window = text[:max_chars]
        # Try sentence boundary
        for sep in (". ", ".\n", "\n\n", "\n"):
            pos = window.rfind(sep)
            if pos > max_chars // 2:  # at least half way through
                return pos + len(sep)
        return max_chars


# ---------------------------------------------------------------------------
# Utility: build parent lookup dict
# ---------------------------------------------------------------------------


def build_parent_lookup(parents: List[Chunk]) -> Dict[str, Chunk]:
    """
    Build a ``{chunk_id: Chunk}`` lookup dict from a list of parent chunks.

    Used by the postprocessing expansion stage to quickly retrieve parent
    context from a child's ``parent_chunk_id``.

    Args:
        parents: List of parent ``Chunk`` objects.

    Returns:
        Dict mapping ``chunk_id`` → ``Chunk``.
    """
    return {p.chunk_id: p for p in parents}

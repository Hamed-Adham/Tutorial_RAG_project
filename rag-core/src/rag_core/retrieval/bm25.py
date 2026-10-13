"""
In-Memory BM25 Retriever for 7th-Grade Science Textbook.
Performs lexical inverted index retrieval with Persian stemming, returning typed RetrievalCandidate objects.
"""

from pathlib import Path
from typing import List, Optional
import structlog

from rag_core.indexing.science_schema import RetrievalCandidate
from rag_core.indexing.bm25_index import ScienceBM25Index
from rag_core.contracts import IBM25Retriever

logger = structlog.get_logger(__name__)


class BM25Retriever(IBM25Retriever):
    """Sub-5ms in-memory BM25 lexical retriever."""

    def __init__(self, index: Optional[ScienceBM25Index] = None, index_path: Optional[str | Path] = None):
        if index is not None:
            self.index = index
        elif index_path is not None:
            self.index = ScienceBM25Index.load(index_path)
        else:
            default_path = Path("data_index/science_bm25.pkl")
            if default_path.exists():
                self.index = ScienceBM25Index.load(default_path)
            else:
                self.index = ScienceBM25Index()
                logger.warning("BM25 index not found at default path, initialized empty index.")

    def search(
        self,
        query: str,
        top_k: int = 50,
        chapters: Optional[List[int]] = None
    ) -> List[RetrievalCandidate]:
        """Perform sub-5ms BM25 lexical lookup returning typed RetrievalCandidate objects with optional chapter filtering."""
        if not query or not query.strip():
            return []

        raw_results = self.index.search(query, top_k=top_k, chapter_filter=chapters)
        candidates = []

        for rank, (chunk, score) in enumerate(raw_results, start=1):
            cand = RetrievalCandidate(
                chunk_id=chunk.chunk_id,
                doc_id=getattr(chunk, "doc_id", "science_textbook"),
                section_id=chunk.section_id,
                chapter=chunk.chapter,
                page=chunk.page,
                text=chunk.text,
                breadcrumbs=chunk.breadcrumbs,
                bm25_score=score,
                bm25_rank=rank
            )
            candidates.append(cand)

        return candidates

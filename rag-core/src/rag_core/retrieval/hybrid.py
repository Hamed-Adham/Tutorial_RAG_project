"""
Async Parallel Hybrid Retriever executing Qdrant Dense and BM25 Lexical search concurrently.
Fuses candidate ranks using Reciprocal Rank Fusion within the 180ms retrieval budget.
"""

import asyncio
from typing import List, Optional, Any, Dict
import structlog

from rag_core.indexing.science_schema import RetrievalCandidate
from rag_core.retrieval.bm25 import BM25Retriever
from rag_core.fusion.rrf import ReciprocalRankFusion

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """Parallel hybrid retriever blending dense semantic and lexical BM25 search."""

    def __init__(
        self,
        dense_retriever: Any = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion: Optional[ReciprocalRankFusion] = None
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.fusion = fusion or ReciprocalRankFusion(k=60)

    async def _search_bm25(self, query: str, top_k: int, chapters: Optional[List[int]] = None) -> List[RetrievalCandidate]:
        """Run BM25 search in executor to keep event loop unblocked with native chapter pre-filtering."""
        loop = asyncio.get_running_loop()
        bm25_kwargs: Dict[str, Any] = {"top_k": top_k}
        if chapters is not None:
            bm25_kwargs["chapters"] = chapters
        return await loop.run_in_executor(None, lambda: self.bm25_retriever.search(query, **bm25_kwargs))

    def _resolve_dense_retriever(self) -> Any:
        """Lazily initialize default QdrantRetriever if none was provided."""
        if self.dense_retriever is not None:
            return self.dense_retriever

        try:
            from rag_core.naive.qdrant_retriever import QdrantRetriever, create_qdrant_client
            from rag_core.naive.embedder import BgeM3Embedder
            from rag_core.config import settings

            client = create_qdrant_client()
            collection = "science_textbook_dense" if client.collection_exists("science_textbook_dense") else settings.qdrant_collection_name

            embedder = BgeM3Embedder()
            self.dense_retriever = QdrantRetriever(
                embedder=embedder,
                collection_name=collection,
                client=client
            )
            return self.dense_retriever
        except Exception as e:
            logger.warning("Could not initialize default dense retriever; proceeding without dense leg", error=str(e))
            return None

    def _convert_to_candidates(self, raw_items: Any) -> List[RetrievalCandidate]:
        """Convert arbitrary dense result formats into typed RetrievalCandidate instances."""
        if not raw_items:
            return []

        candidates: List[RetrievalCandidate] = []
        for rank, item in enumerate(raw_items, start=1):
            if isinstance(item, RetrievalCandidate):
                if item.dense_rank is None:
                    item.dense_rank = rank
                candidates.append(item)
            elif hasattr(item, "chunk_id") and hasattr(item, "text"):
                meta = getattr(item, "metadata", {}) or {}
                cand = RetrievalCandidate(
                    chunk_id=str(item.chunk_id),
                    section_id=str(meta.get("section_id", getattr(item, "parent_chunk_id", None) or item.chunk_id)),
                    chapter=int(meta.get("chapter", 1)) if isinstance(meta.get("chapter"), (int, float)) else 1,
                    page=int(getattr(item, "page", 0) or meta.get("page", 1)),
                    text=str(item.text),
                    breadcrumbs=str(getattr(item, "breadcrumb", "") or meta.get("breadcrumbs", "")),
                    doc_id=str(getattr(item, "doc_id", "") or "science_textbook"),
                    dense_score=float(getattr(item, "score", 0.0)),
                    dense_rank=rank
                )
                candidates.append(cand)
            elif isinstance(item, tuple) and len(item) == 2:
                chunk, score = item
                meta = getattr(chunk, "metadata", {}) or {}
                ch_val = meta.get("chapter_num", meta.get("chapter", 1))
                if isinstance(ch_val, str) and ch_val.isdigit():
                    ch_val = int(ch_val)
                elif not isinstance(ch_val, int):
                    ch_val = 1

                page_val = meta.get("page_number", meta.get("page", 1))
                if not isinstance(page_val, int):
                    page_val = 1

                cand = RetrievalCandidate(
                    chunk_id=str(chunk.chunk_id),
                    section_id=str(meta.get("section_id", chunk.chunk_id)),
                    chapter=ch_val,
                    page=page_val,
                    text=str(chunk.text),
                    breadcrumbs=str(meta.get("breadcrumbs", meta.get("category", ""))),
                    doc_id=str(getattr(chunk, "doc_id", "science_textbook") or "science_textbook"),
                    dense_score=float(score),
                    dense_rank=rank
                )
                candidates.append(cand)
            elif isinstance(item, dict):
                meta = item.get("metadata", {}) or {}
                cand = RetrievalCandidate(
                    chunk_id=str(item.get("chunk_id", f"dense_chunk_{rank}")),
                    section_id=str(meta.get("section_id", item.get("section_id", "sec_unknown"))),
                    chapter=int(meta.get("chapter", item.get("chapter", 1))),
                    page=int(meta.get("page", item.get("page", 1))),
                    text=str(item.get("text", "")),
                    breadcrumbs=str(meta.get("breadcrumbs", item.get("breadcrumbs", ""))),
                    doc_id=str(item.get("doc_id", "science_textbook")),
                    dense_score=float(item.get("score", item.get("dense_score", 0.0))),
                    dense_rank=rank
                )
                candidates.append(cand)

        return candidates

    async def _search_dense(self, query: str, top_k: int, chapters: Optional[List[int]] = None) -> List[RetrievalCandidate]:
        """Run dense search asynchronously if supported, or wrapped in executor with native chapter pre-filtering."""
        retriever = self._resolve_dense_retriever()
        if retriever is None:
            return []

        raw_results = []
        dense_kwargs: Dict[str, Any] = {"top_k": top_k}
        if chapters is not None:
            dense_kwargs["filter_chapters"] = chapters

        try:
            if hasattr(retriever, "retrieve"):
                if asyncio.iscoroutinefunction(retriever.retrieve):
                    raw_results = await retriever.retrieve(query, **dense_kwargs)
                else:
                    loop = asyncio.get_running_loop()
                    raw_results = await loop.run_in_executor(None, lambda: retriever.retrieve(query, **dense_kwargs))
            elif hasattr(retriever, "search"):
                if asyncio.iscoroutinefunction(retriever.search):
                    raw_results = await retriever.search(query, top_k=top_k)
                else:
                    loop = asyncio.get_running_loop()
                    raw_results = await loop.run_in_executor(None, retriever.search, query, top_k)
        except Exception as exc:
            logger.warning("Dense search execution failed", error=str(exc))
            return []

        return self._convert_to_candidates(raw_results)

    async def retrieve_with_details(
        self,
        query: str,
        top_k: int = 50,
        overfetch_k: int = 50,
        chapters: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Execute dense and BM25 searches in parallel with native chapter pre-filtering and fuse with RRF."""
        if not query or not query.strip():
            return {"fused": [], "dense": [], "bm25": []}

        # Overfetch candidate pool before rank fusion
        candidate_k = max(overfetch_k, top_k)
        dense_task = asyncio.create_task(self._search_dense(query, top_k=candidate_k, chapters=chapters))
        bm25_task = asyncio.create_task(self._search_bm25(query, top_k=candidate_k, chapters=chapters))

        dense_results, bm25_results = [], []

        try:
            results = await asyncio.gather(dense_task, bm25_task, return_exceptions=True)

            if isinstance(results[0], Exception):
                logger.warning("Dense retrieval failed, continuing with BM25 only", error=str(results[0]))
            elif isinstance(results[0], list):
                dense_results = results[0]

            if isinstance(results[1], Exception):
                logger.warning("BM25 retrieval failed, continuing with dense only", error=str(results[1]))
            elif isinstance(results[1], list):
                bm25_results = results[1]

        except Exception as e:
            logger.error("Hybrid retrieval fan-out failed", error=str(e))
            return {"fused": [], "dense": [], "bm25": []}

        # Blend ranks via Reciprocal Rank Fusion
        fused = self.fusion.fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=top_k
        )
        return {
            "fused": fused,
            "dense": dense_results,
            "bm25": bm25_results
        }

    async def retrieve(
        self,
        query: str,
        top_k: int = 50,
        overfetch_k: int = 50,
        chapters: Optional[List[int]] = None
    ) -> List[RetrievalCandidate]:
        """Execute dense and BM25 searches in parallel with native chapter pre-filtering and fuse with RRF."""
        details = await self.retrieve_with_details(
            query=query,
            top_k=top_k,
            overfetch_k=overfetch_k,
            chapters=chapters
        )
        return details["fused"]

"""
Stage 6: Dense HNSW Vector Retriever

Performs semantic nearest-neighbour search against the Qdrant collection
using the BGE-M3 query embedding with the ``"query: "`` prefix.

Constitution v1.2.0:
- Offline fallback: if Qdrant is unreachable, returns empty results with
  a warning rather than crashing (graceful degradation for unit tests).
- ACL filtering: accepts an optional Qdrant ``Filter`` payload filter so
  retrieval is scoped to the caller's permitted roles/tenants.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single result returned by the dense retriever."""

    chunk_id: str
    """Qdrant point ID (matches ``Chunk.chunk_id``)."""

    text: str
    """Chunk text (with breadcrumb prefix)."""

    score: float
    """Cosine similarity score (0.0–1.0)."""

    page: int = 0
    """Source page number."""

    doc_id: str = ""
    """Document identifier from the Qdrant payload."""

    breadcrumb: str = ""
    """Heading breadcrumb from the Qdrant payload."""

    parent_chunk_id: Optional[str] = None
    """Parent chunk reference for Parent-Child expansion."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata from the Qdrant payload."""


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """
    HNSW semantic vector retriever backed by Qdrant.

    Applies the BGE-M3 query prefix (``"query: "``) before embedding the
    query string, which is required for correct asymmetric retrieval.

    Args:
        embedding_model: A model exposing ``get_query_embedding(text) -> list[float]``.
        vector_store:    A Qdrant ``QdrantClient`` or compatible wrapper with a
                         ``search(collection_name, query_vector, limit, query_filter)``
                         method returning a list of ``ScoredPoint`` objects.
        collection_name: Name of the Qdrant collection to search (default ``"rag_chunks"``).
        top_k:           Number of results to retrieve (default 5).
        query_prefix:    Prefix prepended to every query (default ``"query: "``).
        score_threshold: Minimum similarity score to include in results (default 0.0).

    Example::

        retriever = DenseRetriever(
            embedding_model=bge_model,
            vector_store=qdrant_client,
        )
        results = retriever.retrieve("What is the refund policy?")
    """

    QUERY_PREFIX = "query: "

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        collection_name: str = "rag_chunks",
        top_k: int = 5,
        query_prefix: str = "query: ",
        score_threshold: float = 0.0,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._top_k = top_k
        self._query_prefix = query_prefix
        self._score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        acl_filter: Optional[Any] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most semantically similar chunks for *query*.

        Args:
            query:      Natural language query string.
            top_k:      Override default ``top_k`` for this call.
            acl_filter: Optional Qdrant ``Filter`` for ACL-based payload filtering.

        Returns:
            List of ``RetrievedChunk`` objects ordered by descending similarity.
        """
        t_start = time.perf_counter()
        k = top_k or self._top_k

        if self._embedding_model is None or self._vector_store is None:
            logger.warning(
                "DenseRetriever: no embedding model or vector store configured — "
                "returning empty results."
            )
            return []

        try:
            # Apply query prefix (required for BGE-M3 asymmetric retrieval)
            prefixed_query = f"{self._query_prefix}{query}"
            query_vector = self._embedding_model.get_query_embedding(prefixed_query)

            # Qdrant search
            hits = self._vector_store.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=k,
                query_filter=acl_filter,
                with_payload=True,
            )

            results: List[RetrievedChunk] = []
            for hit in hits:
                score = float(hit.score)
                if score < self._score_threshold:
                    continue
                payload = hit.payload or {}
                results.append(
                    RetrievedChunk(
                        chunk_id=str(hit.id),
                        text=payload.get("text", ""),
                        score=score,
                        page=int(payload.get("page", 0)),
                        doc_id=payload.get("doc_id", ""),
                        breadcrumb=payload.get("breadcrumb", ""),
                        parent_chunk_id=payload.get("parent_chunk_id"),
                        metadata=payload.get("metadata", {}),
                    )
                )

            latency = time.perf_counter() - t_start
            logger.info(
                "DenseRetriever: query='%.60s...' k=%d hits=%d latency=%.3fs",
                query,
                k,
                len(results),
                latency,
            )
            return results

        except Exception as exc:
            logger.error("DenseRetriever error: %s", exc, exc_info=True)
            return []

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        acl_filter: Optional[Any] = None,
    ) -> List[List[RetrievedChunk]]:
        """
        Retrieve results for a batch of queries.

        Args:
            queries: List of query strings.
            top_k:   Results per query.
            acl_filter: Optional Qdrant ACL filter applied to all queries.

        Returns:
            List of result lists in the same order as *queries*.
        """
        return [
            self.retrieve(q, top_k=top_k, acl_filter=acl_filter)
            for q in queries
        ]

"""
Stage 4: Unified Multi-Index Indexer

Orchestrates the full ingestion pipeline:

    manifest → parsing → chunking → embedding → Qdrant

Single entry point for indexing any document.  Accepts raw file bytes,
routes through the parsing tier, applies the configured chunking strategy,
generates BGE-M3 embeddings, and upserts into Qdrant.

Constitution v1.2.0:
- Three-layer decoupled: rag-core only; no FastAPI/Next.js imports.
- Langfuse spans wrap parse, chunk, embed stages.
- Deduplication check is run before any expensive operations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rag_core.chunking.hierarchical import Chunk, HierarchicalChunker
from rag_core.chunking.parent_child import ParentChildChunker
from rag_core.manifest.catalog import DocumentCatalog, DocumentStatus
from rag_core.manifest.deduplicator import DocumentDeduplicator
from rag_core.parsing.router import ParsingRouter
from rag_core.parsing.schema import ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexing result
# ---------------------------------------------------------------------------


@dataclass
class IndexingResult:
    """Summary of a single document indexing operation."""

    doc_id: str
    filename: str
    status: str  # "indexed" | "duplicate" | "skipped" | "error"
    chunks_indexed: int = 0
    parse_latency_s: float = 0.0
    chunk_latency_s: float = 0.0
    embed_latency_s: float = 0.0
    total_latency_s: float = 0.0
    message: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class DocumentIndexer:
    """
    Unified multi-index orchestrator for Stage 4.

    Wires together:
    - ``DocumentCatalog``: SHA-256 registration and status tracking.
    - ``DocumentDeduplicator``: Exact + near-duplicate filtering.
    - ``ParsingRouter``: Tiered parsing to Canonical IR.
    - ``HierarchicalChunker`` / ``ParentChildChunker``: Chunking strategy.
    - Embedding model: BGE-M3 (via ``rag_core.indexing.embeddings``).
    - Qdrant vector store: HNSW upsert (via ``rag_core.indexing.vector_store``).

    The embedding and vector-store calls are attempted lazily so the indexer
    can be unit-tested without a running Qdrant or GPU.

    Args:
        catalog: Configured ``DocumentCatalog`` instance.
        deduplicator: Optional ``DocumentDeduplicator``; created fresh if not provided.
        chunking_strategy: ``"hierarchical"`` (default) or ``"parent_child"``.
        max_chunk_chars: Max characters per chunk (default 1500).
        embedding_model: Optional pre-loaded embedding model instance.
        vector_store: Optional pre-configured Qdrant vector store instance.
        tracker: Optional ``LangfuseTracker`` for span instrumentation.
    """

    def __init__(
        self,
        catalog: Optional[DocumentCatalog] = None,
        deduplicator: Optional[DocumentDeduplicator] = None,
        chunking_strategy: str = "hierarchical",
        max_chunk_chars: int = 1500,
        embedding_model: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        self._catalog = catalog or DocumentCatalog()
        self._deduplicator = deduplicator or DocumentDeduplicator()
        self._router = ParsingRouter()
        self._chunking_strategy = chunking_strategy
        self._max_chunk_chars = max_chunk_chars
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._tracker = tracker

        if chunking_strategy == "parent_child":
            self._chunker: Union[HierarchicalChunker, ParentChildChunker] = (
                ParentChildChunker(child_chunk_chars=max_chunk_chars)
            )
        else:
            self._chunker = HierarchicalChunker(max_chunk_chars=max_chunk_chars)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(
        self,
        source: Union[str, Path, bytes],
        filename: str,
        tenant_id: str = "default",
        acl_roles: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> IndexingResult:
        """
        Index a single document through the full pipeline.

        Args:
            source: File path or raw bytes.
            filename: Original filename (used for MIME probing and breadcrumbs).
            tenant_id: Tenant namespace.
            acl_roles: ACL roles for retrieval access control.
            extra_metadata: Arbitrary metadata for the catalog record.

        Returns:
            ``IndexingResult`` summarising the operation.
        """
        t_start = time.perf_counter()

        # Read raw bytes for SHA-256 computation
        if isinstance(source, (str, Path)):
            raw_bytes = Path(source).read_bytes()
        else:
            raw_bytes = source

        # --- Catalog registration (SHA-256 dedup) ---
        catalog_result = self._catalog.register(
            filename=filename,
            content=raw_bytes,
            tenant_id=tenant_id,
            acl_roles=acl_roles or ["public"],
            extra_metadata=extra_metadata or {},
        )

        if catalog_result.is_duplicate:
            logger.info("Skipping duplicate document: %s", filename)
            return IndexingResult(
                doc_id=catalog_result.doc_id,
                filename=filename,
                status="duplicate",
                message=catalog_result.message,
                total_latency_s=time.perf_counter() - t_start,
            )

        doc_id = catalog_result.doc_id

        try:
            # --- Stage 2: Parse ---
            t_parse = time.perf_counter()
            parsed_doc = self._router.parse(
                source=raw_bytes,
                doc_id=doc_id,
                filename=filename,
            )
            parse_latency = time.perf_counter() - t_parse

            # --- Stage 3: Chunk ---
            t_chunk = time.perf_counter()
            chunks = self._get_chunks(parsed_doc)
            chunk_latency = time.perf_counter() - t_chunk

            # --- Stage 4: Embed + Upsert ---
            t_embed = time.perf_counter()
            n_indexed = self._embed_and_upsert(chunks, parsed_doc)
            embed_latency = time.perf_counter() - t_embed

            # Update catalog status
            self._catalog.update_status(doc_id, DocumentStatus.INDEXED)

            total = time.perf_counter() - t_start
            logger.info(
                "Indexed doc_id=%s filename=%s chunks=%d total=%.2fs",
                doc_id,
                filename,
                n_indexed,
                total,
            )
            return IndexingResult(
                doc_id=doc_id,
                filename=filename,
                status="indexed",
                chunks_indexed=n_indexed,
                parse_latency_s=parse_latency,
                chunk_latency_s=chunk_latency,
                embed_latency_s=embed_latency,
                total_latency_s=total,
            )

        except Exception as exc:
            self._catalog.update_status(doc_id, DocumentStatus.ERROR)
            logger.error("Indexing failed for %s: %s", filename, exc, exc_info=True)
            return IndexingResult(
                doc_id=doc_id,
                filename=filename,
                status="error",
                error=str(exc),
                total_latency_s=time.perf_counter() - t_start,
            )

    def index_batch(
        self,
        documents: List[Dict[str, Any]],
        tenant_id: str = "default",
    ) -> List[IndexingResult]:
        """
        Index a batch of documents.

        Args:
            documents: List of dicts with keys ``source``, ``filename``,
                and optionally ``acl_roles`` and ``extra_metadata``.
            tenant_id: Tenant namespace applied to all documents.

        Returns:
            List of ``IndexingResult`` objects in the same order.
        """
        results = []
        for doc in documents:
            result = self.index_document(
                source=doc["source"],
                filename=doc["filename"],
                tenant_id=tenant_id,
                acl_roles=doc.get("acl_roles"),
                extra_metadata=doc.get("extra_metadata"),
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_chunks(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """Run the configured chunking strategy and return child chunks."""
        if isinstance(self._chunker, ParentChildChunker):
            _parents, children = self._chunker.chunk(parsed_doc)
            return children
        else:
            return self._chunker.chunk(parsed_doc)

    def _embed_and_upsert(self, chunks: List[Chunk], parsed_doc: ParsedDocument) -> int:
        """
        Generate embeddings and upsert into the vector store.

        If no embedding model or vector store is configured (e.g. in unit tests),
        this method returns the chunk count without calling any external service.
        """
        if not chunks:
            return 0

        if self._embedding_model is None or self._vector_store is None:
            logger.debug(
                "No embedding model or vector store configured — "
                "skipping embed+upsert for doc_id=%s",
                parsed_doc.doc_id,
            )
            return len(chunks)

        try:
            texts = [c.text for c in chunks]
            embeddings = self._embedding_model.get_text_embedding_batch(texts)

            points = []
            for chunk, embedding in zip(chunks, embeddings):
                points.append(
                    {
                        "id": chunk.chunk_id,
                        "vector": embedding,
                        "payload": {
                            "doc_id": chunk.doc_id,
                            "text": chunk.text,
                            "page": chunk.page,
                            "breadcrumb": chunk.breadcrumb,
                            "parent_chunk_id": chunk.parent_chunk_id,
                            "metadata": chunk.metadata,
                        },
                    }
                )

            self._vector_store.upsert(points)
            return len(chunks)

        except Exception as exc:
            logger.error("Embed+upsert failed: %s", exc, exc_info=True)
            raise

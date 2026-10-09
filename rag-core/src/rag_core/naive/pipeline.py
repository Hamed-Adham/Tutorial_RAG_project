#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end Naive RAG Pipeline.
Serves as the empirical "Ground Zero" baseline adhering to the Law of Parsimony.
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from rag_core.config import settings
from rag_core.naive.document_loader import load_all_documents, Document
from rag_core.naive.chunker import chunk_documents, Chunk
from rag_core.naive.embedder import BgeM3Embedder
from rag_core.naive.retriever import NaiveRetriever
from rag_core.naive.qdrant_retriever import QdrantRetriever
from rag_core.naive.synthesizer import NaiveSynthesizer
from rag_core.observability.langfuse import LangfuseTracker

class NaiveRAGPipeline:
    """Complete end-to-end baseline RAG pipeline with pluggable Qdrant / Numpy retriever."""

    def __init__(
        self,
        embedder: Optional[BgeM3Embedder] = None,
        retriever: Optional[Any] = None,
        retriever_type: str = "qdrant",
        synthesizer: Optional[NaiveSynthesizer] = None,
        data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        self.data_dir = data_dir or settings.data_dir
        self.cache_dir = cache_dir or (settings.workspace_root / "data_index")
        self.retriever_type = retriever_type.lower()
        
        # Centralized Langfuse telemetry manager
        self.tracker = LangfuseTracker()
        
        self.embedder = embedder or BgeM3Embedder()
        
        if retriever is not None:
            self.retriever = retriever
        elif self.retriever_type == "qdrant":
            self.retriever = QdrantRetriever(self.embedder, collection_name=settings.qdrant_collection_name)
        else:
            self.retriever = NaiveRetriever(self.embedder, cache_dir=self.cache_dir)
            
        self.synthesizer = synthesizer or NaiveSynthesizer(
            base_url=settings.nine_router_base_url,
            api_key=settings.nine_router_api_key,
            model=settings.llm_model
        )
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self._is_indexed = False

    def build_index(self, force_reindex: bool = False):
        """Load documents, chunk them, and index into the naive retriever."""
        print(f"Building naive index from {self.data_dir}...")
        self.documents = load_all_documents(self.data_dir)
        self.chunks = chunk_documents(self.documents, chunk_size=500, overlap=50)
        print(f"Loaded {len(self.documents)} documents -> {len(self.chunks)} chunks.")
        self.retriever.index_chunks(self.chunks, force_reindex=force_reindex)
        self._is_indexed = True
        print("Naive index ready.")

    def query(
        self,
        question: str,
        top_k: int = 3,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute complete query: retrieve chunks, synthesize response, return telemetry."""
        if not self._is_indexed:
            self.build_index()

        t0 = time.time()
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - t0

        t1 = time.time()
        answer = self.synthesizer.synthesize(question, retrieved)
        generation_time = time.time() - t1
        total_time = time.time() - t0

        retrieved_contexts = [chunk.text for chunk, _ in retrieved]
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "source_name": chunk.source_name,
                "score": score
            }
            for chunk, score in retrieved
        ]

        # Log trace to Langfuse if active
        combined_tags = [f"retriever:{self.retriever_type}"]
        if tags:
            combined_tags.extend(tags)

        trace_id = self.tracker.trace_query(
            name="rag_query",
            question=question,
            answer=answer,
            top_k=top_k,
            retrieved_count=len(retrieved),
            sources=sources,
            retrieval_latency=retrieval_time,
            generation_latency=generation_time,
            session_id=session_id,
            tags=combined_tags,
            metadata={"retriever_type": self.retriever_type, **(metadata or {})},
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "sources": sources,
            "retriever_type": self.retriever_type,
            "retrieval_latency": round(retrieval_time, 3),
            "generation_latency": round(generation_time, 3),
            "total_latency": round(total_time, 3),
            "trace_id": trace_id
        }


    def close(self):
        """Release underlying retriever resources (e.g. Qdrant connection)."""
        if hasattr(self.retriever, "close"):
            self.retriever.close()
        self.tracker.flush()

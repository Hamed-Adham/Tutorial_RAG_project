#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naive dense retriever using cosine similarity over in-memory or persisted numpy arrays.
Caches embeddings to avoid repeated inference overhead.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from rag_core.naive.chunker import Chunk
from rag_core.naive.embedder import BgeM3Embedder

class NaiveRetriever:
    """In-memory cosine similarity dense retriever."""

    def __init__(self, embedder: BgeM3Embedder, cache_dir: Optional[Path] = None):
        self.embedder = embedder
        self.cache_dir = cache_dir or Path(os.getcwd()) / "data_index"
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def index_chunks(self, chunks: List[Chunk], force_reindex: bool = False):
        """Compute or load embeddings for the provided chunks."""
        self.chunks = chunks
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "naive_index.npz"
        meta_file = self.cache_dir / "naive_metadata.json"

        if (not force_reindex) and cache_file.exists() and meta_file.exists():
            print(f"Loading cached naive index from {cache_file}...")
            data = np.load(str(cache_file))
            cached_embs = data["embeddings"]
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if len(meta) == len(chunks):
                self.embeddings = cached_embs
                print(f"Loaded {len(chunks)} cached embeddings.")
                return

        print(f"Embedding {len(chunks)} chunks using BGE-M3...")
        texts = [c.text for c in chunks]
        self.embeddings = self.embedder.embed_documents(texts)

        # Save cache
        np.savez_compressed(str(cache_file), embeddings=self.embeddings)
        meta_records = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_type": c.source_type,
                "source_name": c.source_name,
                "metadata": c.metadata
            }
            for c in chunks
        ]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_records, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(chunks)} embeddings to {cache_file}.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Retrieve top_k chunks by cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("Index is empty. Call index_chunks() before retrieving.")

        query_vec = self.embedder.embed_query(query)
        # Cosine similarity: (N, D) @ (D,) since vectors are unit normalized
        scores = np.dot(self.embeddings, query_vec)
        
        # Sort indices descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])
            results.append((chunk, score))

        return results

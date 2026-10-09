#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant vector store retriever for Unified RAG.
Supports embedded disk storage (Law of Parsimony default) as well as remote Qdrant instances.
Stores chunk payloads directly in Qdrant for decoupled, persistent vector search.
"""

import os
import json
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from rag_core.config import settings
from rag_core.naive.chunker import Chunk
from rag_core.naive.embedder import BgeM3Embedder

def create_qdrant_client(
    path: Optional[str] = None,
    url: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None
) -> QdrantClient:
    """
    Intelligently instantiate a QdrantClient.
    Priority:
    1. Embedded SQLite-like mode (Zero Docker required) if settings.qdrant_mode == "embedded"
    2. Explicit URL (if provided or in settings.qdrant_url)
    3. Live HTTP service check on host:port (if online in "server" mode)
    4. Fall back to embedded local storage on disk
    """
    local_path = path or settings.qdrant_path

    # 1. Embedded SQLite-like mode (Zero Docker required)
    if settings.qdrant_mode == "embedded":
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Using embedded local Qdrant database at: {local_path}")
        return QdrantClient(path=local_path)

    # 2. Explicit URL
    target_url = url or settings.qdrant_url
    if target_url:
        print(f"Connecting to Qdrant at URL: {target_url}")
        return QdrantClient(url=target_url, api_key=api_key or settings.qdrant_api_key)

    # 3. Check if remote host:port is responsive
    h = host or settings.qdrant_host
    p = port or settings.qdrant_port
    try:
        check_url = f"http://{h}:{p}/healthz"
        with urllib.request.urlopen(check_url, timeout=0.5) as resp:
            if resp.status == 200:
                print(f"Connecting to live Qdrant server at http://{h}:{p}")
                return QdrantClient(host=h, port=p, api_key=api_key or settings.qdrant_api_key)
    except Exception:
        pass

    # 4. Fall back to embedded local storage on disk
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Using embedded local Qdrant database at: {local_path}")
    return QdrantClient(path=local_path)


class QdrantRetriever:
    """
    Qdrant-backed dense retriever with payload storage and cosine similarity search.
    """

    def __init__(
        self,
        embedder: BgeM3Embedder,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        path: Optional[str] = None
    ):
        self.embedder = embedder
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = client or create_qdrant_client(path=path)

    def index_chunks(self, chunks: List[Chunk], force_reindex: bool = False):
        """
        Embed and persist chunks and their metadata payloads in Qdrant.
        Leverages precomputed cached embeddings if available to eliminate redundant inference.
        """
        exists = self.client.collection_exists(self.collection_name)
        if exists:
            if not force_reindex:
                info = self.client.get_collection(self.collection_name)
                # In Qdrant client, points_count may be in info.points_count
                points_count = getattr(info, "points_count", None) or getattr(info, "vectors_count", 0)
                if points_count == len(chunks):
                    print(f"Qdrant collection '{self.collection_name}' already exists with {points_count} points. Reusing index.")
                    return
            print(f"Recreating collection '{self.collection_name}' in Qdrant...")
            self.client.delete_collection(self.collection_name)

        # Create collection with Cosine distance and 1024 vector dimensions
        vector_dim = getattr(self.embedder, "dimension", 1024)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        print(f"Created Qdrant collection '{self.collection_name}' (dim={vector_dim}, distance=Cosine).")

        # Check if we have precomputed cached embeddings in data_index/naive_index.npz
        cache_npz = settings.workspace_root / "data_index" / "naive_index.npz"
        cache_meta = settings.workspace_root / "data_index" / "naive_metadata.json"
        cached_embeddings = None

        if cache_npz.exists() and cache_meta.exists() and not force_reindex:
            try:
                data = np.load(str(cache_npz))
                raw_embs = data["embeddings"]
                with open(cache_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if len(raw_embs) == len(chunks) and len(meta) == len(chunks):
                    print(f"Reusing {len(chunks)} pre-computed embeddings from {cache_npz.name} for rapid Qdrant indexing...")
                    cached_embeddings = raw_embs
            except Exception as e:
                print(f"Could not load precomputed embeddings cache: {e}. Will re-embed.")

        if cached_embeddings is not None:
            embeddings = cached_embeddings
        else:
            print(f"Embedding {len(chunks)} chunks via BGE-M3 for Qdrant...")
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_documents(texts)

        # Prepare points
        points: List[PointStruct] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            vector_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "source_type": chunk.source_type,
                "source_name": chunk.source_name,
                "metadata": chunk.metadata
            }
            points.append(PointStruct(id=idx, vector=vector_list, payload=payload))

        # Batch upsert in chunks of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            print(f"  Upserted points {i + 1} to {min(i + batch_size, len(points))} / {len(points)}")

        print(f"Successfully indexed and persisted {len(points)} chunks in Qdrant collection '{self.collection_name}'.")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_source_type: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Embed query and retrieve top_k most similar chunks from Qdrant with cosine similarity.
        """
        query_vec = self.embedder.embed_query(query)
        vector_list = query_vec.tolist() if isinstance(query_vec, np.ndarray) else list(query_vec)

        query_filter = None
        if filter_source_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=filter_source_type)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector_list,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        results: List[Tuple[Chunk, float]] = []
        for pt in response.points:
            p = pt.payload or {}
            chunk = Chunk(
                chunk_id=p.get("chunk_id", str(pt.id)),
                doc_id=p.get("doc_id", ""),
                text=p.get("text", ""),
                source_type=p.get("source_type", "unknown"),
                source_name=p.get("source_name", "unknown"),
                metadata=p.get("metadata", {})
            )
            score = float(pt.score) if pt.score is not None else 0.0
            results.append((chunk, score))

        return results

    def count_points(self) -> int:
        """Return the number of points in the active collection."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        info = self.client.get_collection(self.collection_name)
        return getattr(info, "points_count", None) or getattr(info, "vectors_count", 0) or 0

    def close(self):
        """Close connection to client if applicable."""
        try:
            self.client.close()
        except Exception:
            pass

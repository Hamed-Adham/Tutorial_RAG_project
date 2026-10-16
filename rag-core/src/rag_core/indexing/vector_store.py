#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular Qdrant Vector Store for Unified RAG (Stage 4 / Stage 6).
Handles HNSW indexing, collection lifecycle, vector similarity search, and payload filtering.
"""

import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    Filter
)
from rag_core.config import settings


_SHARED_EMBEDDED_CLIENTS: Dict[str, QdrantClient] = {}


def get_qdrant_client(
    path: Optional[str] = None,
    url: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None
) -> QdrantClient:
    """
    Instantiate a QdrantClient.
    - If qdrant_mode == 'embedded' (default), directly open local disk storage (SQLite-like).
      Reuses shared instance per local path to avoid file lock contention in the same process.
    - If qdrant_mode == 'server', connect via URL or host:port with fallback to embedded.
    """
    local_path = path or settings.qdrant_path

    # 1. Embedded SQLite-like mode (Zero Docker required)
    if settings.qdrant_mode == "embedded":
        abs_path = str(Path(local_path).resolve())
        if abs_path in _SHARED_EMBEDDED_CLIENTS:
            return _SHARED_EMBEDDED_CLIENTS[abs_path]
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=local_path)
        _SHARED_EMBEDDED_CLIENTS[abs_path] = client
        return client

    # 2. Explicit URL
    target_url = url or settings.qdrant_url
    if target_url:
        return QdrantClient(url=target_url, api_key=api_key or settings.qdrant_api_key)

    # 3. Host / Port with fallback to embedded storage
    h = host or settings.qdrant_host
    p = port or settings.qdrant_port
    try:
        check_url = f"http://{h}:{p}/healthz"
        with urllib.request.urlopen(check_url, timeout=0.5) as resp:
            if resp.status == 200:
                return QdrantClient(host=h, port=p, api_key=api_key or settings.qdrant_api_key)
    except Exception:
        pass

    abs_path = str(Path(local_path).resolve())
    if abs_path in _SHARED_EMBEDDED_CLIENTS:
        return _SHARED_EMBEDDED_CLIENTS[abs_path]
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=local_path)
    _SHARED_EMBEDDED_CLIENTS[abs_path] = client
    return client


class QdrantVectorStore:
    """Manages collection initialization, HNSW configurations, vector upserts and queries."""

    def __init__(
        self,
        collection_name: str = settings.qdrant_collection_name,
        vector_dim: int = 1024,
        distance: Distance = Distance.COSINE,
        client: Optional[QdrantClient] = None
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.distance = distance
        self.client = client or get_qdrant_client()
        self.init_collection()

    def init_collection(self):
        """Create or verify collection with HNSW parameters."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=self.distance
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100
                )
            )

    def count(self) -> int:
        """Return total vectors in collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def upsert_points(self, points: List[PointStruct]):
        """Upsert a batch of points."""
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None
    ) -> List[Any]:
        """Perform dense vector search with optional payload filter and score cutoff."""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter
        )

    def close(self):
        """Close connection to Qdrant."""
        if hasattr(self.client, "close"):
            self.client.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Indexer for 7th-Grade Science Dense Vectors into Qdrant.
Reads ScienceChunk objects from data_index/science_bm25.pkl (or data/vlm_parsed/manifest.json),
embeds them via BgeM3Embedder (dim=1024), and indexes them with cosine similarity into Qdrant.
Caches generated embeddings to data_index/science_dense_index.npz for sub-second reindexing.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rag_core.config import settings
from rag_core.indexing.bm25_index import ScienceBM25Index
from rag_core.indexing.science_schema import ScienceChunk
from rag_core.naive.embedder import BgeM3Embedder
from rag_core.naive.qdrant_retriever import create_qdrant_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index_science_qdrant")


def load_science_chunks(bm25_path: Path) -> List[ScienceChunk]:
    """Load pre-parsed ScienceChunk objects from serialized BM25 index."""
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run build_science_bm25.py first.")

    index = ScienceBM25Index.load(bm25_path)
    logger.info(f"Loaded {len(index.chunks)} science chunks from {bm25_path}")
    return index.chunks


def index_chunks_into_qdrant(
    chunks: List[ScienceChunk],
    collection_name: str = "science_textbook_dense",
    client: QdrantClient = None,
    cache_path: Path = None,
    force_reindex: bool = False,
    batch_size: int = 64
) -> int:
    """Embed and index science chunks into Qdrant with payload."""
    client = client or create_qdrant_client()
    cache_path = cache_path or (settings.workspace_root / "data_index" / "science_dense_index.npz")

    # Check if collection exists and has points
    exists = client.collection_exists(collection_name)
    if exists and not force_reindex:
        info = client.get_collection(collection_name)
        points_count = getattr(info, "points_count", None) or getattr(info, "vectors_count", 0)
        if points_count == len(chunks):
            logger.info(f"Collection '{collection_name}' already populated with {points_count} points. Reusing index.")
            return points_count

    embedder = BgeM3Embedder()
    vector_dim = 1024

    cached_embeddings = None
    if cache_path.exists() and not force_reindex:
        try:
            data = np.load(str(cache_path))
            raw_embs = data["embeddings"]
            if len(raw_embs) == len(chunks):
                logger.info(f"Reusing {len(chunks)} pre-computed embeddings from {cache_path.name}")
                cached_embeddings = raw_embs
        except Exception as e:
            logger.warning(f"Could not load cached embeddings: {e}. Re-embedding.")

    if cached_embeddings is not None:
        embeddings = cached_embeddings
    else:
        logger.info(f"Generating BGE-M3 embeddings for {len(chunks)} science chunks...")
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts, batch_size=batch_size)

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_path), embeddings=embeddings)
        logger.info(f"Cached {len(embeddings)} embeddings to {cache_path}")

    # Recreate / initialize collection
    if exists:
        logger.info(f"Recreating collection '{collection_name}'...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
    )
    logger.info(f"Created Qdrant collection '{collection_name}' (dim={vector_dim}, distance=Cosine)")

    # Prepare Qdrant points
    points: List[PointStruct] = []
    for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        vector_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "section_id": chunk.section_id,
            "chapter": chunk.chapter,
            "page": chunk.page,
            "breadcrumbs": chunk.breadcrumbs,
            "text": chunk.text,
            "source_type": "science_textbook",
            "source_name": "science_textbook_g7.pdf",
            "token_count": chunk.token_count,
            "metadata": chunk.metadata
        }
        points.append(PointStruct(id=idx, vector=vector_list, payload=payload))

    # Batch upsert
    upsert_batch_size = 100
    for i in range(0, len(points), upsert_batch_size):
        batch = points[i : i + upsert_batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(f"  Upserted points {i + 1} to {min(i + upsert_batch_size, len(points))} / {len(points)}")

    logger.info(f"Successfully indexed {len(points)} chunks into Qdrant collection '{collection_name}'")
    return len(points)


def main():
    parser = argparse.ArgumentParser(description="Index 7th-grade science chunks into Qdrant.")
    parser.add_argument(
        "--bm25-index",
        default="data_index/science_bm25.pkl",
        help="Path to serialized BM25 index containing ScienceChunk objects."
    )
    parser.add_argument(
        "--collection",
        default="science_textbook_dense",
        help="Target Qdrant collection name."
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-embedding and recreation of collection."
    )
    args = parser.parse_args()

    bm25_path = Path(args.bm25_index)
    chunks = load_science_chunks(bm25_path)
    count = index_chunks_into_qdrant(
        chunks=chunks,
        collection_name=args.collection,
        force_reindex=args.force_reindex
    )
    print(f"Done! {count} chunks indexed in Qdrant collection '{args.collection}'.")


if __name__ == "__main__":
    main()

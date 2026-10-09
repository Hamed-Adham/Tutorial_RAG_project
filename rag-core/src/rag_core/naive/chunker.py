#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naive fixed-character chunker.
Implements naive sliding window chunking without hierarchical heading awareness.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from rag_core.naive.document_loader import Document

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_type: str
    source_name: str
    metadata: Dict[str, Any]

def chunk_text_fixed_window(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """Split text into fixed-size sliding character windows."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Chunk]:
    """
    Chunk documents according to naive strategy:
    - Administrative FAQ: Kept as atomic single chunks (since each row is self-contained QA).
    - Science Textbook: Fixed character window sliding chunks.
    """
    all_chunks = []
    for doc in documents:
        if doc.source_type == "administrative_faq":
            # Atomic chunk for FAQ items
            all_chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_c0",
                doc_id=doc.doc_id,
                text=doc.text,
                source_type=doc.source_type,
                source_name=doc.source_name,
                metadata=doc.metadata
            ))
        else:
            # Fixed window for textbook pages
            text_chunks = chunk_text_fixed_window(doc.text, chunk_size=chunk_size, overlap=overlap)
            for i, chunk_text in enumerate(text_chunks):
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = i
                all_chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_c{i:02d}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    source_type=doc.source_type,
                    source_name=doc.source_name,
                    metadata=chunk_meta
                ))

    return all_chunks

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build and Index the Isolated Naive Vector Database ('naive_baseline_chunks').

Implements:
1. FAQ points: Embeds ONLY row['question'] using BgeM3Embedder.
   Stores category, question, answer, citation, and gold_identifier in payload.
2. Science Textbook points: Embeds chunk text, computes exact book_page = start_page + pure_page - 1,
   and formats citation as 'فصل {chapter_num} - {chapter_title} – صفحه {book_page}'.
3. Stores points with payload indexes in Qdrant ('naive_baseline_chunks')
   and updates data_index/naive_index.npz + naive_metadata.json for NumPy fallback.
"""

import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

from rag_core.config import settings
from rag_core.naive.embedder import BgeM3Embedder
from rag_core.naive.chunker import chunk_text_fixed_window, Chunk
from rag_core.postprocessing.citations import DOC_TO_CHAPTER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_naive_database")


def extract_page_range_from_filename(filename: str) -> Tuple[int, int]:
    """
    Extract start_page and end_page from filename, e.g.
    'C706_olom_haftom-7-144-86-97.pdf' -> (86, 97)
    """
    m = re.search(r'-(\d+)-(\d+)\.pdf$', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1, 10


def build_naive_records(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load and structure all FAQ entries and Science Textbook chunks.
    Returns: (records_list, texts_to_embed)
    """
    records: List[Dict[str, Any]] = []
    texts_to_embed: List[str] = []

    # 1. Process FAQ CSV (data/faq_qa_pairs.csv)
    faq_path = data_dir / "faq_qa_pairs.csv"
    if faq_path.exists():
        logger.info(f"Loading FAQ records from {faq_path}...")
        with open(faq_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                category = (row.get("category") or "").strip()
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                if not question:
                    continue

                chunk_id = f"faq_row_{idx:03d}"
                citation = f"پرسش‌های متداول | دسته: {category} – {question}"
                gold_id = f"faq_qa_pairs.csv#row={idx}"
                full_text = f"دسته‌بندی: {category}\nپرسش: {question}\nپاسخ: {answer}"

                record = {
                    "chunk_id": chunk_id,
                    "doc_id": chunk_id,
                    "source_type": "administrative_faq",
                    "source_name": "faq_qa_pairs.csv",
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "chapter": 0,
                    "chapter_title": "",
                    "page": idx + 1,
                    "book_page": idx + 1,
                    "citation": citation,
                    "gold_identifier": gold_id,
                    "text": full_text,
                    "metadata": {
                        "category": category,
                        "question": question,
                        "answer": answer,
                        "row_index": idx,
                        "gold_identifier": gold_id
                    }
                }
                records.append(record)
                # CRITICAL (Point 2): Embed ONLY the question for FAQ items
                texts_to_embed.append(question)
        logger.info(f"Prepared {len(records)} FAQ records (embedded strictly by question).")

    # 2. Process Science Textbook PDFs (data/Olum_7_splitted)
    pdf_dir = data_dir / "Olum_7_splitted"
    if pdf_dir.exists():
        logger.info(f"Loading Science Textbook PDFs from {pdf_dir}...")
        try:
            import pypdf
            reader_cls = pypdf.PdfReader
        except ImportError:
            import PyPDF2
            reader_cls = PyPDF2.PdfReader

        pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
        sci_count = 0
        for pdf_path in pdf_files:
            filename = pdf_path.name
            start_page, end_page = extract_page_range_from_filename(filename)
            ch_num, ch_title = DOC_TO_CHAPTER.get(filename, (1, "علوم تجربی هفتم"))

            try:
                reader = reader_cls(str(pdf_path))
                for page_idx, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text() or ""
                    page_text = page_text.strip()
                    if not page_text:
                        continue

                    # Book page calculation: start_page + pure_page_number - 1
                    book_page = start_page + page_idx - 1
                    citation = f"فصل {ch_num} - {ch_title} – صفحه {book_page}"
                    gold_id = f"{filename}#page={page_idx}"

                    # Sliced fixed window chunks
                    chunks = chunk_text_fixed_window(page_text, chunk_size=500, overlap=50)
                    for c_idx, chunk_text in enumerate(chunks):
                        c_id = f"{pdf_path.stem}_p{page_idx:02d}_c{c_idx:02d}"
                        rec = {
                            "chunk_id": c_id,
                            "doc_id": f"{pdf_path.stem}_p{page_idx:02d}",
                            "source_type": "science_textbook",
                            "source_name": filename,
                            "category": "",
                            "question": "",
                            "answer": "",
                            "chapter": ch_num,
                            "chapter_title": ch_title,
                            "page": page_idx,           # pure relative page for PDF viewer opening
                            "book_page": book_page,     # actual printed textbook page for visual guidance
                            "citation": citation,       # e.g. "فصل 10 - گرما و بهینه‌سازی مصرف انرژی – صفحه 86"
                            "gold_identifier": gold_id,
                            "text": chunk_text,
                            "metadata": {
                                "chapter": ch_num,
                                "chapter_title": ch_title,
                                "page": page_idx,
                                "page_number": page_idx,
                                "book_page": book_page,
                                "filename": filename,
                                "file_path": str(pdf_path),
                                "gold_identifier": gold_id,
                                "citation": citation
                            }
                        }
                        records.append(rec)
                        texts_to_embed.append(chunk_text)
                        sci_count += 1
            except Exception as e:
                logger.warning(f"Could not read {filename}: {e}")

        logger.info(f"Prepared {sci_count} Science Textbook chunks across 15 chapters.")

    return records, texts_to_embed


def populate_naive_database(force_reindex: bool = True):
    """
    Build vectors, upsert to Qdrant collection 'naive_baseline_chunks',
    and persist data_index/naive_index.npz + naive_metadata.json.
    """
    records, texts_to_embed = build_naive_records(settings.data_dir)
    logger.info(f"Total naive records: {len(records)}. Embedding via BgeM3Embedder...")

    embedder = BgeM3Embedder()
    embeddings = embedder.embed_documents(texts_to_embed)
    logger.info(f"Generated {len(embeddings)} vectors with dimension {embeddings.shape[1]}.")

    # 1. Update data_index/naive_index.npz and naive_metadata.json
    cache_dir = settings.workspace_root / "data_index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "naive_index.npz"
    meta_path = cache_dir / "naive_metadata.json"

    np.savez_compressed(str(npz_path), embeddings=embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved cache to {npz_path} and {meta_path}.")

    # 2. Upsert to Qdrant if accessible
    collection_name = getattr(settings, "naive_qdrant_collection_name", "naive_baseline_chunks")
    try:
        from rag_core.naive.qdrant_retriever import create_qdrant_client
        client = create_qdrant_client()

        # Recreate collection
        vector_dim = embeddings.shape[1]
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection '{collection_name}' (dim={vector_dim}).")

        # Create payload schema indexes for fast pre-filtering
        try:
            client.create_payload_index(collection_name=collection_name, field_name="source_type", field_schema=PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name=collection_name, field_name="category", field_schema=PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name=collection_name, field_name="chapter", field_schema=PayloadSchemaType.INTEGER)
            logger.info("Created payload indexes on source_type, category, and chapter.")
        except Exception as ie:
            logger.warning(f"Payload index creation note: {ie}")

        # Upsert in batches of 100
        points: List[PointStruct] = []
        for idx, (rec, vec) in enumerate(zip(records, embeddings)):
            vector_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            points.append(PointStruct(id=idx, vector=vector_list, payload=rec))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"Upserted {min(i + batch_size, len(points))} / {len(points)} to Qdrant.")

        logger.info(f"Successfully populated Qdrant collection '{collection_name}' with {len(points)} points!")
    except Exception as qe:
        logger.warning(f"Could not populate Qdrant directly (likely locked by server process): {qe}. In-memory NPZ cache is ready and in sync.")


if __name__ == "__main__":
    populate_naive_database()

"""
CLI Builder for Administrative FAQ BM25 Lexical Index.
Reads administrative FAQ data, extracts QA items with category metadata,
and serializes an in-memory BM25 index artifact to data_index/faq_bm25.pkl.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from rag_core.indexing.science_schema import RetrievalCandidate
from rag_core.indexing.bm25_index import ScienceBM25Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_faq_bm25")


def load_faq_chunks(metadata_path: Path) -> List[RetrievalCandidate]:
    """Load and format FAQ chunks from naive_metadata.json into RetrievalCandidate objects."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    faq_candidates: List[RetrievalCandidate] = []
    for item in data:
        if item.get("source_type") != "administrative_faq":
            continue

        chunk_id = str(item.get("chunk_id", f"faq_{len(faq_candidates)}"))
        category = item.get("category", "عمومی")
        question = item.get("question", "")
        answer = item.get("answer", "")
        row_idx = item.get("metadata", {}).get("row_index", item.get("page", 0))
        text = item.get("text") or f"دسته‌بندی: {category}\nپرسش: {question}\nپاسخ: {answer}"
        breadcrumbs = f"پرسش‌های متداول > {category} > {question}"

        cand = RetrievalCandidate(
            chunk_id=chunk_id,
            doc_id="administrative_faq",
            source_type="administrative_faq",
            section_id=f"faq_cat_{category}",
            chapter=0,
            page=row_idx,
            category=category,
            row=row_idx,
            text=text,
            breadcrumbs=breadcrumbs
        )
        faq_candidates.append(cand)

    logger.info("Loaded %d FAQ items from %s", len(faq_candidates), metadata_path)
    return faq_candidates


def build_faq_index(
    input_path: Path = Path("data_index/naive_metadata.json"),
    output_path: Path = Path("data_index/faq_bm25.pkl")
) -> Path:
    """Build and serialize FAQ BM25 index artifact."""
    logger.info("Building FAQ BM25 index from %s -> %s", input_path, output_path)
    chunks = load_faq_chunks(input_path)
    if not chunks:
        raise ValueError(f"No administrative FAQ chunks found in {input_path}")

    index = ScienceBM25Index()
    index.add_chunks(chunks)
    index.build()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(output_path)
    logger.info("Successfully persisted FAQ BM25 index with %d chunks to %s", len(chunks), output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build Administrative FAQ BM25 index.")
    parser.add_argument("--input", type=str, default="data_index/naive_metadata.json", help="Path to naive_metadata.json")
    parser.add_argument("--output", type=str, default="data_index/faq_bm25.pkl", help="Path to save BM25 pickle artifact")
    args = parser.parse_args()

    build_faq_index(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

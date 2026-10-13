"""
CLI Builder for 7th-Grade Science BM25 Lexical Index.
Reads data/vlm_parsed/manifest.json across all 15 curriculum chapters, creates atomic ScienceChunk
objects with heading breadcrumbs, and persists an in-memory ScienceBM25Index artifact.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from rag_core.indexing.science_schema import ScienceChunk
from rag_core.indexing.bm25_index import ScienceBM25Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_science_bm25")


def extract_chapter_number(doc_name: str, content: str) -> int:
    """Extract chapter number (1 to 15) from document name or heading content."""
    # Check for chapter heading in content: e.g. '# فصل ۱' or '# فصل 1'
    match = re.search(r'#\s*فصل\s*([۰-۹0-9]+)', content)
    if match:
        raw_num = match.group(1)
        # Persian digit conversion
        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        latin_num = ""
        for char in raw_num:
            if char in persian_digits:
                latin_num += str(persian_digits.index(char))
            else:
                latin_num += char
        try:
            return int(latin_num)
        except ValueError:
            pass

    # Fallback to document numbering in filename if available
    match = re.search(r'haftom-(\d+)', doc_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return 1


def chunk_section(section: Dict[str, Any], doc_name: str, chapter: int) -> List[ScienceChunk]:
    """Break a curriculum section into atomic ~512 token child chunks with breadcrumbs."""
    content = section.get("content", "").strip()
    if not content:
        return []

    section_id = section.get("section_id", "SEC-UNKNOWN")
    page = section.get("page", 1)
    title = section.get("title") or f"بخش {section_id}"
    tag = section.get("tag") or section.get("section_type", "main")

    # Extract primary heading if available
    first_line = content.split("\n")[0]
    heading = first_line.replace("#", "").strip() if first_line.startswith("#") else title
    breadcrumbs = f"فصل {chapter} > {heading}"

    # Approximate token count by word count
    words = content.split()
    chunk_size_words = 350  # ~512 tokens

    if len(words) <= chunk_size_words:
        chunk = ScienceChunk(
            chunk_id=f"{section_id}_c0",
            doc_id=doc_name,
            section_id=section_id,
            chapter=chapter,
            page=page,
            breadcrumbs=breadcrumbs,
            text=content,
            token_count=len(words),
            metadata={"tag": tag, "is_pedagogical": section.get("is_pedagogical_activity", False)}
        )
        return [chunk]

    # Split long sections into overlapping sub-chunks
    chunks = []
    stride = 280
    for i in range(0, len(words), stride):
        sub_words = words[i:i + chunk_size_words]
        sub_text = " ".join(sub_words)
        chunk = ScienceChunk(
            chunk_id=f"{section_id}_c{len(chunks)}",
            doc_id=doc_name,
            section_id=section_id,
            chapter=chapter,
            page=page,
            breadcrumbs=breadcrumbs,
            text=sub_text,
            token_count=len(sub_words),
            metadata={"tag": tag, "is_pedagogical": section.get("is_pedagogical_activity", False)}
        )
        chunks.append(chunk)
        if i + chunk_size_words >= len(words):
            break

    return chunks


def build_bm25_from_manifest(manifest_path: str | Path, output_path: str | Path) -> ScienceBM25Index:
    """Build ScienceBM25Index from VLM parsed manifest and persist artifact."""
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    documents = manifest_data.get("documents", [])
    logger.info(f"Loaded {len(documents)} documents from manifest: {manifest_path}")

    all_chunks: List[ScienceChunk] = []
    chapter_counts: Dict[int, int] = {}

    for doc_idx, doc in enumerate(documents, start=1):
        doc_name = doc.get("document", f"doc_{doc_idx}")
        sections = doc.get("sections", [])
        # Determine chapter
        first_content = sections[0].get("content", "") if sections else ""
        chapter = extract_chapter_number(doc_name, first_content)
        # If extraction fell back to 1 and doc_idx != 1, use doc_idx as chapter fallback
        if chapter == 1 and doc_idx > 1 and doc_idx <= 15:
            chapter = doc_idx

        chapter_counts[chapter] = chapter_counts.get(chapter, 0) + len(sections)

        for sec in sections:
            chunks = chunk_section(sec, doc_name, chapter)
            all_chunks.extend(chunks)

    logger.info(f"Assembled {len(all_chunks)} atomic child chunks across {len(chapter_counts)} chapters.")
    for ch in sorted(chapter_counts.keys()):
        logger.info(f"  Chapter {ch:02d}: {chapter_counts[ch]} sections")

    index = ScienceBM25Index()
    logger.info("Tokenizing and building BM25 inverted index...")
    index.add_chunks(all_chunks)
    index.build()

    out_file = Path(output_path)
    index.save(out_file)
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully saved ScienceBM25Index to {out_file} ({file_size_mb:.2f} MB)")
    return index


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for 7th-grade science textbook.")
    parser.add_argument("--manifest", default="data/vlm_parsed/manifest.json", help="Path to parsed manifest JSON.")
    parser.add_argument("--output", default="data_index/science_bm25.pkl", help="Output path for serialized index.")
    args = parser.parse_args()

    build_bm25_from_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()

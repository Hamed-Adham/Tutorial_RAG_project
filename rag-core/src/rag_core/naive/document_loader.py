#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naive document loader for Science Textbook PDFs and Administrative FAQ CSV.
Extracts raw text without complex OCR or layout reconstruction (baseline approach).
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Document:
    doc_id: str
    text: str
    source_type: str  # 'science_textbook' or 'administrative_faq'
    source_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

from rag_core.postprocessing.citations import DOC_TO_CHAPTER
import re

def extract_page_range_from_filename(filename: str) -> tuple[int, int]:
    m = re.search(r'-(\d+)-(\d+)\.pdf$', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1, 10

def load_faq_csv(csv_path: Path) -> List[Document]:
    """Load Q&A pairs from INIF FAQ CSV into Document objects."""
    if not csv_path.exists():
        raise FileNotFoundError(f"FAQ CSV not found at {csv_path}")

    documents = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            category = (row.get("category") or "").strip()
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question and not answer:
                continue

            text_content = f"دسته‌بندی: {category}\nپرسش: {question}\nپاسخ: {answer}"
            citation = f"پرسش‌های متداول | دسته: {category} – {question}"
            gold_id = f"faq_qa_pairs.csv#row={i-1}"
            doc = Document(
                doc_id=f"faq_row_{i:03d}",
                text=text_content,
                source_type="administrative_faq",
                source_name="faq_qa_pairs.csv",
                metadata={
                    "row_index": i,
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "chapter": 0,
                    "chapter_title": "",
                    "page": i,
                    "book_page": i,
                    "citation": citation,
                    "gold_identifier": gold_id
                }
            )
            documents.append(doc)

    return documents

def load_pdf_documents(pdf_dir: Path) -> List[Document]:
    """Load textbook chapter PDFs page-by-page into Document objects with exact book citations."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found at {pdf_dir}")

    try:
        import pypdf
        reader_cls = pypdf.PdfReader
    except ImportError:
        try:
            import PyPDF2
            reader_cls = PyPDF2.PdfReader
        except ImportError:
            reader_cls = None

    documents = []
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))

    for pdf_path in pdf_files:
        filename = pdf_path.name
        start_page, end_page = extract_page_range_from_filename(filename)
        ch_num, ch_title = DOC_TO_CHAPTER.get(filename, (1, "علوم تجربی هفتم"))

        if reader_cls is None:
            documents.append(Document(
                doc_id=f"{pdf_path.stem}_placeholder",
                text=f"Chapter {pdf_path.stem} text extraction pending reader dependency.",
                source_type="science_textbook",
                source_name=filename,
                metadata={
                    "chapter": ch_num,
                    "chapter_title": ch_title,
                    "page": 1,
                    "book_page": start_page,
                    "citation": f"فصل {ch_num} - {ch_title} – صفحه {start_page}",
                    "file_path": str(pdf_path)
                }
            ))
            continue

        try:
            reader = reader_cls(str(pdf_path))
            for page_idx, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                # Actual textbook page number = start_page + pure_page_idx - 1
                book_page = start_page + page_idx - 1
                citation = f"فصل {ch_num} - {ch_title} – صفحه {book_page}"
                gold_id = f"{filename}#page={page_idx}"

                doc = Document(
                    doc_id=f"{pdf_path.stem}_p{page_idx:02d}",
                    text=page_text,
                    source_type="science_textbook",
                    source_name=filename,
                    metadata={
                        "chapter": ch_num,
                        "chapter_title": ch_title,
                        "page": page_idx,           # pure relative page for PDF viewer navigation (#page=X)
                        "page_number": page_idx,
                        "book_page": book_page,     # actual printed textbook page for visual citation
                        "citation": citation,       # e.g. "فصل 10 - گرما و بهینه‌سازی مصرف انرژی – صفحه 86"
                        "filename": filename,
                        "file_path": str(pdf_path),
                        "gold_identifier": gold_id
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")

    return documents

def load_all_documents(data_dir: Path) -> List[Document]:
    """Load both FAQ CSV and Science Textbook PDFs."""
    faq_path = data_dir / "faq_qa_pairs.csv"
    pdf_dir = data_dir / "Olum_7_splitted"

    all_docs = []
    if faq_path.exists():
        faq_docs = load_faq_csv(faq_path)
        all_docs.extend(faq_docs)
    if pdf_dir.exists():
        pdf_docs = load_pdf_documents(pdf_dir)
        all_docs.extend(pdf_docs)

    return all_docs

if __name__ == "__main__":
    from rag_core.config import settings
    docs = load_all_documents(settings.data_dir)
    print(f"Loaded {len(docs)} documents total:")
    faq_c = sum(1 for d in docs if d.source_type == "administrative_faq")
    pdf_c = sum(1 for d in docs if d.source_type == "science_textbook")
    print(f"  - Administrative FAQ: {faq_c} records")
    print(f"  - Science Textbook Pages: {pdf_c} pages")

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
            doc = Document(
                doc_id=f"faq_row_{i:03d}",
                text=text_content,
                source_type="administrative_faq",
                source_name="faq_qa_pairs.csv",
                metadata={
                    "row_index": i,
                    "category": category,
                    "question": question,
                    "answer": answer
                }
            )
            documents.append(doc)

    return documents

def load_pdf_documents(pdf_dir: Path) -> List[Document]:
    """Load textbook chapter PDFs page-by-page into Document objects."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found at {pdf_dir}")

    # Try importing pypdf or PyPDF2
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
        chapter_name = pdf_path.stem
        if reader_cls is None:
            # Fallback if PDF reader not yet available
            documents.append(Document(
                doc_id=f"{chapter_name}_placeholder",
                text=f"Chapter {chapter_name} text extraction pending reader dependency.",
                source_type="science_textbook",
                source_name=pdf_path.name,
                metadata={"file_path": str(pdf_path)}
            ))
            continue

        try:
            reader = reader_cls(str(pdf_path))
            for page_idx, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                doc = Document(
                    doc_id=f"{chapter_name}_p{page_idx:02d}",
                    text=page_text,
                    source_type="science_textbook",
                    source_name=pdf_path.name,
                    metadata={
                        "chapter": chapter_name,
                        "page_number": page_idx,
                        "file_path": str(pdf_path)
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not read {pdf_path.name}: {e}")

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

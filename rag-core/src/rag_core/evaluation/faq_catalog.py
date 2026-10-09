#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical Catalog Mapper and Ingestion Engine for Administrative FAQs.
Assigns deterministic permanent IDs (faq_000 to faq_133) across data/faq_qa_pairs.csv.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from rag_core.config import settings


@dataclass
class FAQEntry:
    doc_id: str
    row_index: int
    category: str
    question: str
    answer: str

    @property
    def chunk_identifier(self) -> str:
        return f"faq_qa_pairs.csv#row={self.row_index}"

    @property
    def full_text(self) -> str:
        return f"دسته‌بندی: {self.category}\nسوال: {self.question}\nپاسخ: {self.answer}"


class FAQCatalog:
    """
    Catalog of authoritative Administrative FAQ documents.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or (settings.data_dir / "faq_qa_pairs.csv")
        self._entries: Dict[str, FAQEntry] = {}
        self._categories: Dict[str, List[FAQEntry]] = {}
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"FAQ CSV not found at {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                doc_id = f"faq_{idx:03d}"
                entry = FAQEntry(
                    doc_id=doc_id,
                    row_index=idx,
                    category=row.get("category", "").strip(),
                    question=row.get("question", "").strip(),
                    answer=row.get("answer", "").strip(),
                )
                self._entries[doc_id] = entry
                self._categories.setdefault(entry.category, []).append(entry)

    def get(self, doc_id: str) -> Optional[FAQEntry]:
        return self._entries.get(doc_id)

    def get_by_index(self, row_idx: int) -> Optional[FAQEntry]:
        doc_id = f"faq_{row_idx:03d}"
        return self._entries.get(doc_id)

    def all_entries(self) -> List[FAQEntry]:
        return list(self._entries.values())

    def categories(self) -> List[str]:
        return list(self._categories.keys())

    def by_category(self, category: str) -> List[FAQEntry]:
        return self._categories.get(category, [])

    def __len__(self) -> int:
        return len(self._entries)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curriculum Section Grounder for 15-Chapter Persian Science Textbook.
Reads data/vlm_parsed/manifest.json and structured Markdown to provide
precise chapter, section, and pedagogical tag grounding across all 412 sections.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from rag_core.config import settings


@dataclass
class TextbookSection:
    global_id: str
    local_id: str
    chapter_num: int
    source_pdf: str
    page: int
    section_type: str
    tag: Optional[str]
    content: str
    is_main_text: bool
    is_pedagogical_activity: bool

    @property
    def chunk_identifier(self) -> str:
        pdf_name = Path(self.source_pdf).name
        return f"{pdf_name}#page={self.page}"


class TextbookGrounder:
    """
    Curriculum grounder indexing all 412 sections across 15 chapters.
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or (settings.data_dir / "vlm_parsed" / "manifest.json")
        self._sections: Dict[str, TextbookSection] = {}
        self._chapters: Dict[int, List[TextbookSection]] = {}
        self._load()

    def _load(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = data.get("documents", [])
        for doc_idx, doc in enumerate(documents, start=1):
            source_pdf = doc.get("source_pdf", f"chapter_{doc_idx}.pdf")
            sections = doc.get("sections", [])
            for sec in sections:
                local_id = sec.get("section_id", "")
                global_id = f"CH{doc_idx:02d}-{local_id}"
                page = sec.get("page", 1)
                item = TextbookSection(
                    global_id=global_id,
                    local_id=local_id,
                    chapter_num=doc_idx,
                    source_pdf=source_pdf,
                    page=page,
                    section_type=sec.get("section_type", "main_body"),
                    tag=sec.get("tag"),
                    content=sec.get("content", ""),
                    is_main_text=sec.get("is_main_text", True),
                    is_pedagogical_activity=sec.get("is_pedagogical_activity", False),
                )
                self._sections[global_id] = item
                self._chapters.setdefault(doc_idx, []).append(item)

    def total_sections(self) -> int:
        return len(self._sections)

    def chapters(self) -> List[int]:
        return sorted(list(self._chapters.keys()))

    def get_sections(self, chapter_num: int) -> List[TextbookSection]:
        return self._chapters.get(chapter_num, [])

    def get_experiments(self, chapter_num: Optional[int] = None) -> List[TextbookSection]:
        if chapter_num:
            return [s for s in self.get_sections(chapter_num) if s.is_pedagogical_activity]
        return [s for s in self._sections.values() if s.is_pedagogical_activity]

    def get_main_body(self, chapter_num: Optional[int] = None) -> List[TextbookSection]:
        if chapter_num:
            return [s for s in self.get_sections(chapter_num) if s.is_main_text]
        return [s for s in self._sections.values() if s.is_main_text]

    def get_by_id(self, global_id: str) -> Optional[TextbookSection]:
        return self._sections.get(global_id)

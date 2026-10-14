"""
Parent Section Expansion and LongContextReorder.
Dynamically resolves retrieved child chunks to full parent curriculum sections,
deduplicates shared sections, caps context at 2,500 tokens, and mitigates lost-in-the-middle degradation.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import structlog

from rag_core.config import settings
from rag_core.indexing.science_schema import (
    RetrievalCandidate,
    ParentSection,
    ExpandedContext
)

logger = structlog.get_logger(__name__)


class LongContextReorder:
    """Lost-in-the-middle mitigation reordering context items.
    
    Places highest-scoring items at the beginning and the end of the prompt window,
    with lower-scoring items in the middle.
    Disabled by default via settings.enable_context_reordering.
    """

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = settings.enable_context_reordering if enabled is None else enabled

    def reorder(self, items: List[Any], force: bool = False) -> List[Any]:
        """Alternating reorder placing best at 0, 2nd best at -1, and worst in the middle.
        If disabled (default), returns unmodified items preserving natural rank order."""
        if not (self.enabled or force) or len(items) <= 2:
            return list(items)

        result = [None] * len(items)
        left = 0
        right = len(items) - 1
        for i, item in enumerate(items):
            if i % 2 == 0:
                result[left] = item
                left += 1
            else:
                result[right] = item
                right -= 1
        return result


class ParentSectionExpander:
    """Expands fine-grained child chunks into complete, cohesive parent textbook sections."""

    def __init__(
        self,
        parent_catalog: Optional[Dict[str, ParentSection]] = None,
        manifest_path: Optional[str | Path] = None,
        max_parents: int = 3,
        max_tokens: int = 2500,
        reorderer: Optional[LongContextReorder] = None,
    ):
        self.max_parents = max_parents
        self.max_tokens = max_tokens
        self.reorderer = reorderer or LongContextReorder()

        if parent_catalog is not None:
            self.parent_catalog = parent_catalog
        else:
            self.parent_catalog = self._load_from_manifest(manifest_path)

    def _load_from_manifest(self, manifest_path: Optional[str | Path]) -> Dict[str, ParentSection]:
        """Load parent sections from manifest.json."""
        path = Path(manifest_path or "data/vlm_parsed/manifest.json")
        if not path.exists():
            logger.warning(f"Manifest not found at {path}, parent catalog empty")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        catalog = {}
        for doc_idx, doc in enumerate(data.get("documents", [])):
            doc_name = doc.get("document", "")
            chapter = doc_idx + 1
            for sec in doc.get("sections", []):
                sec_id = sec.get("section_id")
                if not sec_id:
                    continue

                content = sec.get("content", "")
                words = content.split()
                parent = ParentSection(
                    section_id=sec_id,
                    chapter=chapter,
                    title=sec.get("title") or f"بخش {sec_id}",
                    page_start=sec.get("page", 1),
                    page_end=sec.get("page", 1),
                    full_text=content,
                    token_count=len(words)
                )
                if doc_name:
                    catalog[f"{doc_name}_{sec_id}"] = parent
                catalog[f"{chapter}_{sec_id}"] = parent
                if sec_id not in catalog:
                    catalog[sec_id] = parent

        logger.info(f"Loaded {len(catalog)} parent sections into expansion catalog")
        return catalog

    def expand(
        self,
        candidates: List[Any],
        max_sections: Optional[int] = None,
        token_ceiling: Optional[int] = None
    ) -> ExpandedContext:
        """Dynamically expand candidates to unique parent sections with token budget capping."""
        if not candidates:
            return ExpandedContext(
                parent_sections=[],
                total_tokens=0,
                source_chunk_ids=[],
                reordered_texts=[]
            )

        max_p = max_sections if max_sections is not None else self.max_parents
        max_t = token_ceiling if token_ceiling is not None else self.max_tokens

        seen_sections = set()
        selected_parents: List[ParentSection] = []
        source_chunk_ids = []
        accumulated_tokens = 0

        for item in candidates:
            cand = item.candidate if hasattr(item, "candidate") else item
            source_chunk_ids.append(cand.chunk_id)
            sec_id = cand.section_id

            # Use composite section key to ensure uniqueness per chapter
            unique_sec_key = f"{cand.chapter}_{sec_id}"
            if unique_sec_key in seen_sections:
                continue

            parent = (
                self.parent_catalog.get(f"{cand.doc_id}_{sec_id}")
                or self.parent_catalog.get(f"{cand.chapter}_{sec_id}")
                or self.parent_catalog.get(sec_id)
            )
            if parent is None:
                # Fallback to candidate text as standalone parent section if section not in catalog
                parent = ParentSection(
                    section_id=sec_id,
                    chapter=cand.chapter,
                    title=cand.breadcrumbs,
                    page_start=cand.page,
                    page_end=cand.page,
                    full_text=cand.text,
                    token_count=len(cand.text.split())
                )

            seen_sections.add(unique_sec_key)

            # Check context token and count budgets
            if len(selected_parents) >= max_p:
                break
            if accumulated_tokens + parent.token_count > max_t and selected_parents:
                break

            if accumulated_tokens + parent.token_count > self.max_tokens and len(selected_parents) > 0:
                break

            seen_sections.add(sec_id)
            selected_parents.append(parent)
            accumulated_tokens += parent.token_count

        # Apply LongContextReorder to texts
        parent_texts = [p.full_text for p in selected_parents]
        reordered_texts = self.reorderer.reorder(parent_texts)

        return ExpandedContext(
            parent_sections=selected_parents,
            total_tokens=accumulated_tokens,
            source_chunk_ids=source_chunk_ids,
            reordered_texts=reordered_texts
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sub-Query Decomposition for Persian Science Curriculum Inquiries.
Splits composite, multi-concept, or comparison questions into atomic sub-queries
to enable multi-hop hybrid retrieval.
"""

import re
from typing import List, Optional


class SubQueryDecomposer:
    """
    Decomposes complex multi-part questions into focused atomic sub-queries.
    Uses regex clause boundary detection and Persian sentence structure patterns.
    """

    # Clause boundary patterns where two distinct questions or tasks are conjoined
    SPLIT_PATTERNS = [
        r"[؟\?]\s*",                                                    # Question mark split
        r"\s+و\s+(?:بگویید|توضیح\s+دهید|بیان\s+کنید|مشخص\s+کنید)\s+",   # "و بگویید / توضیح دهید"
        r"\s+و\s+همچنین\s+",                                            # "و همچنین"
        r"(?:چیست|کیست|است|دارد|می‌شود|می‌گردد|هستند|باشند)\s+و\s+",      # "... چیست و ..."
        r"را\s+مقایسه\s+کنید\s+و\s+",                                   # "... را مقایسه کنید و ..."
    ]

    def __init__(self):
        self._compiled_splits = [re.compile(p, re.IGNORECASE) for p in self.SPLIT_PATTERNS]

    def decompose(self, query: str) -> List[str]:
        """
        Decomposes a query into atomic sub-queries.
        Returns a single-item list if the query is already atomic.
        """
        cleaned = query.strip()
        if not cleaned:
            return []

        # Check for explicit compound indicators
        sub_queries: List[str] = []

        # 1. Try splitting by question marks first if multiple questions exist
        qmark_parts = [p.strip() for p in re.split(r"[؟\?]", cleaned) if p.strip()]
        if len(qmark_parts) > 1:
            for part in qmark_parts:
                normalized = self._normalize_subquery(part)
                if normalized:
                    sub_queries.append(normalized)
            if len(sub_queries) > 1:
                return sub_queries

        # 2. Try conjunction splitting
        for pattern in self._compiled_splits[1:]:
            parts = pattern.split(cleaned)
            if len(parts) > 1:
                accumulated = []
                for p in parts:
                    sub = self._normalize_subquery(p.strip())
                    if sub and len(sub) > 5:
                        accumulated.append(sub)
                if len(accumulated) > 1:
                    return self._contextualize_subqueries(accumulated, cleaned)

        # 3. If no compound pattern matched, query is atomic
        return [cleaned]

    def _normalize_subquery(self, sub: str) -> str:
        """Strip trailing punctuation, conjunctions, and clean up whitespace."""
        s = sub.strip()
        # Remove leading "و" or "همچنین"
        s = re.sub(r"^(?:و\s+|همچنین\s+)", "", s).strip()
        # Remove trailing question mark
        s = re.sub(r"[؟\?]+$", "", s).strip()
        return s

    def _contextualize_subqueries(self, parts: List[str], full_query: str) -> List[str]:
        """
        Ensure dependent sub-queries carry relevant subject context if they start with pronouns.
        For example: ['نقطه جوش آب خالص و آب نمک را مقایسه کنید', 'فشار هوا چه تاثیری بر آن دارد']
        -> converts 'آن' to refer to 'نقطه جوش'.
        """
        results: List[str] = []
        lead_subject = ""

        # Extract possible subject from part 0 (e.g., 'نقطه جوش' or 'سلول گیاهی')
        match = re.search(r"^([^\s]+(?:\s+[^\s]+){1,3})", parts[0])
        if match:
            lead_subject = match.group(1).strip()

        for idx, part in enumerate(parts):
            if idx == 0:
                results.append(part)
                continue

            # If part has unresolved pronouns like "بر آن", "در آن", "بر این", prepend context
            if re.search(r"\b(آن|این|آنها|این‌ها|آن‌ها)\b", part) and lead_subject:
                # Replace 'بر آن' with 'بر <lead_subject>' or augment
                augmented = re.sub(r"\bبر\s+آن\b", f"بر {lead_subject}", part)
                augmented = re.sub(r"\bدر\s+آن\b", f"در {lead_subject}", augmented)
                results.append(augmented)
            else:
                results.append(part)

        return results

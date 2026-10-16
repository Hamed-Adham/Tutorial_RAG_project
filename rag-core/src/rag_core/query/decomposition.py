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

    # Conjunction pattern where the second clause starts with an interrogative or directive word
    INTERROGATIVE_CONJUNCTION = re.compile(
        r"\s+و\s+(?=(?:چه|چگونه|چطور|کدام|چرا|آیا|کجا|کی|چه\s+زمانی|بگویید|توضیح\s+دهید|بیان\s+کنید|مشخص\s+کنید|نام\s+ببرید|ذکر\s+کنید)\b)",
        re.IGNORECASE
    )

    # Conjunction phrase like "و همچنین"
    CONJUNCTION_PHRASE = re.compile(r"\s+و\s+همچنین\s+", re.IGNORECASE)

    # Clause boundary verbs preceding "و"
    VERB_CONJUNCTION = re.compile(
        r"((?:چیست|کیست|است|دارد|می‌شود|میشود|می‌گردد|میگردد|هستند|باشند|می‌کند|میکند|کند|مقایسه\s+کنید))\s+و\s+",
        re.IGNORECASE
    )

    def decompose(self, query: str) -> List[str]:
        """
        Decomposes a query into atomic sub-queries.
        Returns a single-item list if the query is already atomic.
        """
        cleaned = query.strip().strip('"\'«»“”‘’')
        if not cleaned:
            return []

        # 1. Try splitting by question marks first if multiple questions exist
        qmark_parts = [p.strip() for p in re.split(r"[؟\?]", cleaned) if p.strip()]
        if len(qmark_parts) > 1:
            valid_subs = []
            for part in qmark_parts:
                normalized = self._normalize_subquery(part)
                # Ensure each sub-query has meaningful textual content (not just quotes or symbols)
                if normalized and len(normalized) >= 4 and re.search(r"[\u0600-\u06FF\w]", normalized):
                    valid_subs.append(normalized)
            if len(valid_subs) > 1:
                return self._contextualize_subqueries(valid_subs, cleaned)

        # 2. Try conjunction splitting on interrogative/directive words (e.g. "کار میکند و چه بخشهایی دارد")
        parts = self.INTERROGATIVE_CONJUNCTION.split(cleaned)
        if len(parts) > 1:
            accum = [self._normalize_subquery(p) for p in parts]
            valid_accum = [p for p in accum if p and len(p) >= 4 and re.search(r"[\u0600-\u06FF\w]", p)]
            if len(valid_accum) > 1:
                return self._contextualize_subqueries(valid_accum, cleaned)

        # 3. Try conjunction phrase ("و همچنین")
        parts = self.CONJUNCTION_PHRASE.split(cleaned)
        if len(parts) > 1:
            accum = [self._normalize_subquery(p) for p in parts]
            valid_accum = [p for p in accum if p and len(p) >= 4 and re.search(r"[\u0600-\u06FF\w]", p)]
            if len(valid_accum) > 1:
                return self._contextualize_subqueries(valid_accum, cleaned)

        # 4. Try verb conjunction splitting with captured verbs
        v_parts = self.VERB_CONJUNCTION.split(cleaned)
        if len(v_parts) >= 3:
            accum = []
            i = 0
            while i < len(v_parts):
                if i + 1 < len(v_parts) and (i % 2 == 0):
                    clause = f"{v_parts[i]} {v_parts[i+1]}".strip()
                    norm = self._normalize_subquery(clause)
                    if norm and len(norm) >= 4:
                        accum.append(norm)
                    i += 2
                else:
                    norm = self._normalize_subquery(v_parts[i])
                    if norm and len(norm) >= 4:
                        accum.append(norm)
                    i += 1
            if len(accum) > 1:
                return self._contextualize_subqueries(accum, cleaned)

        # 5. If no compound pattern matched, query is atomic
        return [cleaned]

    def _normalize_subquery(self, sub: str) -> str:
        """Strip trailing punctuation, quotes, conjunctions, and clean up whitespace."""
        s = sub.strip().strip('"\'«»“”‘’')
        # Remove leading "و" or "همچنین"
        s = re.sub(r"^(?:و\s+|همچنین\s+)", "", s).strip()
        # Remove trailing question mark and quotes
        s = re.sub(r"[؟\?]+$", "", s).strip().strip('"\'«»“”‘’')
        return s

    def _contextualize_subqueries(self, parts: List[str], full_query: str) -> List[str]:
        """
        Ensure dependent sub-queries carry relevant subject context if they start with pronouns
        or lack the lead subject.
        For example: ['دستگاه گوارش چگونه کار میکند', 'چه بخشهایی دارد']
        -> ['دستگاه گوارش چگونه کار میکند', 'دستگاه گوارش چه بخشهایی دارد']
        """
        results: List[str] = []
        lead_subject = ""

        # Extract possible subject from part 0 before question words (e.g. 'دستگاه گوارش' from 'دستگاه گوارش چگونه کار میکند')
        p0 = parts[0]
        split_match = re.split(r"\s+(?:چگونه|چطور|چیست|کیست|کدام|چرا|آیا|چه)\b", p0)
        if split_match and split_match[0].strip():
            lead_subject = split_match[0].strip()
        else:
            match = re.search(r"^([^\s]+(?:\s+[^\s]+){0,2})", p0)
            if match:
                lead_subject = match.group(1).strip()

        for idx, part in enumerate(parts):
            if idx == 0:
                results.append(part)
                continue

            # If part has unresolved pronouns like "بر آن", "در آن", "بر این", prepend context
            if re.search(r"\b(آن|این|آنها|این‌ها|آن‌ها)\b", part) and lead_subject:
                augmented = re.sub(r"\bبر\s+آن\b", f"بر {lead_subject}", part)
                augmented = re.sub(r"\bدر\s+آن\b", f"در {lead_subject}", augmented)
                results.append(augmented)
            elif lead_subject and lead_subject not in part and len(part.split()) <= 4:
                # Prepend lead subject to short dependent subquery like 'چه بخشهایی دارد'
                results.append(f"{lead_subject} {part}")
            else:
                results.append(part)

        return results

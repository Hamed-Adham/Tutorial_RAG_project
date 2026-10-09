#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic Zero-Token Retrieval Scorer (US3).
Evaluates Hit@1, Hit@3, Hit@5, MRR, and Context Recall independently from LLM generation.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Optional
from rag_core.evaluation.benchmark_schema import RetrievalMetricsModel


@dataclass
class RetrievalQueryEvaluation:
    """Evaluation result for a single retrieval query."""
    gold_chunks: List[str]
    retrieved_chunks: List[str]
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    context_recall: float = 0.0

    def __post_init__(self):
        # Auto-calculate if not provided and chunks are present
        if self.hit_at_1 == 0.0 and self.hit_at_3 == 0.0 and self.hit_at_5 == 0.0 and self.mrr == 0.0 and self.context_recall == 0.0:
            if not self.gold_chunks:
                self.hit_at_1 = self.hit_at_3 = self.hit_at_5 = self.mrr = self.context_recall = 1.0
                return

            gold_set = set(g.strip().lower() for g in self.gold_chunks if g)
            first_hit_rank: Optional[int] = None
            matched = set()

            for idx, ret in enumerate(self.retrieved_chunks[:5], start=1):
                r = ret.strip().lower()
                for g in gold_set:
                    if g in r or r in g:
                        if first_hit_rank is None:
                            first_hit_rank = idx
                        matched.add(g)

            if first_hit_rank is not None:
                self.hit_at_1 = 1.0 if first_hit_rank <= 1 else 0.0
                self.hit_at_3 = 1.0 if first_hit_rank <= 3 else 0.0
                self.hit_at_5 = 1.0 if first_hit_rank <= 5 else 0.0
                self.mrr = 1.0 / first_hit_rank
            if gold_set:
                self.context_recall = len(matched) / len(gold_set)



class RetrievalScorer:
    """
    Deterministic retrieval scorer for computing top-k hit rates, MRR, and context recall.
    Runs at 0 tokens and sub-second execution speeds.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def _matches(self, gold_chunk: str, retrieved_chunk: str) -> bool:
        """Determines if a retrieved chunk matches a gold reference chunk."""
        if not gold_chunk or not retrieved_chunk:
            return False
        
        g = gold_chunk.strip().lower()
        r = retrieved_chunk.strip().lower()

        if g == r:
            return True
        
        # Substring or URI fragment matching (e.g. 'doc.pdf#page=2' in 'http://.../doc.pdf#page=2')
        if g in r or r in g:
            return True

        return False

    def score_query(self, gold_chunks: Sequence[str], retrieved_chunks: Sequence[str]) -> RetrievalQueryEvaluation:
        """
        Scores a single query's retrieval against gold reference chunks.
        """
        gold_list = [g for g in gold_chunks if g]
        retrieved_list = list(retrieved_chunks)

        if not gold_list:
            # If no gold chunks required (e.g., pure refusal query with no relevant documents)
            return RetrievalQueryEvaluation(
                gold_chunks=gold_list,
                retrieved_chunks=retrieved_list,
                hit_at_1=1.0,
                hit_at_3=1.0,
                hit_at_5=1.0,
                mrr=1.0,
                context_recall=1.0
            )

        first_hit_rank: Optional[int] = None
        matched_gold_indices = set()

        for r_idx, ret in enumerate(retrieved_list[:self.top_k], start=1):
            for g_idx, gold in enumerate(gold_list):
                if self._matches(gold, ret):
                    if first_hit_rank is None:
                        first_hit_rank = r_idx
                    matched_gold_indices.add(g_idx)

        hit_1 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 1) else 0.0
        hit_3 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 3) else 0.0
        hit_5 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 5) else 0.0
        mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        recall = len(matched_gold_indices) / len(gold_list) if gold_list else 0.0

        return RetrievalQueryEvaluation(
            gold_chunks=gold_list,
            retrieved_chunks=retrieved_list,
            hit_at_1=hit_1,
            hit_at_3=hit_3,
            hit_at_5=hit_5,
            mrr=mrr,
            context_recall=recall
        )

    def aggregate(self, evaluations: Sequence[RetrievalQueryEvaluation]) -> RetrievalMetricsModel:
        """
        Aggregates per-query evaluations into dataset-level RetrievalMetricsModel.
        """
        if not evaluations:
            return RetrievalMetricsModel(
                hit_at_1=0.0,
                hit_at_3=0.0,
                hit_at_5=0.0,
                mrr=0.0,
                context_recall=0.0
            )

        n = len(evaluations)
        h1 = sum(e.hit_at_1 for e in evaluations) / n
        h3 = sum(e.hit_at_3 for e in evaluations) / n
        h5 = sum(e.hit_at_5 for e in evaluations) / n
        mrr = sum(e.mrr for e in evaluations) / n
        recall = sum(e.context_recall for e in evaluations) / n

        return RetrievalMetricsModel(
            hit_at_1=round(h1, 4),
            hit_at_3=round(h3, 4),
            hit_at_5=round(h5, 4),
            mrr=round(mrr, 4),
            context_recall=round(recall, 4)
        )

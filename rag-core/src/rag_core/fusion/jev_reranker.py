#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JevReranker: Fast System One Candidate Chunk Reranker & Filter.
Scores and reranks candidate chunks in a SINGLE parallel speculative Jev request,
achieving sub-350ms reranking without heavyweight local cross-encoder models.
"""

import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from rag_core.evaluation.jev_client import TypeSafeJevClient
from rag_core.naive.chunker import Chunk


@dataclass
class RerankedChunk:
    chunk: Chunk
    initial_score: float
    jev_relevance: float
    rank: int
    fallback_used: bool = False


class JevReranker:
    """
    Reranks candidate chunks retrieved from vector/hybrid search.
    Packages all chunk relevance judgments into ONE single parallel Jev request.
    """

    def __init__(
        self,
        jev_client: Optional[TypeSafeJevClient] = None,
        min_relevance_threshold: float = 0.40,
        max_chunk_text_len: int = 400
    ):
        self.client = jev_client or TypeSafeJevClient()
        self.min_relevance_threshold = min_relevance_threshold
        self.max_chunk_text_len = max_chunk_text_len

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_n: Optional[int] = None
    ) -> List[RerankedChunk]:
        """
        Rerank retrieved (chunk, score) candidates using parallel Jev Noul questions.
        If Jev fails or is unavailable, falls back to the original retrieval ranking.

        Args:
            query: The user query string.
            candidates: List of (Chunk, similarity_score) tuples from initial retriever.
            top_n: Optional limit on the number of top reranked chunks to return.

        Returns:
            List of RerankedChunk objects sorted by jev_relevance descending.
        """
        if not candidates:
            return []

        # Prepare state: query and candidate chunks preview
        chunk_previews = [
            {"index": i, "chunk_id": c.chunk_id, "text": c.text[:self.max_chunk_text_len]}
            for i, (c, _) in enumerate(candidates)
        ]
        state = {
            "user_query": query,
            "candidate_chunks": chunk_previews
        }

        # Build parallel questions: 1 Noul question per candidate chunk
        questions: Dict[str, Dict[str, Any]] = {}
        for i in range(len(candidates)):
            questions[f"chunk_{i}"] = {
                "type": "noul",
                "instructions": (
                    f"Does candidate chunk {i} contain factual evidence, definitions, "
                    f"or concepts that directly help answer `user_query`?"
                ),
                "criteria": {
                    "true": "The chunk contains specific facts, explanations, or numbers that help answer the question.",
                    "false": "The chunk is off-topic, boilerplate, or does not provide evidence to answer the question."
                }
            }

        t0 = time.perf_counter()
        try:
            # 1 Single parallel fan-out call for all candidate chunks
            res = self.client.system_one(state=state, questions=questions)
            answers = res.get("answers", {})

            reranked: List[RerankedChunk] = []
            for i, (chunk, init_score) in enumerate(candidates):
                ans = answers.get(f"chunk_{i}", {})
                relevance = float(ans.get("noul", 0.5))
                reranked.append(RerankedChunk(
                    chunk=chunk,
                    initial_score=init_score,
                    jev_relevance=relevance,
                    rank=0,
                    fallback_used=False
                ))

            # Sort descending by Jev calibrated relevance probability
            reranked.sort(key=lambda x: x.jev_relevance, reverse=True)
            for r_idx, item in enumerate(reranked, start=1):
                item.rank = r_idx

            # Apply relevance threshold and top_n truncation
            filtered = [c for c in reranked if c.jev_relevance >= self.min_relevance_threshold]
            if not filtered:
                # If all were below threshold, retain at least top 1 to avoid empty context
                filtered = reranked[:1]

            return filtered[:top_n] if top_n else filtered

        except Exception:
            # Fallback to initial similarity ranking
            reranked = [
                RerankedChunk(
                    chunk=chunk,
                    initial_score=init_score,
                    jev_relevance=init_score,
                    rank=i + 1,
                    fallback_used=True
                )
                for i, (chunk, init_score) in enumerate(candidates)
            ]
            return reranked[:top_n] if top_n else reranked

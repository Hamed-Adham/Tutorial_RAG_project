"""
Reciprocal Rank Fusion (RRF) for Blending Dense and Lexical Candidates.
Formula: RRF_score(d) = sum(1 / (k + rank_i(d))) across all retrieval lists.
"""

from typing import List, Dict
from rag_core.indexing.science_schema import RetrievalCandidate
from rag_core.contracts import IRRFFusion


class ReciprocalRankFusion(IRRFFusion):
    """Non-parametric rank fusion combining dense and lexical candidate lists."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        dense_results: List[RetrievalCandidate],
        bm25_results: List[RetrievalCandidate],
        top_k: int = 50,
        k_constant: int = 60
    ) -> List[RetrievalCandidate]:
        """Combine dense and BM25 candidate ranks using Reciprocal Rank Fusion."""
        k = k_constant or self.k
        candidate_map: Dict[str, RetrievalCandidate] = {}
        rrf_scores: Dict[str, float] = {}

        # 1. Process dense candidates
        for rank, cand in enumerate(dense_results, start=1):
            chunk_id = cand.chunk_id
            if chunk_id not in candidate_map:
                candidate_map[chunk_id] = cand.model_copy()
            candidate_map[chunk_id].dense_rank = rank
            candidate_map[chunk_id].dense_score = cand.dense_score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # 2. Process BM25 candidates
        for rank, cand in enumerate(bm25_results, start=1):
            chunk_id = cand.chunk_id
            if chunk_id not in candidate_map:
                candidate_map[chunk_id] = cand.model_copy()
            candidate_map[chunk_id].bm25_rank = rank
            candidate_map[chunk_id].bm25_score = cand.bm25_score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # 3. Assign rrf_score and sort in descending order
        fused_list: List[RetrievalCandidate] = []
        for chunk_id, cand in candidate_map.items():
            cand.rrf_score = rrf_scores[chunk_id]
            fused_list.append(cand)

        fused_list.sort(key=lambda c: c.rrf_score, reverse=True)
        return fused_list[:top_k]

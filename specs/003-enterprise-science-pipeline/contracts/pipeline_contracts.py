"""
Component Interface Contracts for Enterprise Science RAG Pipeline.
Defines typing and abstract protocols for all decoupled pipeline stages.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ScienceChunkContract(BaseModel):
    chunk_id: str
    section_id: str
    chapter: int
    page: int
    breadcrumbs: str
    text: str

class CandidateContract(BaseModel):
    chunk_id: str
    section_id: str
    chapter: int
    page: int
    text: str
    breadcrumbs: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0

class RerankedContract(BaseModel):
    candidate: CandidateContract
    score: float
    rank: int

class RewriteContract(BaseModel):
    needs_rewrite: bool
    confidence: float
    rewritten_query: Optional[str]
    sub_queries: List[str]

class IBM25Retriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 50) -> List[CandidateContract]:
        """Perform sub-5ms BM25 lexical search over tokenized science corpus."""
        pass

class IRRFFusion(ABC):
    @abstractmethod
    def fuse(
        self,
        dense_results: List[CandidateContract],
        bm25_results: List[CandidateContract],
        top_k: int = 50,
        k_constant: int = 60
    ) -> List[CandidateContract]:
        """Combine dense and BM25 candidate ranks using Reciprocal Rank Fusion."""
        pass

class ICrossEncoderReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[CandidateContract],
        top_k: int = 5,
        timeout_ms: float = 350.0
    ) -> List[RerankedContract]:
        """Re-rank candidate chunks using neural cross-attention with timeout fallback."""
        pass

class IQueryRewriter(ABC):
    @abstractmethod
    async def evaluate_and_rewrite(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> RewriteContract:
        """Confidence-gate rewrite necessity and decompose multi-concept queries."""
        pass

class IAdvancedRAGPipeline(ABC):
    @abstractmethod
    async def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Execute full end-to-end enterprise RAG pipeline within 2.0s p95 budget."""
        pass

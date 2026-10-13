"""
Core Abstract Interfaces & Contracts for the Enterprise Science RAG Pipeline.
Provides decoupled protocols for lexical retrieval, fusion, re-ranking, and orchestration.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from rag_core.indexing.science_schema import (
    RetrievalCandidate,
    RerankedCandidate,
    RewriteDecision,
    AdvancedRAGResult
)


class IBM25Retriever(ABC):
    """Protocol for sub-5ms BM25 lexical retrieval over Persian science corpus."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 50,
        chapters: Optional[List[int]] = None
    ) -> List[RetrievalCandidate]:
        """Perform lexical inverted index lookup returning top candidates with optional chapter filtering."""
        pass


class IRRFFusion(ABC):
    """Protocol for reciprocal rank fusion combining dense and lexical ranks."""

    @abstractmethod
    def fuse(
        self,
        dense_results: List[RetrievalCandidate],
        bm25_results: List[RetrievalCandidate],
        top_k: int = 50,
        k_constant: int = 60
    ) -> List[RetrievalCandidate]:
        """Combine dense and BM25 candidate ranks using non-parametric RRF formula."""
        pass


class ICrossEncoderReranker(ABC):
    """Protocol for neural cross-encoder re-ranking with latency circuit breaker."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 5,
        timeout_ms: float = 350.0
    ) -> List[RerankedCandidate]:
        """Re-rank candidate chunks using full cross-attention with timeout fallback."""
        pass


class IQueryRewriter(ABC):
    """Protocol for confidence-gated query rewriting and sub-query decomposition."""

    @abstractmethod
    async def evaluate_and_rewrite(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        selected_chapters: Optional[List[int]] = None
    ) -> RewriteDecision:
        """Score rewrite necessity and return self-contained query or sub-queries."""
        pass


class IAdvancedRAGPipeline(ABC):
    """Protocol for the complete latency-budgeted enterprise RAG pipeline."""

    @abstractmethod
    async def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        overfetch_k: int = 50,
        selected_chapters: Optional[List[int]] = None,
    ) -> List[RetrievalCandidate]:
        """Execute decoupled retrieval funnel: Gated Rewriting -> Hybrid Search -> RRF -> CrossEncoder."""
        pass

    @abstractmethod
    async def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        overfetch_k: int = 50,
        selected_chapters: Optional[List[int]] = None,
    ) -> AdvancedRAGResult:
        """Execute full pipeline within 2.0s p95 budget and record Langfuse traces."""
        pass

    @abstractmethod
    async def query_stream(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        overfetch_k: int = 50,
        selected_chapters: Optional[List[int]] = None,
    ):
        """Execute full pipeline yielding response tokens as an asynchronous generator with TTFT < 1.0s."""
        pass

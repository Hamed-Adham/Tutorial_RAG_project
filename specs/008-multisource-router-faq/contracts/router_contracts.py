"""
Contract interfaces for Level-0 Source Routing and Multi-Corpus BM25 Retrieval.
Feature: 008-multisource-router-faq
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

from rag_core.indexing.science_schema import RetrievalCandidate


class Level0RoutingDecision(BaseModel):
    raw_query: str
    routing_tier: str              # "science_only" | "faq_only" | "both_domains" | "out_of_scope"
    active_sources: List[str]      # ["science_textbook"], ["administrative_faq"], or both
    confidence: float
    probabilities: Dict[str, float]
    is_refusal: bool
    is_adversarial: bool = False
    refusal_reason: Optional[str] = None
    latencies: Dict[str, float] = {}


class ILevel0SourceRouter(ABC):
    """Sub-30ms System One Source Router evaluating cross-corpus intent."""

    @abstractmethod
    def route_query(
        self,
        query: str,
        enabled_sources: Optional[List[str]] = None
    ) -> Level0RoutingDecision:
        """Evaluate query and return active destination corpora."""
        pass


class IMultiCorpusBM25Retriever(ABC):
    """Dispatches lexical retrieval across separate per-corpus BM25 indexes."""

    @abstractmethod
    def search(
        self,
        query: str,
        source_domain: str,
        top_k: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalCandidate]:
        """Execute sub-5ms BM25 search against the specified corpus index."""
        pass

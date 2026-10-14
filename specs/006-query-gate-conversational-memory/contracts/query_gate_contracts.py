# -*- coding: utf-8 -*-
"""
Typed Contracts & Interfaces for Feature 006: QueryGate & Conversational Memory Routing.
Defines Pydantic models, Enums, and Protocol interfaces for:
1. QueryGate (Model 1)
2. SubQ Decision Model 2
3. Missing Summary Backfill Client
4. LangGraph State Node Signatures
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field


class CurriculumIntentTier(str, Enum):
    """3-tier curriculum intent classification for QueryGate."""
    COMPLETELY_IRRELEVANT = "completely_irrelevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    RELEVANT = "relevant"


class ContextNeedTier(str, Enum):
    """4-tier dispatching classification for QueryGate."""
    NEEDS_HISTORY_AND_RETRIEVE = "needs_history_and_retrieve"
    NEEDS_HISTORY_ONLY = "needs_history_only"
    NEEDS_RETRIEVE_ONLY = "needs_retrieve_only"
    NEEDS_NEITHER = "needs_neither"


class HistoryRelevanceTier(str, Enum):
    """4-tier rubric evaluated by SubQ Decision Model 2 for each historical turn."""
    NOT_NEEDED_NOT_RELEVANT = "not_needed_not_relevant"
    NOT_NEEDED_RELEVANT = "not_needed_relevant"
    NEEDED_NOT_RELEVANT = "needed_not_relevant"
    NEEDED_RELEVANT = "needed_relevant"


class SelectedHistoryItem(BaseModel):
    """An individual historical turn chosen as context for synthesis/retrieval."""
    turn_id: str
    turn_index: int
    is_recent_5: bool
    content_type: str = Field(description="'full_text' if is_recent_5, else 'summary'")
    question: str
    answer: str
    summary: Optional[str] = None
    relevance_tier: HistoryRelevanceTier
    confidence: float = 1.0


class QueryGateDecision(BaseModel):
    """Full decision model returned by QueryGate."""
    raw_query: str
    effective_query: str
    intent_tier: CurriculumIntentTier
    context_need_tier: ContextNeedTier
    is_refusal: bool = False
    bypass_retrieval: bool = False
    selected_history_items: List[SelectedHistoryItem] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)
    reasoning: str
    latencies: Dict[str, float] = Field(default_factory=dict)


class QASummaryRecord(BaseModel):
    """Persisted Q&A summary record."""
    turn_id: str
    session_id: str
    question_text: str
    answer_text: str
    summary_text: str
    created_at: float


@runtime_checkable
class IQueryGate(Protocol):
    """Protocol for primary query gating and dispatching."""

    async def evaluate_and_route(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        selected_chapters: Optional[List[int]] = None,
        selected_context_turn_id: Optional[str] = None,
        history_enabled: bool = True
    ) -> QueryGateDecision:
        ...


@runtime_checkable
class ISubQDecisionClient(Protocol):
    """Protocol for SubQ Decisions API (System One)."""

    async def evaluate_history_turns(
        self,
        query: str,
        candidate_turns: List[Dict[str, Any]]
    ) -> Dict[str, HistoryRelevanceTier]:
        ...

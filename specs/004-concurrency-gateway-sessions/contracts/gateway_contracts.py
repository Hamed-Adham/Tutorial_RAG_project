"""
Core Abstract Interfaces & Typed Models for Concurrency Gateway & Session Management.
Feature: 004-concurrency-gateway-sessions
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime
from pydantic import BaseModel, Field

from rag_core.indexing.science_schema import PageCitation, AdvancedRAGResult


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------

class DialogueTurnModel(BaseModel):
    turn_id: str
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: datetime
    citations: List[PageCitation] = Field(default_factory=list)
    latency_profile: Dict[str, float] = Field(default_factory=dict)


class SessionStateModel(BaseModel):
    session_id: str
    created_at: datetime
    last_active_at: datetime
    turns: List[DialogueTurnModel] = Field(default_factory=list)
    selected_chapters: Optional[List[int]] = None
    ttl_seconds: int = 7200


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language scientific inquiry")
    session_id: Optional[str] = Field(None, description="Unique session ID for multi-turn history continuity")
    selected_chapters: Optional[List[int]] = Field(None, description="Mode 1 scoped practice chapters [1..15]")
    top_k: int = Field(default=5, ge=1, le=10, description="Final number of retrieved chunks to synthesize")
    stream: bool = Field(default=False, description="Whether to stream response tokens progressively")


class ChatResponse(BaseModel):
    session_id: str
    query: str
    effective_query: str
    answer: str
    citations: List[PageCitation] = Field(default_factory=list)
    is_refusal: bool = False
    latency_profile: Dict[str, float] = Field(default_factory=dict)
    trace_id: str


class OperationalMetricsModel(BaseModel):
    active_requests: int = Field(..., ge=0, le=50)
    available_semaphore_slots: int = Field(..., ge=0, le=50)
    avg_queue_wait_ms: float
    total_requests: int
    circuit_breaker_trips: int
    circuit_breaker_trip_rate: float
    cache_hits: int
    cache_hit_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    est_cost_per_1k_usd: float


class HealthStatusModel(BaseModel):
    status: str
    qdrant_connected: bool
    bm25_loaded: bool
    redis_connected: bool
    cross_encoder_warm: bool
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Abstract Protocol Interfaces
# ---------------------------------------------------------------------------

class ISessionStore(ABC):
    """Protocol for conversation session storage (Redis with In-Memory fallback)."""

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionStateModel]:
        """Fetch session state by ID or return None if expired / absent."""
        pass

    @abstractmethod
    async def save_session(self, session: SessionStateModel) -> None:
        """Persist or update session state and refresh expiration TTL."""
        pass

    @abstractmethod
    async def append_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        citations: List[PageCitation],
        latencies: Dict[str, float],
        selected_chapters: Optional[List[int]] = None
    ) -> SessionStateModel:
        """Append a dialogue turn, enforce sliding window (last 4 turns), and reset TTL."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Explicitly purge session state."""
        pass


class IConcurrencyGateway(ABC):
    """Protocol for the async gateway handling rate limits, semaphores, and worker pools."""

    @abstractmethod
    async def execute_chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a full RAG inquiry with queue delay tracking and semaphore protection."""
        pass

    @abstractmethod
    async def execute_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Execute streaming inquiry yielding SSE frames with TTFT < 1.0s."""
        pass


class IOperationalMetricsTracker(ABC):
    """Protocol for live production KPI and telemetry aggregation."""

    @abstractmethod
    def record_request_start(self) -> float:
        """Mark start timestamp and increment active requests counter."""
        pass

    @abstractmethod
    def record_request_finish(
        self,
        start_time: float,
        queue_wait_ms: float,
        total_latency_ms: float,
        circuit_breaker_tripped: bool,
        cache_hit: bool,
        prompt_tokens: int,
        completion_tokens: int
    ) -> None:
        """Record completed query telemetry and update running latency percentiles."""
        pass

    @abstractmethod
    def get_metrics_snapshot(self) -> OperationalMetricsModel:
        """Return point-in-time metrics snapshot for /api/v1/metrics."""
        pass

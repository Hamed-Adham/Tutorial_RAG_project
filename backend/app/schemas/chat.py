"""
Pydantic schemas for Chat REST and SSE streaming endpoints.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from rag_core.indexing.science_schema import PageCitation


class DialogueTurn(BaseModel):
    """A single user inquiry or assistant response in a session."""
    turn_id: str = Field(..., description="Unique turn UUID")
    turn_pair_id: Optional[str] = Field(default=None, description="Common unique UUID linking query and answer in the same turn")
    role: str = Field(..., pattern="^(user|assistant)$", description="Role of the speaker")
    content: str = Field(..., description="Message text")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: List[PageCitation] = Field(default_factory=list)
    latency_profile: Dict[str, float] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Persistent or in-memory conversation session state."""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    turns: List[DialogueTurn] = Field(default_factory=list, description="Capped at last 4 dialogue turns")
    selected_chapters: Optional[List[int]] = Field(default=None, description="Mode 1 scoped chapters")
    ttl_seconds: float = 7200.0


class ChatRequest(BaseModel):
    """Inbound request payload for chat endpoints."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language scientific inquiry")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn history continuity")
    selected_chapters: Optional[List[int]] = Field(None, description="Mode 1 scoped practice chapters [1..15]")
    top_k: int = Field(default=5, ge=1, le=10, description="Final number of retrieved chunks to synthesize")
    stream: bool = Field(default=False, description="Whether to stream response tokens progressively")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Client-side prior dialogue turns")
    selected_context_turn_id: Optional[str] = Field(default=None, description="Explicit turn ID selected by user when history is disabled")
    history_enabled: bool = Field(default=True, description="Whether conversational memory is enabled")


class ChatResponse(BaseModel):
    """Outbound batch response from /api/v1/chat."""
    session_id: str
    turn_id: Optional[str] = Field(default=None, description="Unique turn pair ID for this Q&A exchange")
    query: str
    effective_query: str
    query_rewritten: bool = Field(default=False, description="Whether query was rewritten/expanded from conversational history")
    answer: str
    citations: List[PageCitation] = Field(default_factory=list)
    is_refusal: bool = False
    conversational_context_badge: Optional[str] = None
    intent_tier: Optional[str] = None
    context_need_tier: Optional[str] = None
    latency_profile: Dict[str, float] = Field(default_factory=dict)
    trace_id: str = Field(default="", description="Langfuse root trace ID")


class StreamEventData(BaseModel):
    """Payload inside a Server-Sent Event frame."""
    content: Optional[str] = None
    seq: Optional[int] = None
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    effective_query: Optional[str] = None
    query_rewritten: Optional[bool] = None
    citations: Optional[List[Dict[str, Any]]] = None
    is_refusal: Optional[bool] = None
    conversational_context_badge: Optional[str] = None
    intent_tier: Optional[str] = None
    context_need_tier: Optional[str] = None
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    trace_id: Optional[str] = None
    detail: Optional[str] = None


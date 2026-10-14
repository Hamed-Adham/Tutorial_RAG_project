"""
Session and Dialogue Turn domain models matching ISessionStore contract.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from rag_core.indexing.science_schema import PageCitation


class DialogueTurnModel(BaseModel):
    """Model representing an individual dialogue turn."""
    turn_id: str
    turn_pair_id: Optional[str] = Field(default=None, description="Common unique UUID linking query and answer in the same turn")
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: List[PageCitation] = Field(default_factory=list)
    latency_profile: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def q_and_a_summary(self) -> Optional[str]:
        return self.metadata.get("q_and_a_summary")



class SessionStateModel(BaseModel):
    """Container for multi-turn conversational session history."""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    turns: List[DialogueTurnModel] = Field(default_factory=list)
    selected_chapters: Optional[List[int]] = None
    ttl_seconds: float = 7200.0


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool

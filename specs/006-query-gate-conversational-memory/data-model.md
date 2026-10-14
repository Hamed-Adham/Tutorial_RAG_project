# Data Model & Schema Specifications: Feature 006

**Feature**: QueryGate & Conversational Memory Routing  
**Directory**: `specs/006-query-gate-conversational-memory`  
**Date**: 2026-10-14  

---

## 1. Domain Enums

### 1.1 `CurriculumIntentTier`
Represents the 3-tier classification of curriculum relevance evaluated by `QueryGate`:
```python
from enum import Enum

class CurriculumIntentTier(str, Enum):
    COMPLETELY_IRRELEVANT = "completely_irrelevant"  # Prompt injection, chitchat, pop culture, non-science
    PARTIALLY_RELEVANT = "partially_relevant"        # Mixed query containing out-of-scope text + science inquiry
    RELEVANT = "relevant"                            # Direct in-scope science inquiry
```

### 1.2 `ContextNeedTier`
Represents the 4-tier dispatching decision for retrieval and conversational history:
```python
class ContextNeedTier(str, Enum):
    NEEDS_HISTORY_AND_RETRIEVE = "needs_history_and_retrieve"  # Option 1: Requires history + textbook retrieval
    NEEDS_HISTORY_ONLY = "needs_history_only"                  # Option 2: Meta-inquiry; requires history, 0 retrieval
    NEEDS_RETRIEVE_ONLY = "needs_retrieve_only"                # Option 3: Fresh question; requires retrieval, 0 history
    NEEDS_NEITHER = "needs_neither"                            # Option 4: Refusal or chit-chat; 0 history, 0 retrieval
```

### 1.3 `HistoryRelevanceTier`
Represents the 4-tier rubric evaluated by SubQ Decision Model 2 for each candidate historical Q&A turn:
```python
class HistoryRelevanceTier(str, Enum):
    NOT_NEEDED_NOT_RELEVANT = "not_needed_not_relevant"
    NOT_NEEDED_RELEVANT = "not_needed_relevant"
    NEEDED_NOT_RELEVANT = "needed_not_relevant"
    NEEDED_RELEVANT = "needed_relevant"
```

---

## 2. Core Entities & Pydantic Contracts

### 2.1 `SelectedHistoryItem`
Represents an individual historical turn chosen by Decision Model 2 (or manually selected in UI):
```python
from pydantic import BaseModel, Field
from typing import Optional

class SelectedHistoryItem(BaseModel):
    turn_id: str = Field(description="Unique identifier for the session dialogue turn")
    turn_index: int = Field(description="0-indexed position within candidate history window (0 = oldest, 19 = newest)")
    is_recent_5: bool = Field(description="True if turn is among the 5 most recent turns of the session")
    content_type: str = Field(description="'full_text' if is_recent_5, else 'summary'")
    question: str = Field(description="User prompt text of this historical turn")
    answer: str = Field(description="Generated assistant response text of this historical turn")
    summary: Optional[str] = Field(default=None, description="Persisted summary paragraph (<= 512 tokens)")
    relevance_tier: HistoryRelevanceTier = Field(description="Classification output from SubQ Decision Model 2")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Decision confidence from SubQ")
```

### 2.2 `QueryGateDecision`
Canonical outcome model returned by `QueryGate.evaluate_and_route()`:
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class QueryGateDecision(BaseModel):
    raw_query: str = Field(description="Original user query input")
    effective_query: str = Field(description="Purified or transformed query string forwarded to downstream stages")
    intent_tier: CurriculumIntentTier = Field(description="3-tier curriculum intent classification")
    context_need_tier: ContextNeedTier = Field(description="4-tier dispatching classification")
    is_refusal: bool = Field(default=False, description="True if query is completely irrelevant or needs neither")
    bypass_retrieval: bool = Field(default=False, description="True if needs_history_only or refusal")
    selected_history_items: List[SelectedHistoryItem] = Field(default_factory=list, description="Historical turns selected by Decision Model 2")
    sub_queries: List[str] = Field(default_factory=list, description="Decomposed sub-questions for compound inquiries")
    reasoning: str = Field(description="Traceable diagnostic explanation for routing decision")
    latencies: Dict[str, float] = Field(default_factory=dict, description="Execution breakdown in ms")
```

### 2.3 `QASummaryRecord`
Structured entity for cached turn summaries stored in session storage:
```python
class QASummaryRecord(BaseModel):
    turn_id: str
    session_id: str
    question_text: str
    answer_text: str
    summary_text: str = Field(max_length=2000, description="Persian summary paragraph, <= 512 tokens")
    created_at: float = Field(default_factory=time.time)
```

---

## 3. LangGraph Orchestrator State Extensions

The `RAGGraphState` `TypedDict` in `rag-core/src/rag_core/graph/state.py` is augmented with the following attributes:

| Field | Type | Description |
|---|---|---|
| `query_gate_decision` | `Optional[Dict[str, Any]]` | Full serialized `QueryGateDecision` model dump |
| `intent_tier` | `Optional[str]` | Serialized `CurriculumIntentTier` value |
| `context_need_tier` | `Optional[str]` | Serialized `ContextNeedTier` value |
| `selected_history_items` | `List[Dict[str, Any]]` | List of selected history items (`full_text` or `summary`) |
| `selected_context_turn_id` | `Optional[str]` | Manual context turn ID selected by user when history is OFF |
| `q_and_a_summary` | `Optional[str]` | Generated summary of the current turn produced during synthesis |
| `conversational_context_badge` | `Optional[str]` | Label (e.g. "پاسخ بر اساس گفتگوی پیشین") displayed when retrieval is bypassed |

---

## 4. REST API Schema Updates (`backend/app/schemas/chat.py`)

### 4.1 `ChatRequest` Extensions
```python
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    selected_chapters: Optional[List[int]] = None
    history: Optional[List[Dict[str, str]]] = None
    history_enabled: bool = Field(default=True, description="Toggle for automatic conversation memory")
    selected_context_turn_id: Optional[str] = Field(
        default=None,
        description="Manual context turn ID attached via Qwen-style UI quote when history is OFF"
    )
```

### 4.2 `ChatResponse` Extensions
```python
class ChatResponse(BaseModel):
    response: str
    session_id: str
    citations: List[PageCitation]
    retrieved_chunks: List[Dict[str, Any]]
    effective_query: str
    timing: Dict[str, float]
    # New fields:
    conversational_context_badge: Optional[str] = Field(
        default=None,
        description="Populated with 'پاسخ بر اساس گفتگوی پیشین' when retrieval is bypassed (needs_history_only)"
    )
    q_and_a_summary: Optional[str] = Field(
        default=None,
        description="Summary of this completed turn generated for session caching"
    )
    query_gate_decision: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnostic decision breakdown from QueryGate and Decision Model 2"
    )
```

---

## 5. SubQ Request & Response JSON Contracts

### 5.1 SubQ Batch Request Payload
```json
{
  "model": "subq-snap-preview",
  "state": {
    "current_query": "چرا این آزمایش خطا دارد؟",
    "instructions": "Determine if each candidate historical turn is needed and relevant to answer current_query."
  },
  "questions": {
    "turn_0": {
      "type": "choice",
      "instructions": "Is turn 0 needed and relevant to understand or answer current_query?",
      "criteria": {
        "not_needed_not_relevant": "Turn discusses unrelated topic and is not required",
        "not_needed_relevant": "Turn discusses related science topic but is not directly required",
        "needed_not_relevant": "Turn contains essential context or constraints despite different topic",
        "needed_relevant": "Turn directly defines the antecedent concept needed to answer current_query"
      }
    }
  }
}
```

### 5.2 SubQ Response Payload
```json
{
  "model": "subq-snap-preview",
  "answers": {
    "turn_0": {
      "type": "choice",
      "choice": "needed_relevant",
      "probabilities": {
        "not_needed_not_relevant": 0.02,
        "not_needed_relevant": 0.05,
        "needed_not_relevant": 0.03,
        "needed_relevant": 0.90
      },
      "confidence": 0.95
    }
  }
}
```

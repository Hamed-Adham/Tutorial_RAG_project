# Data Model: LangGraph Orchestrator & Next.js Chatbot UI

**Feature Branch**: `005-langgraph-orchestrator-frontend`  
**Specification**: [spec.md](spec.md)  
**Date**: 2026-10-14  

---

## 1. Frontend Entities (TypeScript Models)

### `Session`
Represents an ongoing conversational thread in the user interface.
```typescript
export interface Session {
  id: string                     // Unique session identifier, e.g. "sess_m1a2b3c4"
  title: string                  // First user prompt truncated to 36 chars, default "گفتگوی جدید"
  createdAt: string              // ISO 8601 creation timestamp
  updatedAt: string              // ISO 8601 last interaction timestamp
  selectedChapters?: number[]    // Active chapter constraint [1..15] or undefined for global search
}
```

### `Message`
Represents a single conversational bubble within a session.
```typescript
export interface Message {
  id: string                     // Unique message ID, e.g. "msg_user_123"
  role: "user" | "assistant"     // Sender persona
  content: string                // Markdown text body
  timestamp: string              // ISO 8601 timestamp
  citations?: Citation[]         // Verified textbook citations (assistant only)
  latency_profile?: LatencyProfile // Stage-by-stage execution timings (ms)
  is_refusal?: boolean           // True if inquiry was rejected as out-of-scope
  effective_query?: string       // Query after pronoun antecedent resolution
  trace_id?: string              // Langfuse trace correlation ID
}
```

### `Citation`
Pedagogical provenance link binding an answer claim to official textbook content.
```typescript
export interface Citation {
  doc_id: string                 // Document slice identifier, e.g. "C706_olom_haftom-7-144-2-5.pdf"
  filename: string               // Display name, e.g. "کتاب علوم تجربی هفتم"
  page: number                   // Physical textbook page number (1-indexed)
  chunk_id?: string              // Unique chunk hash
  score?: number                 // Neural cross-encoder relevance score (0.0 to 1.0+)
  metadata?: {
    title?: string               // Curriculum breadcrumb path, e.g. "فصل ۲ > چگالی و اندازه‌گیری"
  }
}
```

### `LatencyProfile`
Waterfall telemetry object capturing millisecond execution durations across all RAG stages.
```typescript
export interface LatencyProfile {
  queue_wait_ms?: number         // Milliseconds spent queued behind concurrency semaphore
  rewrite_ms?: number            // Query understanding and antecedent resolution latency
  retrieval_ms?: number          // Parallel Qdrant + BM25 search latency
  rerank_ms?: number             // Cross-encoder neural reranking latency
  expansion_ms?: number          // Parent section expansion latency
  synthesis_ms?: number          // Generation model completion latency
  total_ms?: number              // Total end-to-end turnaround latency
}
```

---

## 2. Orchestration Graph State (`rag_core.graph.state.RAGGraphState`)

Strongly-typed Python TypedDict passed along all nodes and conditional edges in the LangGraph StateGraph.

```python
class RAGGraphState(TypedDict, total=False):
    # User Inputs & Context
    user_query: str                          # Original user query string
    effective_query: str                     # Disambiguated / resolved query
    session_id: Optional[str]                # Session UUID
    history: Optional[List[Dict[str, str]]]  # 4-turn sliding dialogue turns
    selected_chapters: Optional[List[int]]   # Active chapter filter [1..15]
    
    # Pre-Retrieval & Confidence Gating
    proceed_to_retrieval: bool               # Flag: whether query enters search stages
    is_refusal: bool                         # Flag: whether query was refused early
    sub_queries: List[str]                   # Decomposed queries for multi-hop
    rewrite_decision: Optional[Dict[str, Any]]

    # Search & Candidate Sets
    dense_candidates: List[Any]              # Raw Qdrant BGE-M3 matches
    bm25_candidates: List[Any]               # Raw Hazm BM25 matches
    fused_candidates: List[Any]              # Blended candidates post-RRF
    retrieved_chunks: List[Any]              # Top retrieved candidates
    reranked_chunks: List[Any]               # Top candidates post cross-encoder
    expanded_sections: List[Any]             # Parent sections (max 3, 2500 tokens)
    context_text: str                        # Assembled prompt context
    citations: List[Any]                     # PageCitation dataclass objects

    # Generation & Metrics
    agent_response: str                      # Final generated Persian answer
    latencies: Dict[str, float]              # Per-stage execution timings (ms)
    tokens: Dict[str, int]                   # Prompt and completion token counts
    trace_id: Optional[str]                  # Langfuse trace identifier
```

---

## 3. Entity State Transitions

```text
User Submits Question (Frontend)
       │
       ▼
[LearningSession] Updated with new User Message
       │
       ▼
[RAGGraphState] Initialized with query, history, and selected_chapters
       │
       ├─► [gated_rewrite_node] ──(Out-of-Scope)──► [refusal_node] ──► Final State (Refusal)
       │
       └─► [hybrid_retrieve_node] ──► dense & bm25 candidates fused via RRF
              │
              ▼
           [neural_rerank_node] ──► cross-encoder scores assigned
              │
              ▼
           [parent_expansion_node] ──► child chunks expanded to parent sections & citations built
              │
              ▼
           [synthesis_node] ──► progressive token generation with citations
              │
              ▼
[Assistant Message] Appended to Session in LocalStorage & Redis with Citations & LatencyProfile
```

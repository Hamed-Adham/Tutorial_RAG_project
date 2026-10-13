# Data Models & Schemas: Enterprise Science RAG Pipeline

**Feature**: `003-enterprise-science-pipeline`  
**Date**: 2026-10-10  
**Specification**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/spec.md)  

---

## Entity Relationship Overview

```text
┌─────────────────────────┐
│     ParentSection       │  (15 Chapters, ~300 sections)
│  section_id: str        │  Preserves full tables, procedures, and formulas
│  full_text: str         │
│  page_range: [start,end]│
└────────────┬────────────┘
             │ 1
             │
             │ *
┌────────────▼────────────┐
│      ScienceChunk       │  (~1,200 child chunks, ~512 tokens)
│  chunk_id: str          │  Indexed in Qdrant (Dense) & BM25 (Sparse)
│  breadcrumbs: str       │
│  page: int              │
└────────────┬────────────┘
             │ Matched in search
             ▼
┌─────────────────────────┐
│   RetrievalCandidate    │  Blended via Reciprocal Rank Fusion
│  dense_rank / bm25_rank │  rrf_score: float
└────────────┬────────────┘
             │ Re-ranked (Top 50 -> Top 5)
             ▼
┌─────────────────────────┐
│    RerankedCandidate    │  Cross-Encoder BGE-Reranker v2 M3
│  cross_encoder_score    │  Expanded back to parent ParentSection
└─────────────────────────┘
```

---

## 1. Domain Entities & Storage Schemas

### `ScienceChunk`
Atomic retrieval unit (~512 tokens) indexed across both Dense vector store and Lexical inverted index.
```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ScienceChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. olum7_ch03_c012")
    doc_id: str = Field(..., description="Source document identifier, e.g. olum_7_chapter_03")
    section_id: str = Field(..., description="Enclosing parent section identifier, e.g. sec_03_02")
    chapter: int = Field(..., ge=1, le=15, description="Textbook chapter number (1 to 15)")
    page: int = Field(..., ge=1, description="Textbook page number for strict citation verification")
    breadcrumbs: str = Field(..., description="Hierarchical heading path: 'فصل ۳ > اتم‌ها الفبای مواد > ذره‌های سازنده مواد'")
    text: str = Field(..., description="Atomic text content with intact tables or diagram descriptions")
    token_count: int = Field(..., ge=1, le=1024, description="Token size of chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags (is_experiment, is_diagram)")
```

### `ParentSection`
Cohesive curriculum section encapsulating adjacent child chunks for grounded synthesis.
```python
from typing import List

class ParentSection(BaseModel):
    section_id: str = Field(..., description="Canonical section identifier")
    chapter: int = Field(..., ge=1, le=15)
    title: str = Field(..., description="Section title")
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    full_text: str = Field(..., description="Unbroken text of entire section")
    child_chunk_ids: List[str] = Field(..., description="List of constituent child chunk IDs")
    token_count: int = Field(..., ge=1, description="Total tokens across the full section")
```

---

## 2. Pre-Retrieval & Query Intelligence Models

### `RewriteDecision`
Output of the TypeSafe Jev confidence gate.
```python
class RewriteDecision(BaseModel):
    needs_rewrite: bool = Field(..., description="True if query is ambiguous, conversational, or multi-concept")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    rewritten_query: Optional[str] = Field(None, description="Self-contained query if rewritten; None if bypassed")
    sub_queries: List[str] = Field(default_factory=list, description="List of decomposed sub-queries for multi-hop synthesis")
    reasoning: str = Field("", description="Justification for decision")
```

---

## 3. Retrieval & Ranking Models

### `RetrievalCandidate`
Unified candidate object produced by dense and lexical retrievers.
```python
class RetrievalCandidate(BaseModel):
    chunk_id: str
    section_id: str
    chapter: int
    page: int
    text: str
    breadcrumbs: str
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0
```

### `RerankedCandidate`
Candidate enriched with cross-encoder neural relevance.
```python
class RerankedCandidate(BaseModel):
    candidate: RetrievalCandidate
    cross_encoder_score: float = Field(..., description="Raw logit or normalized probability from bge-reranker-v2-m3")
    final_rank: int = Field(..., ge=1, description="Final rank position (1 to 5)")
```

---

## 4. Context Assembly & Grounded Response Models

### `ExpandedContext`
Context payload delivered to the LLM generator after parent expansion and lost-in-the-middle reordering.
```python
class ExpandedContext(BaseModel):
    parent_sections: List[ParentSection] = Field(..., max_items=3, description="Capped at maximum 3 unique parent sections")
    total_tokens: int = Field(..., le=2500, description="Strictly capped at 2,500 prompt tokens")
    source_chunk_ids: List[str] = Field(..., description="Child chunks that triggered this context")
    reordered_texts: List[str] = Field(..., description="Formatted context blocks ordered via LongContextReorder")
```

### `AdvancedRAGResult`
Complete end-to-end execution result matching enterprise observability contracts.
```python
from rag_core.postprocessing.citations import PageCitation

class AdvancedRAGResult(BaseModel):
    query: str
    effective_query: str
    answer: str
    citations: List[PageCitation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalCandidate] = Field(default_factory=list)
    top_parents: List[str] = Field(default_factory=list, description="Section IDs used in synthesis")
    latency_profile: Dict[str, float] = Field(..., description="Per-stage wall clock timings in milliseconds")
    trace_id: str = Field(..., description="Langfuse root trace ID")
    is_refusal: bool = Field(default=False)
```

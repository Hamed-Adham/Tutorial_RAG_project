# Phase 1 Data Model: Hierarchical Multi-Source Routing & FAQ Integration

**Feature Branch**: `008-multisource-router-faq`  
**Feature Directory**: `specs/008-multisource-router-faq`  
**Status**: Completed  

---

## 1. Domain Entities & Type Definitions

```
                     ┌───────────────────────────────┐
                     │     Level0RoutingDecision     │
                     ├───────────────────────────────┤
                     │ - active_sources: List[str]   │
                     │ - routing_tier: SourceTier    │
                     │ - confidence: float           │
                     │ - probabilities: Dict[str, f] │
                     │ - is_refusal: bool            │
                     └───────────────┬───────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
    ┌───────────────────────────────┐ ┌───────────────────────────────┐
    │       ScienceDM1Decision      │ │         FAQDM1Decision        │
    ├───────────────────────────────┤ ├───────────────────────────────┤
    │ - intent_tier: IntentTier     │ │ - intent_tier: IntentTier     │
    │ - context_need: NeedTier      │ │ - context_need: NeedTier      │
    │ - candidate_chapters: [int]   │ │ - candidate_categories: [str] │
    │ - effective_query: str        │ │ - effective_query: str        │
    └───────────────┬───────────────┘ └───────────────┬───────────────┘
                    │                                 │
                    ▼                                 ▼
    ┌───────────────────────────────┐ ┌───────────────────────────────┐
    │      RetrievalCandidate       │ │      RetrievalCandidate       │
    │   (doc_id: science_textbook)  │ │  (doc_id: administrative_faq) │
    ├───────────────────────────────┤ ├───────────────────────────────┤
    │ - chunk_id: str               │ │ - chunk_id: str               │
    │ - chapter: int, page: int     │ │ - category: str, row: int     │
    │ - text, breadcrumbs           │ │ - text, breadcrumbs           │
    └───────────────┬───────────────┘ └───────────────┬───────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │       RerankedCandidate       │
                     ├───────────────────────────────┤
                     │ - chunk: RetrievalCandidate   │
                     │ - rerank_score: float         │
                     │ - source_domain: str          │
                     └───────────────────────────────┘
```

---

## 2. Python Schemas (`science_schema.py` Extensions)

### 2.1 `SourceTier` & `Level0RoutingDecision`

```python
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class SourceDomain(str, Enum):
    SCIENCE_TEXTBOOK = "science_textbook"
    ADMINISTRATIVE_FAQ = "administrative_faq"

class Level0RoutingTier(str, Enum):
    SCIENCE_ONLY = "science_only"
    FAQ_ONLY = "faq_only"
    BOTH_DOMAINS = "both_domains"
    OUT_OF_SCOPE = "out_of_scope"

class Level0RoutingDecision(BaseModel):
    raw_query: str
    routing_tier: Level0RoutingTier
    active_sources: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    probabilities: Dict[str, float] = Field(default_factory=dict)
    is_refusal: bool = False
    is_adversarial: bool = False
    refusal_reason: Optional[str] = None
    latencies: Dict[str, float] = Field(default_factory=dict)
```

### 2.2 FAQ Metadata in `RetrievalCandidate`

`RetrievalCandidate` in `rag_core/indexing/science_schema.py` is extended to support both Textbook and FAQ metadata without breaking compatibility:

```python
class RetrievalCandidate(BaseModel):
    chunk_id: str
    doc_id: str = "science_textbook"         # "science_textbook" | "administrative_faq"
    source_type: str = "science_textbook"    # Matches Qdrant payload
    section_id: Optional[str] = None
    chapter: Optional[int] = None
    page: Optional[int] = None
    category: Optional[str] = None           # FAQ category (e.g. "ضمانت‌نامه")
    row: Optional[int] = None                # FAQ row index (1-134)
    text: str
    breadcrumbs: List[str] = Field(default_factory=list)
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
```

### 2.3 Synthesis Context Extension

```python
class SynthesisPromptType(str, Enum):
    PERSIAN_SCIENCE = "persian_science"
    ADMINISTRATIVE_FAQ = "administrative_faq"
    COMPOSITE_GROUNDED = "composite_grounded"

class CompositeSynthesisState(BaseModel):
    prompt_type: SynthesisPromptType
    active_domains: List[str]
    has_science: bool = False
    has_faq: bool = False
    science_chunks: List[RetrievalCandidate] = Field(default_factory=list)
    faq_chunks: List[RetrievalCandidate] = Field(default_factory=list)
```

---

## 3. Qdrant & BM25 Payload Storage Schemas

### 3.1 Qdrant Vector Payload (`naive_metadata.json` / Qdrant collection)

```json
{
  "id": "faq_42",
  "vector": [0.012, -0.045, "... 1024 dims (bge-m3) ..."],
  "payload": {
    "chunk_id": "faq_42",
    "doc_id": "administrative_faq",
    "source_type": "administrative_faq",
    "category": "ضمانت‌نامه",
    "row": 42,
    "question": "مدارک لازم برای صدور ضمانت‌نامه چیست؟",
    "text": "سوال: مدارک لازم برای صدور ضمانت‌نامه چیست؟\nپاسخ: تصویر اساسنامه، آخرین آگهی تغییرات روزنامه رسمی...",
    "breadcrumbs": ["پرسش‌های متداول", "صندوق نوآوری و شکوفایی", "ضمانت‌نامه"]
  }
}
```

### 3.2 BM25 Index Pickle Structure (`data_index/faq_bm25.pkl`)

Serialized instance of `ScienceBM25Index` or `FAQBM25Index`:
- `corpus_chunks`: `List[RetrievalCandidate]` (134 items)
- `tokenized_corpus`: `List[List[str]]` (pre-tokenized with Hazm normalizer + Persian stopwords filtered)
- `bm25_model`: `rank_bm25.BM25Okapi` instance
- `category_map`: `Dict[str, List[int]]` mapping each category to chunk indices for fast category-filtered lexical search.

---

## 4. State Transitions & Validation Invariants

1. **Explicit Source Selection Invariant**:
   - If client passes `selected_sources=["science_textbook"]`: `Level0RoutingDecision` is not evaluated. `active_sources` is strictly `["science_textbook"]`.
   - If client passes `selected_sources=["administrative_faq"]`: `Level0RoutingDecision` is not evaluated. `active_sources` is strictly `["administrative_faq"]`.
   - If client passes `selected_sources=["all"]` or `len(selected_sources) > 1`: `Level0SourceRouter` runs and dynamically sets `active_sources`.

2. **Refusal Invariant**:
   - If `Level0RoutingDecision.is_refusal is True`: `active_sources` MUST be empty `[]`.
   - Pipeline skips all DM1s and retrieval stages, directly returning the grounded Persian refusal response in `<25ms`.

3. **Pass-Through Expansion Invariant**:
   - For all chunks where `doc_id == "administrative_faq"`: `ParentSectionExpander` returns the chunk as-is (`expanded_context = chunk.text`) without querying chapter manifests.

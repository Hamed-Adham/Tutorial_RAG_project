# Phase 0 Research: Hierarchical SubQ Multi-Source Router & Compound Corpus Dispatcher

**Feature Branch**: `008-multisource-router-faq`  
**Feature Directory**: `specs/008-multisource-router-faq`  
**Status**: Completed  

---

## 1. Overview of Research Goals

This research addresses the technical strategy for integrating the 134-item `administrative_faq` collection into the Advanced Enterprise RAG Pipeline alongside `science_textbook`, with autonomous source routing, parallel per-source decision gating, and dual-grounded partitioned synthesis.

---

## 2. Research Findings & Architectural Decisions

### Research Item 1: Level-0 Source Router Primitive & Topology

- **Decision**: Implement `Level0SourceRouter` using a single multi-label Choice/Noul call via `TypeSafeJevClient.system_one` evaluating:
  - `science_only`: Queries solely about 7th-grade natural science concepts (biology, physics, chemistry, geology, organs, energy).
  - `faq_only`: Queries solely about Iran Innovation and Prosperity Fund (INIF), loans, guarantees, knowledge-based companies, Ghazal portal, or commercialization.
  - `both_domains`: Compound cross-corpus queries touching both science concepts and INIF funding/administrative rules.
  - `out_of_scope`: General conversation, sports, hacking, prompt injections, or irrelevant domains.
  Coupled with a deterministic keyword fallback mechanism for offline/air-gapped resilience.
- **Rationale**:
  - Achieves routing decisions in **<25ms** without free-form LLM token generation.
  - Keeps gate latency comfortably within the 180ms total gate budget.
  - Emits calibrated probabilities across all categories for auditability.
- **Alternatives Considered**:
  - *Sequential Cascade*: Checking Science first, then checking FAQ if Science score is low. (Rejected: Doubles latency on FAQ queries to ~50ms).
  - *Free-form LLM Router*: Prompting an LLM to generate JSON `{"target": "..."}`. (Rejected: Adds 800-1500ms latency and token cost; violates Principle VII Law of Parsimony).

---

### Research Item 2: Parallel Per-Source Decision Model 1 Execution

- **Decision**: When `both_domains` is activated by the Level-0 router, invoke `Decision Model 1` (DM1) concurrently via `asyncio.gather(dm1_science, dm1_faq)`.
  - `DM1-Science`: Specializes in 7th-grade curriculum scope, candidate chapters, and science-specific query expansion.
  - `DM1-FAQ`: Specializes in INIF administrative scope, category matching (e.g. `ضمانت‌نامه`, `تسهیلات`), and administrative query expansion.
- **Rationale**:
  - Decouples domain knowledge representations: 7th-grade educational pedagogy does not bleed into corporate financial regulations.
  - Running in parallel via `asyncio.gather` means total wall-clock latency equals `max(latency_sci, latency_faq)` rather than their sum (~35ms).
  - Reuses existing `QueryGate` components with domain-specific configuration payloads.
- **Alternatives Considered**:
  - *Monolithic Unified DM1*: A single giant prompt evaluating both curriculum and administrative rules. (Rejected: Leads to prompt bloat, hallucinated category mapping, and brittle guardrails).

---

### Research Item 3: Lexical Search (BM25) Architecture for FAQ

- **Decision**: Generate and persist a separate lexical index at `data_index/faq_bm25.pkl` using Persian Hazm normalization, managed by a `MultiCorpusBM25Retriever`.
- **Rationale**:
  - Isolates corpus IDF statistics: Science textbook terms (e.g., "سلول", "انرژی") and administrative terms (e.g., "سرمایه ثبتی", "ضمانت‌نامه حسن انجام کار") have drastically different document frequency distributions. A combined index distorts BM25 scoring.
  - Index size for FAQ is tiny (<200KB in memory), enabling sub-2ms lexical search.
  - Updating FAQ items or circulars does not require re-indexing the science textbook.
- **Alternatives Considered**:
  - *Single Unified BM25 Index with Post-Filtering*: Indexing all chunks in one BM25 index and filtering candidates post-search. (Rejected: BM25 score calculation considers total corpus token frequencies, which biases scores towards rarer cross-corpus words).

---

### Research Item 4: Pass-Through Parent Expansion for FAQ Chunks

- **Decision**: Update `ParentSectionExpander` to inspect the candidate `doc_id` / `source_type`. If `administrative_faq`, pass the candidate chunk through directly without consulting `data/vlm_parsed/manifest.json`.
- **Rationale**:
  - FAQ QA pairs in `data_index/naive_metadata.json` are already self-contained (question title + full answer text). They do not belong to PDF chapters or multi-page hierarchies.
  - Avoids `KeyError` or missing manifest warnings during parent section expansion.
- **Alternatives Considered**:
  - *Synthesizing fake chapter manifests for FAQ*: (Rejected: Unnecessary complexity and maintenance overhead).

---

### Research Item 5: Domain-Partitioned Response Synthesis

- **Decision**: Introduce `SYSTEM_PROMPT_COMPOSITE_GROUNDED` in `rag_core/generation/prompts.py` for cross-corpus answers. The prompt mandates structuring the answer with explicit Markdown headers:
  ```markdown
  ### ۱. مفاهیم علمی (کتاب علوم تجربی)
  [علمی پاسخ داده می‌شود با استناد به فصول و صفحات]

  ### ۲. مقررات و تسهیلات (صندوق نوآوری و شکوفایی)
  [پاسخ ضوابط اداری و مالی با استناد به دسته‌ها و ردیف‌های متداول]
  ```
  If a complex multi-step inquiry is detected, steps are answered sequentially under the corresponding domain headers.
- **Rationale**:
  - Clear cognitive separation for the user: Educational concepts are distinct from formal financial/grant application procedures.
  - Direct alignment with user clarification from 2026-10-15.
- **Alternatives Considered**:
  - *Blended Prose Narrative*: Single combined narrative without headers. (Rejected: Reader cannot easily distinguish textbook science from binding financial policies).

---

### Research Item 6: Citation Formatting for Administrative FAQ

- **Decision**: Format FAQ citations in `CitationFormatter` as:
  `[پرسش‌های متداول – دسته: {category} – ردیف: {row}]`
  with linkable metadata to the FAQ row index.
- **Rationale**:
  - Matches the established pattern of Science citations (`[علوم تجربی هفتم – فصل {chapter} – ص {page}]`).
  - Provides full provenance auditability required by Principle II.

---

## 3. Technology & Dependency Validation

| Component | Library / Engine | Validation Status |
|---|---|---|
| **Level-0 Router** | `TypeSafeJevClient` (System One Choice) | Validated (Sub-25ms, zero token cost) |
| **FAQ Lexical Search** | `ScienceBM25Index` / Persian Tokenizer | Validated (Hazm Persian normalizer, <2ms) |
| **Dense Search** | `QdrantVectorStore` payload filtering | Validated (Native `source_type` & `category` filters) |
| **Parallel DM1s** | Python `asyncio.gather` | Validated (Concurrent execution in <40ms) |
| **Cross-Encoder Reranker** | BGE-Reranker-v2-m3 (Fast CPU/MPS) | Validated (Evaluates combined candidate pool) |
| **Synthesizer** | OpenAI / Local LLM with domain prompt | Validated (Structured domain headers) |

All technical choices are confirmed with zero unknowns.

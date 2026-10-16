# Implementation Plan: Hierarchical SubQ Multi-Source Router & Compound Corpus Dispatcher

**Branch**: `008-multisource-router-faq` | **Date**: 2026-10-15 | **Spec**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/008-multisource-router-faq/spec.md)

**Input**: Feature specification from `specs/008-multisource-router-faq/spec.md`

---

## Summary

This feature brings the 134-item `administrative_faq` collection into the Advanced Enterprise Pipeline alongside `science_textbook`, enabling seamless multi-corpus intelligent RAG:
1. **Level-0 SubQ Decision Source Router**: A single multi-label TypeSafe Jev System One Choice model call (<25ms) that classifies incoming queries into `science_only`, `faq_only`, `both_domains`, or `out_of_scope`, with a deterministic keyword fallback. Bypassed automatically when a single corpus is explicitly selected.
2. **Parallel Per-Source Decision Model 1 (DM1)**: When multiple sources are active, DM1 runs concurrently via `asyncio.gather(dm1_science, dm1_faq)` with domain-specific states and query expansions.
3. **Decoupled Lexical BM25 Search**: A dedicated `data_index/faq_bm25.pkl` index managed by a `MultiCorpusBM25Retriever`.
4. **Pass-Through Parent Expansion**: Chunks belonging to `administrative_faq` bypass chapter manifest lookups and pass directly to reranking.
5. **Partitioned Dual-Grounded Synthesis**: Cross-corpus queries generate a single structured answer with explicit domain headers (`### ۱. مفاهیم علمی (کتاب علوم)` and `### ۲. مقررات و تسهیلات (صندوق نوآوری)`) and step-by-step guidance for complex inquiries.

---

## Technical Context

**Language/Version**: Python 3.11+, TypeScript 5.4+ (Next.js 14 App Router)  
**Primary Dependencies**:
- `rag-core`: `pydantic>=2.0`, `rank-bm25`, `hazm`, `qdrant-client`, `structlog`, `langchain-core`, `langgraph`
- `backend`: `fastapi`, `uvicorn`, `redis`, `sse-starlette`
- `frontend`: `next`, `react`, `lucide-react`, `tailwindcss`  
**Storage**:
- Vector Store: Qdrant (`science_curriculum_bge_m3` & naive metadata collections with payload filtering)
- Lexical Indexes: In-memory `science_bm25.pkl` and `faq_bm25.pkl`
- Session Cache: Redis / In-Memory Session Store  
**Testing**: `pytest`, `pytest-asyncio`  
**Target Platform**: macOS / Linux Server (CPU + Apple Silicon / CUDA accelerator support)  
**Project Type**: Fullstack Decoupled Enterprise RAG Gateway  
**Performance Goals**:
- Level-0 Source Router: `<25ms`
- Parallel DM1 Evaluation: `<35ms` total wall-clock
- FAQ BM25 Retrieval: `<5ms`
- Total Gate Latency: `<180ms`  
**Constraints**:
- Absolute prohibition on raw LaTeX formatting in output (Global Rule)
- Zero drift in Codebase Memory Knowledge Graph
- Zero regression on existing 184 test suite  
**Scale/Scope**: 15 Textbook Chapters (400+ chunks) + 134 Administrative FAQ QA items across 9 categories.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Principle I: Three-Layer Decoupled Architecture**: `rag-core` contains pure Python routing, indexing, retrieval, and synthesis logic. `backend` wraps it into REST/SSE endpoints. `frontend` provides the multi-source selector UI.
- [x] **Principle II: Evaluation-Driven Quality**: Verification against golden cross-corpus queries; full regression testing against existing 184 tests.
- [x] **Principle III: Observability**: Spans for `source_routing`, `dm1_parallel_eval`, `multi_corpus_retrieval`, and `composite_synthesis` attached to Langfuse root traces with JSON structlog.
- [x] **Principle IV: Agent Orchestration**: LangGraph StateGraph handles multi-source routing and parallel execution flow cleanly.
- [x] **Principle V: Defense-in-Depth Security**: Qdrant payload filters enforce `source_type` and `category` boundaries at the database level.
- [x] **Principle VI: Caching & Cost Efficiency**: Sub-25ms TypeSafe System One Choice replaces expensive token-generation routers.
- [x] **Principle VII: Law of Parsimony & Component Maximization**: Reuses existing `TypeSafeJevClient`, `ScienceBM25Index`, `QdrantVectorStore`, and `ResponseSynthesizer` without introducing unnecessary new frameworks.
- [x] **Principle VIII: Continuous Development Logging**: Milestone updates recorded in `development_log.md`.
- [x] **Principle IX: Environment Hygiene**: Isolated within `.venv`, environment variables managed in `.env`.
- [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance**: `detect_changes` verified before planning; post-implementation index refresh scheduled.
- [x] **Principle XI: Software Engineering & Change Discipline**: Human approval required before commits.
- [x] **Principle XII: Enterprise Rigor & Tool Trinity**: Spec-Kit, Git, and Langfuse rigorously maintained.

---

## Project Structure

### Documentation (this feature)

```text
specs/008-multisource-router-faq/
├── spec.md              # Feature specification with clarifications
├── plan.md              # This implementation plan
├── research.md          # Phase 0 research document
├── data-model.md        # Phase 1 data models & schemas
├── quickstart.md        # Phase 1 quickstart validation guide
└── contracts/           # Phase 1 interface & API contracts
    ├── router_contracts.py
    └── api_contracts.md
```

### Source Code Impact Matrix

```text
rag-core/
├── src/rag_core/
│   ├── indexing/
│   │   ├── science_schema.py            # Add SourceDomain, Level0RoutingDecision, FAQ metadata in RetrievalCandidate
│   │   ├── build_faq_bm25.py            # Builder script for data_index/faq_bm25.pkl
│   │   └── bm25_index.py                # Generic support for category filtering in BM25
│   ├── query/
│   │   ├── jev_router.py                # Upgrade JevRouter with multi-domain choice & Level0SourceRouter
│   │   └── query_gate.py                # Domain-aware DM1 configuration (Science vs FAQ)
│   ├── retrieval/
│   │   ├── bm25.py                      # MultiCorpusBM25Retriever supporting both science & faq indexes
│   │   └── hybrid_retriever.py          # Multi-source parallel dispatch & payload filtering
│   ├── postprocessing/
│   │   ├── section_expansion.py         # Pass-through mode for administrative_faq chunks
│   │   └── citations.py                 # FAQ citation formatter ([پرسش‌های متداول – دسته: X – ردیف: Y])
│   ├── generation/
│   │   ├── prompts.py                   # Add SYSTEM_PROMPT_COMPOSITE_GROUNDED & SYSTEM_PROMPT_ADMIN_FAQ_GROUNDED
│   │   └── synthesizer.py               # Support domain prompt switching & partitioned headers
│   └── graph/
│       ├── pipeline_graph.py            # Multi-source branch in EnterpriseScienceGraph
│       └── nodes.py                     # Parallel DM1 & multi-retrieval node wrappers
└── tests/
    └── test_multisource_router.py       # Comprehensive unit & integration tests for Feature 008

backend/
├── app/
│   ├── schemas/
│   │   └── chat.py                      # selected_sources & selected_faq_categories in ChatRequest
│   └── services/
│       └── rag_service.py               # Route multi-source requests to EnterpriseScienceGraph

frontend/
└── src/
    ├── components/
    │   ├── ChatHeader.tsx               # Source selector dropdown / toggle (Science, FAQ, All)
    │   └── ChatMessage.tsx              # Render FAQ citations alongside chapter badges
    └── types/
        └── index.ts                     # Citation & ChatRequest type updates
```

---

## Complexity Tracking

*No violations detected. Zero unnecessary dependencies added.*

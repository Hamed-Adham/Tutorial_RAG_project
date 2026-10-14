# Tasks: LangGraph Orchestrator & Modern Next.js Chatbot UI

**Feature Branch**: `005-langgraph-orchestrator-frontend`  
**Specification**: [spec.md](spec.md)  
**Plan**: [plan.md](plan.md)  
**Data Model**: [data-model.md](data-model.md)  
**Contracts**: [contracts/api_contracts.md](contracts/api_contracts.md), [contracts/graph_contracts.py](contracts/graph_contracts.py)  

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Frontend tooling, styling tokens, typography assets, and backend proxy configuration.

- [X] T001 Configure Next.js API rewrites proxying `/api/v1/:path*` to `http://127.0.0.1:8000/api/v1/:path*` in `frontend/next.config.js`
- [X] T002 [P] Download and host official Peyda WOFF2 font files (`Peyda-Regular.woff2`, `Peyda-Medium.woff2`, `Peyda-Bold.woff2`) in `frontend/public/fonts/`
- [X] T003 [P] Declare `@font-face` rules for Peyda and establish Persian typography styling tokens (`line-height: 1.7`) in `frontend/src/app/globals.css`
- [X] T004 [P] Configure Tailwind CSS with `Peyda` font family aliases (`font-sans`, `font-peyda`) in `frontend/tailwind.config.js`
- [X] T005 [P] Define curriculum metadata, chapter titles, and suggestion prompts in `frontend/src/lib/constants.ts`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Strongly-typed state definitions, interfaces, and protocol converters shared across the fullstack layers.

**⚠️ CRITICAL**: Must be verified before proceeding to user story integration.

- [X] T006 [P] Define universal TypeScript data interfaces (`Session`, `Message`, `Citation`, `LatencyProfile`, `OperationalMetrics`, `SystemHealth`) in `frontend/src/types/index.ts`
- [X] T007 [P] Define `RAGGraphState` TypedDict for LangGraph workflows tracking queries, active chapters, candidate sets, citations, and latencies in `rag-core/src/rag_core/graph/state.py`
- [X] T008 [P] Implement `RetrievalCandidate.to_langchain_document()` converting chunk candidates into standard `langchain_core.documents.Document` instances in `rag-core/src/rag_core/indexing/science_schema.py`
- [X] T009 [P] Implement `HybridRetriever.retrieve_documents()` conforming to LangChain `BaseRetriever` protocol in `rag-core/src/rag_core/retrieval/hybrid.py`
- [X] T010 Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)` to record foundational state

**Checkpoint**: Shared data models and LangChain Document protocols operational.

---

## Phase 3: User Story 1 - Interactive Conversational Science Learning Interface with Persian RTL (Priority: P1) 🎯 MVP

**Goal**: Deliver a modern, responsive conversational UI featuring native Persian RTL typography, real-time SSE streaming with TTFT < 1.0s, and collapsible session history.

**Independent Test**: Load `http://localhost:3000`, submit a curriculum inquiry, observe progressive token streaming in Peyda font, and manage sessions in the sidebar.

### Implementation for User Story 1
- [X] T011 [P] [US1] Implement root layout with Persian RTL direction (`dir="rtl"`) and viewport metadata in `frontend/src/app/layout.tsx`
- [X] T012 [P] [US1] Implement collapsible `Sidebar` component with session search, New Chat button, session renaming, and deletion in `frontend/src/components/Sidebar.tsx`
- [X] T013 [P] [US1] Implement auto-resizing `ChatInput` textarea with Enter-to-submit, Shift+Enter newlines, and curriculum quick suggestion pills in `frontend/src/components/ChatInput.tsx`
- [X] T014 [US1] Implement `ChatHeader` component with active session title, model indicators, and modal triggers in `frontend/src/components/ChatHeader.tsx`
- [X] T015 [US1] Implement Server-Sent Events (SSE) streaming consumer using `ReadableStream` reader, `TextDecoder`, and `AbortController` cancellation in `frontend/src/app/page.tsx`
- [X] T016 [US1] Implement dual-layer session persistence syncing conversation threads to `LocalStorage` (`unified_rag_sessions`, `unified_rag_messages`) and backend session store in `frontend/src/app/page.tsx`

**Checkpoint**: User Story 1 functional and independently testable (MVP reached).

---

## Phase 4: User Story 2 - Scoped Curriculum Practice vs. Global Open Exploration (Priority: P2)

**Goal**: Enable students to restrict queries to specific textbook chapters (Mode 1 Scoped Practice) or search across all 15 chapters (Mode 2 Global Search).

**Independent Test**: Open Chapter Scope modal, select Chapter 2, submit query, and verify that only Chapter 2 candidates are searched.

### Implementation for User Story 2
- [X] T017 [P] [US2] Implement `ChapterModal` component displaying all 15 chapters of the 7th-grade science textbook with selection checkboxes and select-all/clear actions in `frontend/src/components/ChapterModal.tsx`
- [X] T018 [US2] Wire active chapter scope state into chat request payloads (`selected_chapters: [1, 2]` or `null`) in `frontend/src/app/page.tsx`
- [X] T019 [US2] Bind `OLUM_7_CHAPTER_TOPICS` into Jev prompt instructions for curriculum scope verification in `rag-core/src/rag_core/query/gated_rewriter.py`
- [X] T020 [US2] Wire native Qdrant `FieldCondition` pre-filtering and BM25 candidate index pre-filtering based on active chapters in `rag-core/src/rag_core/retrieval/hybrid.py`

**Checkpoint**: Chapter-scoped practice mode fully operational across UI and retrieval engines.

---

## Phase 5: User Story 3 - Evidence Verification, Citations Drawer & Latency Transparency (Priority: P3)

**Goal**: Display official textbook page citations, an expandable evidence inspection drawer, a latency waterfall pill, and a live Prometheus telemetry dashboard.

**Independent Test**: Generate an answer, inspect citation badges, expand the evidence drawer, review the latency waterfall, and open the operational telemetry modal.

### Implementation for User Story 3
- [X] T021 [P] [US3] Implement Persian citation badges (`[کتاب علوم تجربی هفتم (ص ۴)]`) and expandable evidence drawer in `frontend/src/components/ChatMessage.tsx`
- [X] T022 [P] [US3] Implement expandable latency waterfall badge displaying millisecond timings across queue, rewrite, retrieval, rerank, expansion, and synthesis in `frontend/src/components/ChatMessage.tsx`
- [X] T023 [P] [US3] Implement `MetricsModal` component polling `GET /api/v1/metrics` every 3 seconds to display active requests, circuit breaker status, and latency percentiles in `frontend/src/components/MetricsModal.tsx`
- [X] T024 [US3] Implement background health probe polling `GET /api/v1/health` with online/offline badge indicators in `frontend/src/components/ChatHeader.tsx`

**Checkpoint**: Pedagogical citations, evidence inspection, and live telemetry operational.

---

## Phase 6: User Story 4 - Declarative Graph Orchestration with Grounded Refusal & Observability (Priority: P4)

**Goal**: Deliver a compiled LangGraph `StateGraph` (`EnterpriseScienceGraph`) enforcing deterministic refusal branching, Langfuse callback tracing, and service layer integration.

**Independent Test**: Run `pytest rag-core/tests/test_enterprise_graph.py` verifying compilation, conditional refusal routing, and LangChain Document conversion.

### Tests for User Story 4
- [X] T025 [P] [US4] Write unit and integration tests for `EnterpriseScienceGraph` compilation, node transitions, refusal edge, and document conversions in `rag-core/tests/test_enterprise_graph.py`

### Implementation for User Story 4
- [X] T026 [P] [US4] Implement modular node factories (`create_enterprise_gated_rewrite_node`, `create_enterprise_refusal_node`, `create_enterprise_hybrid_retrieve_node`, `create_enterprise_neural_rerank_node`, `create_enterprise_expansion_node`, `create_enterprise_synthesis_node`) in `rag-core/src/rag_core/graph/nodes.py`
- [X] T027 [US4] Implement `EnterpriseScienceGraph` with conditional refusal edge (`should_enterprise_refuse`), `.ainvoke()`, and `.invoke()` in `rag-core/src/rag_core/graph/pipeline_graph.py`
- [X] T028 [US4] Implement `AdvancedRAGPipeline.as_langgraph_pipeline()` factory bridge method in `rag-core/src/rag_core/pipeline.py`
- [X] T029 [US4] Integrate native Langfuse `CallbackHandler` attachment inside `EnterpriseScienceGraph.ainvoke()` in `rag-core/src/rag_core/graph/pipeline_graph.py`
- [X] T030 [US4] Add `use_langgraph_engine` configuration option and execution route in `backend/app/services/rag_service.py`

**Checkpoint**: Declarative LangGraph orchestrator fully wired and verified.

---

## Phase 7: Polish & Governance

**Purpose**: Build verification, full test suite validation, quickstart validation, and constitution compliance.

- [X] T031 [P] Validate Next.js production build (`next build`) compiling cleanly with zero warnings or errors
- [X] T032 Execute full automated regression test suite (`pytest backend/tests/` and `pytest rag-core/tests/`) asserting 115+ tests pass
- [X] T033 Execute manual quickstart validation scenarios defined in `specs/005-langgraph-orchestrator-frontend/quickstart.md`
- [X] T034 Refresh Codebase Knowledge Graph via codebase-memory-reindex (`index_repository(persistence=True)`)
- [X] T035 Verify Zero Drift (Principle X Gate: `detect_changes` reports `changed_count == 0`)

---

## Dependencies & Execution Order

```text
Phase 1: Setup (T001..T005)
       │
       ▼
Phase 2: Foundational (T006..T010)
       │
       ▼
Phase 3: User Story 1 - Conversational UI & Streaming (T011..T016) 🎯 MVP
       │
       ▼
Phase 4: User Story 2 - Scoped Curriculum Practice (T017..T020)
       │
       ▼
Phase 5: User Story 3 - Evidence & Telemetry Transparency (T021..T024)
       │
       ▼
Phase 6: User Story 4 - LangGraph Orchestrator & Observability (T025..T030)
       │
       ▼
Phase 7: Polish & Governance (T031..T035)
       │
       ▼
Phase 8: Convergence (T036..T040)
```

---

## Parallel Execution Opportunities

- **Phase 1**: Font download (T002), CSS tokens (T003), and constants (T005) can run in parallel.
- **Phase 2**: TypeScript interfaces (T006), Python graph state (T007), and Document protocols (T008, T009) can run in parallel.
- **Phase 3**: Layout (T011), Sidebar (T012), and ChatInput (T013) can run in parallel.
- **Phase 4**: ChapterModal UI (T017) and backend topic prompt binding (T019) can run in parallel.
- **Phase 5**: Citation badges (T021), Latency badge (T022), and MetricsModal (T023) can run in parallel.
- **Phase 6**: Graph tests (T025) and Node factories (T026) can run in parallel.

---

## Implementation Strategy (MVP First)

1. **Step 1 (MVP)**: Deliver **Phase 1 (Setup)**, **Phase 2 (Foundational)**, and **Phase 3 (User Story 1)**. This provides a fully functional conversational UI with streaming tokens and Persian RTL formatting.
2. **Step 2**: Deliver **Phase 4 (User Story 2)** for 15-chapter scoped practice.
3. **Step 3**: Deliver **Phase 5 (User Story 3)** for citation verification and telemetry transparency.
4. **Step 4**: Deliver **Phase 6 (User Story 4)** for compiled LangGraph StateGraph execution and Langfuse tracing.
5. **Step 5**: Execute **Phase 7 (Polish & Governance)** for production build verification and Knowledge Graph synchronization.

---

## Phase 8: Convergence

**Purpose**: Reconcile remaining architectural, integration, and operational gaps identified during the convergence audit.

- [X] T036 Re-index repository into Codebase Knowledge Graph via codebase-memory-mcp and assert zero drift per Constitution X Gate (partial) [CRITICAL]
- [X] T037 Add `use_langgraph_engine` configuration option in `GatewaySettings` and route `RAGService.execute_chat` through `EnterpriseScienceGraph.ainvoke()` per FR-010 and plan (missing) [HIGH]
- [X] T038 Implement token streaming in `EnterpriseScienceGraph` and wire `RAGService.execute_chat_stream` to stream StateGraph tokens per FR-002 and plan (partial) [HIGH]
- [X] T039 Add test cases for `use_langgraph_engine` batch and SSE streaming routes in `backend/tests/test_chat_endpoints.py` per Constitution II (missing) [HIGH]
- [X] T040 Validate end-to-end quickstart scenarios in `specs/005-langgraph-orchestrator-frontend/quickstart.md` per SC-001 through SC-006 (missing) [MEDIUM]

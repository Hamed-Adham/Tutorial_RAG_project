# Implementation Tasks: Hierarchical SubQ Multi-Source Router & Compound Corpus Dispatcher

**Feature Branch**: `008-multisource-router-faq`  
**Feature Directory**: `specs/008-multisource-router-faq`  
**Specification**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/008-multisource-router-faq/spec.md)  
**Implementation Plan**: [plan.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/008-multisource-router-faq/plan.md)  

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Build lexical assets, index generators, and multi-domain prompts needed across all stories.

- [x] T001 Build lexical FAQ dataset extraction script in `rag-core/src/rag_core/indexing/build_faq_bm25.py`
- [x] T002 Generate and persist `data_index/faq_bm25.pkl` using Hazm Persian tokenizer and `ScienceBM25Index`
- [x] T003 [P] Add composite and administrative prompt constants (`SYSTEM_PROMPT_ADMIN_FAQ_GROUNDED`, `SYSTEM_PROMPT_COMPOSITE_GROUNDED`) in `rag-core/src/rag_core/generation/prompts.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data models, schema extensions, and base retrievers that MUST be complete before ANY user story can proceed.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T004 Add `SourceDomain`, `Level0RoutingTier`, `Level0RoutingDecision`, and FAQ metadata (`category`, `row`) in `rag-core/src/rag_core/indexing/science_schema.py`
- [x] T005 [P] Implement `MultiCorpusBM25Retriever` supporting dynamic routing to `science_bm25.pkl` and `faq_bm25.pkl` in `rag-core/src/rag_core/retrieval/bm25.py`
- [x] T006 [P] Implement pass-through mode for `administrative_faq` chunks in `rag-core/src/rag_core/postprocessing/section_expansion.py`
- [x] T007 [P] Implement FAQ citation formatting (`[پرسش‌های متداول – دسته: {category} – ردیف: {row}]`) in `rag-core/src/rag_core/postprocessing/citations.py`
- [x] T008 [P] Refresh Codebase Knowledge Graph via `codebase-memory-mcp index_repository(persistence=True)`

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Level-0 SubQ Source Routing for Multi-Source Queries (Priority: P1) 🎯 MVP

**Goal**: Deliver autonomous, sub-25ms Level-0 Source Routing that evaluates multi-source queries via TypeSafe Jev System One Choice, bypasses when a single corpus is explicitly chosen, and directs queries to the appropriate knowledge domain.

**Independent Test**: Submit an administrative FAQ query (`شرایط دریافت ضمانت‌نامه شرکت‌های دانش‌بنیان چیست؟`) with multi-source mode active; verify that `administrative_faq` is assigned with confidence >= 0.85 and routed to the FAQ pipeline without manual toggling.

### Tests for User Story 1
- [x] T009 [P] [US1] Create unit tests for Level-0 Source Router in `rag-core/tests/test_multisource_router.py`

### Implementation for User Story 1
- [x] T010 [US1] Implement `Level0SourceRouter` with single multi-label TypeSafe Choice model call and deterministic keyword fallback in `rag-core/src/rag_core/query/jev_router.py`
- [x] T011 [US1] Implement explicit single-source bypass logic in `rag-core/src/rag_core/graph/pipeline_graph.py`
- [x] T012 [US1] Integrate `Level0SourceRouter` into LangGraph router node in `rag-core/src/rag_core/graph/nodes.py`
- [x] T013 [US1] Verify User Story 1 independent test: single-source routing for science, FAQ, and out-of-scope refusal in `rag-core/tests/test_multisource_router.py`

**Checkpoint**: At this point, User Story 1 is fully functional and testable independently.

---

## Phase 4: User Story 2 - Compound Cross-Corpus Query & Multi-Source Dispatch (Priority: P2)

**Goal**: Support compound and multi-disciplinary queries spanning both Science and FAQ by executing parallel DM1s (`asyncio.gather`), parallel hybrid retrievals across both collections, reranking combined candidates, and synthesizing a single final response with partitioned domain headers.

**Independent Test**: Submit a compound query containing both a science concept and an administrative loan question; verify that both sources are activated, parallel DM1 instances run, and the final response contains partitioned headings (`### ۱. مفاهیم علمی (کتاب علوم)` and `### ۲. مقررات و تسهیلات (صندوق نوآوری)`).

### Tests for User Story 2
- [x] T014 [P] [US2] Create integration tests for parallel DM1 and cross-corpus retrieval in `rag-core/tests/test_multisource_router.py`

### Implementation for User Story 2
- [x] T015 [US2] Implement domain-customized Decision Model 1 evaluator for FAQ in `rag-core/src/rag_core/query/query_gate.py`
- [x] T016 [US2] Implement parallel per-source DM1 execution via `asyncio.gather` in `rag-core/src/rag_core/graph/nodes.py`
- [x] T017 [US2] Implement parallel hybrid retrieval across Science and FAQ collections in `rag-core/src/rag_core/graph/nodes.py`
- [x] T018 [US2] Implement partitioned dual-grounded response generation in `rag-core/src/rag_core/generation/synthesizer.py`
- [x] T019 [US2] Verify User Story 2 independent test: compound science + FAQ inquiry producing partitioned response in `rag-core/tests/test_multisource_router.py`

**Checkpoint**: At this point, User Stories 1 AND 2 work independently and in combination.

---

## Phase 5: User Story 3 - Full Administrative FAQ Advanced Pipeline Integration (Priority: P3)

**Goal**: Bring the administrative FAQ collection up to full parity in the advanced enterprise pipeline with dedicated FAQ lexical search (`data_index/faq_bm25.pkl`), dense Qdrant search with category filtering, pass-through expansion, and INIF consultant synthesis.

**Independent Test**: Execute an FAQ query with category filter (`ضمانت‌نامه`); verify that BM25 retrieves from `faq_bm25.pkl`, dense search filters by category in Qdrant, and response uses the INIF advisory persona.

### Tests for User Story 3
- [x] T020 [P] [US3] Create tests for category pre-filtering and FAQ advanced retrieval in `rag-core/tests/test_multisource_router.py`

### Implementation for User Story 3
- [x] T021 [US3] Implement category-level payload filtering for FAQ dense retrieval in `rag-core/src/rag_core/indexing/vector_store.py`
- [x] T022 [US3] Implement category filtering in `MultiCorpusBM25Retriever.search` in `rag-core/src/rag_core/retrieval/bm25.py`
- [x] T023 [US3] Update `EnterpriseScienceGraph.ainvoke` and streaming methods to support FAQ-only execution with `SYSTEM_PROMPT_ADMIN_FAQ_GROUNDED` in `rag-core/src/rag_core/graph/pipeline_graph.py`
- [x] T024 [US3] Verify User Story 3 independent test: FAQ retrieval with category filter and consultant persona in `rag-core/tests/test_multisource_router.py`

**Checkpoint**: User Story 3 complete. Administrative FAQ operates with full advanced pipeline capabilities.

---

## Phase 6: User Story 4 - Manual Corpus Override & UI Multi-Corpus Selection (Priority: P4)

**Goal**: Provide users with explicit source controls in the frontend (`ChatHeader.tsx` / `ChatInput.tsx`), pass `selected_sources` and `selected_faq_categories` through backend FastAPI endpoints, and render FAQ citations in chat messages.

**Independent Test**: In the frontend header, switch between "کتاب علوم", "پرسش‌های متداول صندوق", and "همه منابع (ترکیبی)", verify API payloads and citation badge display.

### Implementation for User Story 4
- [x] T025 [P] [US4] Update `ChatRequest` schema with `selected_sources` and `selected_faq_categories` in `backend/app/schemas/chat.py`
- [x] T026 [US4] Wire `selected_sources` and `selected_faq_categories` into `RAGService.execute_chat_stream` and `execute_chat` in `backend/app/services/rag_service.py`
- [x] T027 [P] [US4] Update TypeScript contracts with `selected_sources` and FAQ category citations in `frontend/src/types/index.ts`
- [x] T028 [US4] Add multi-source selector toggle/dropdown in `frontend/src/components/ChatHeader.tsx`
- [x] T029 [US4] Update `frontend/src/components/ChatMessage.tsx` to render FAQ category badges and citations alongside science chapter badges
- [x] T030 [US4] Add API endpoint tests for multi-source chat requests in `backend/tests/test_chat_endpoints.py`

**Checkpoint**: End-to-end fullstack multi-source intelligence operational from UI to retrieval.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: System validation, regression verification, documentation, and Knowledge Graph synchronization.

- [x] T031 [P] Run full end-to-end scenarios from `specs/008-multisource-router-faq/quickstart.md`
- [x] T032 Run full regression test suite (`pytest rag-core/tests backend/tests`) to ensure 100% pass rate
- [x] T033 Update `development_log.md` with Milestone 29 documentation
- [x] T034 Refresh Codebase Knowledge Graph (`.codebase-memory/graph.db.zst`) via `index_repository(persistence=True)`
- [x] T035 Verify Zero Drift (Principle X Gate: `detect_changes` reports `changed_count == 0` / status ready)

---

## Dependencies & Execution Order

### Phase Dependencies
- **Setup (Phase 1)**: Can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (Phase 1). BLOCKS all user stories.
- **User Story 1 (Phase 3 - P1)**: Depends on Foundational completion. MVP milestone.
- **User Story 2 (Phase 4 - P2)**: Depends on US1 completion (builds on Level-0 router).
- **User Story 3 (Phase 5 - P3)**: Depends on Foundational completion (can run in parallel with US1/US2).
- **User Story 4 (Phase 6 - P4)**: Depends on US1, US2, and US3 backend contracts being ready.
- **Polish (Phase 7)**: Depends on all user stories completion.

### Parallel Opportunities
- In Phase 1: `T003` can run in parallel with `T001`/`T002`.
- In Phase 2: `T005`, `T006`, and `T007` are in different files and can be implemented in parallel.
- Test tasks `T009`, `T014`, `T020` can be created in parallel with models.
- In Phase 6: Frontend types `T027` and backend schemas `T025` can proceed in parallel.

---

## Parallel Example: Foundational Phase

```bash
# Launch Foundational components in parallel:
Task: "Implement MultiCorpusBM25Retriever in rag-core/src/rag_core/retrieval/bm25.py"
Task: "Implement pass-through mode in rag-core/src/rag_core/postprocessing/section_expansion.py"
Task: "Implement FAQ citation formatting in rag-core/src/rag_core/postprocessing/citations.py"
```

---

## Implementation Strategy

### MVP First (Phases 1, 2, and 3)
1. Complete Setup (lexical FAQ index and prompts).
2. Complete Foundational (schemas, BM25 retriever, pass-through expansion).
3. Complete User Story 1 (Level-0 SubQ router).
4. **VALIDATE**: Ensure single-source queries auto-route correctly in <25ms.

### Incremental Delivery
1. Add User Story 2: Enable parallel DM1 and partitioned cross-corpus synthesis.
2. Add User Story 3: Enable deep category pre-filtering and INIF consultant persona.
3. Add User Story 4: Connect fullstack Next.js UI controls and citations.
4. Execute Polish & Zero-Drift KG refresh.

# Implementation Tasks: QueryGate & Conversational Memory Routing (006)

**Branch**: `006-query-gate-conversational-memory` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: System-wide renaming from `gated_rewriter` to `query_gate` and environment configuration.

- [X] T001 Rename `rag-core/src/rag_core/query/gated_rewriter.py` to `rag-core/src/rag_core/query/query_gate.py` and rename test file `rag-core/tests/test_gated_rewriter.py` to `rag-core/tests/test_query_gate.py`
- [X] T002 Update imports, class names (`QueryGate`), and references across `rag-core/src/rag_core/query/__init__.py`, `rag-core/src/rag_core/pipeline.py`, `rag-core/src/rag_core/graph/nodes.py`, and `rag-core/src/rag_core/graph/pipeline_graph.py`
- [X] T003 [P] Configure SubQ environment variables (`SUBQ_API_KEY`) and endpoints in `rag-core/src/rag_core/config.py` and `backend/app/core/config.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data models, SubQ client, summarizer, and session store extensions that MUST be complete before user stories begin.

**⚠️ CRITICAL**: No user story implementation can begin until this phase is complete.

- [X] T004 Implement core domain Enums (`CurriculumIntentTier`, `ContextNeedTier`, `HistoryRelevanceTier`) and Pydantic models (`SelectedHistoryItem`, `QueryGateDecision`, `QASummaryRecord`) in `rag-core/src/rag_core/indexing/science_schema.py`
- [X] T005 [P] Implement `SubQDecisionClient` supporting `subq-snap-preview` and parallel 10-10 chunking via `asyncio.gather` against `POST https://api.subq.ai/v1/systemone` in `rag-core/src/rag_core/query/subq_client.py`
- [X] T006 [P] Implement `ConversationSummarizer` with structured batch LLM call for backfilling missing summaries and a 2.0s fallback guard (250 chars) in `rag-core/src/rag_core/query/summarizer.py`
- [X] T007 [P] Extend session store schema and methods to persist `q_and_a_summary` in turn metadata in `backend/app/services/session_store.py`
- [X] T008 Extend `RAGGraphState` in `rag-core/src/rag_core/graph/state.py` with `query_gate_decision`, `intent_tier`, `context_need_tier`, `selected_history_items`, `selected_context_turn_id`, `q_and_a_summary`, and `conversational_context_badge`
- [X] T009 [P] Refresh Codebase Knowledge Graph via `codebase-memory-mcp index_repository(repo_path='/Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG', persistence=True)`

**Checkpoint**: Foundation ready - user story implementation can now begin. Codebase Knowledge Graph is refreshed and synchronized.

---

## Phase 3: User Story 1 - 3-Tier Curriculum Intent Gating & Partial Query Distillation (Priority: P1) 🎯 MVP

**Goal**: Classify queries into 3 tiers (`completely_irrelevant`, `partially_relevant`, `relevant`) and purify mixed queries before downstream processing, halting out-of-scope inquiries in <25ms with 0 vector tokens.

**Independent Test**: Send synthetic pure, mixed, and adversarial queries to `QueryGate` and assert exact 3-tier classification and transformed clean strings.

### Tests for User Story 1 ⚠️
- [X] T010 [P] [US1] Write contract unit tests in `rag-core/tests/test_query_gate.py` asserting exact 3-tier classification and transformed clean strings for pure, mixed, and adversarial queries
 
### Implementation for User Story 1
- [X] T011 [US1] Add prompt templates for partial query distillation in `rag-core/src/rag_core/generation/prompts.py`
- [X] T012 [US1] Implement 3-tier curriculum evaluation (`completely_irrelevant`, `partially_relevant`, `relevant`) in `QueryGate._call_jev_decision_model()` in `rag-core/src/rag_core/query/query_gate.py`
- [X] T013 [US1] Implement upfront query distillation for `partially_relevant` queries before history or retrieval evaluation in `rag-core/src/rag_core/query/query_gate.py`
- [X] T014 [US1] Update `create_enterprise_query_gate_node` and refusal routing in `rag-core/src/rag_core/graph/nodes.py` to immediately route `completely_irrelevant` queries to refusal in <25ms

**Checkpoint**: User Story 1 complete and independently testable (MVP reached).

---

## Phase 4: User Story 2 - Context Need Dispatching (`need_conv_hist_and_retrieve`) (Priority: P1)

**Goal**: Classify approved queries into 4 execution tiers, auto-coerce Turn 0 history requests, and bypass retrieval for `needs_history_only` (<600ms latency).

**Independent Test**: Send "پاسخ قبلی را خلاصه کن" and verify that Qdrant/BM25 retrieval latency is 0ms while previous turn context is supplied to synthesis with a "پاسخ بر اساس گفتگوی پیشین" badge.

### Tests for User Story 2 ⚠️
- [X] T015 [P] [US2] Write unit tests in `rag-core/tests/test_query_gate.py` for 4-tier context dispatching, Turn 0 cold-start coercion, and retrieval bypass

### Implementation for User Story 2
- [X] T016 [US2] Implement `need_conv_hist_and_retrieve` 4-tier choice question (`needs_history_and_retrieve`, `needs_history_only`, `needs_retrieve_only`, `needs_neither`) in `QueryGate` in `rag-core/src/rag_core/query/query_gate.py`
- [X] T017 [US2] Implement Turn 0 cold-start coercion logic (Option 1 -> Option 3, Option 2 -> friendly clarification prompt) in `rag-core/src/rag_core/query/query_gate.py`
- [X] T018 [US2] Update LangGraph conditional routing in `rag-core/src/rag_core/graph/pipeline_graph.py` so `needs_history_only` bypasses `hybrid_retrieve`, `neural_rerank`, and `parent_expansion`, routing directly to `synthesis`
- [X] T019 [US2] Update `synthesis_node` in `rag-core/src/rag_core/graph/nodes.py` and `rag_service.py` to set empty citations (`citations: []`) and attach `conversational_context_badge = "پاسخ بر اساس گفتگوی پیشین"` for `needs_history_only`

**Checkpoint**: User Stories 1 AND 2 both work independently and in combination.

---

## Phase 5: User Story 3 - Decision Model 2: History Relevance Filtering Across 20 Q&As (Priority: P2)

**Goal**: Inspect up to the last 20 Q&A pairs, deterministically backfill missing summaries via a single batch LLM call, execute SubQ Decision Model 2 with 10-10 parallel batching, and apply recent-5 full text vs older summary rule without numerical limit.

**Independent Test**: In a 10-turn session discussing Chapter 2 followed by Chapter 14, ask a question about Chapter 2. Verify that only Chapter 2 turns are flagged as needed, and turns within the 5 most recent receive full text while older turns receive summaries.

### Tests for User Story 3 ⚠️
- [X] T020 [P] [US3] Write contract tests in `rag-core/tests/test_query_gate.py` for SubQ Decision Model 2 parallel evaluation and fidelity-scaled context selection

### Implementation for User Story 3
- [X] T021 [US3] Implement deterministic missing summary detection and single batch LLM backfill call in `QueryGate.evaluate_and_route()` in `rag-core/src/rag_core/query/query_gate.py`
- [X] T022 [US3] Integrate SubQ Decision Model 2 call with 10-10 parallel chunking in `rag-core/src/rag_core/query/query_gate.py`
- [X] T023 [US3] Implement fidelity selection logic: extract full Q&A text for needed turns in recent 5, and summaries for older needed turns (turns 6-20) with no numerical cap in `rag-core/src/rag_core/query/query_gate.py`
- [X] T024 [US3] Wire selected history items into LangGraph state in `rag-core/src/rag_core/graph/nodes.py`

**Checkpoint**: History relevance filtering across 20 turns functional with SubQ parallel execution.

---

## Phase 6: User Story 4 - Hybrid Retrieval Augmentation with Needed Summaries (Priority: P2)

**Goal**: Concatenate purified user query with selected historical Q&A summaries for hybrid dense (BGE-M3) and sparse (BM25) retrieval when `needs_history_and_retrieve`.

**Independent Test**: Assert that a query like "در این فرایند نقش کبد چیست؟" retrieves digestive system chunks (Chapter 13) when the previous turn established digestion, rather than blood circulation chunks.

### Tests for User Story 4 ⚠️
- [X] T025 [P] [US4] Write integration test in `rag-core/tests/test_enterprise_graph.py` verifying that follow-up queries retrieve correct textbook chunks when augmented with needed summaries

### Implementation for User Story 4
- [X] T026 [US4] Implement query enrichment helper in `rag-core/src/rag_core/query/query_gate.py` combining purified query with selected summaries
- [X] T027 [US4] Update `create_enterprise_hybrid_retrieve_node` in `rag-core/src/rag_core/graph/nodes.py` to use enriched query for BGE-M3 and BM25 search

**Checkpoint**: Retrieval augmentation with historical summaries fully verified.

---

## Phase 7: User Story 5 - Incremental Turn Summary Caching in Response Synthesis (Priority: P3)

**Goal**: Synthesizer produces both the grounded answer and a structured summary (<= 512 tokens / ~6 lines) stored under `turn.metadata["q_and_a_summary"]`.

**Independent Test**: Inspect session store after a turn and verify that `turn.metadata["q_and_a_summary"]` is populated with a clean Persian paragraph.

### Tests for User Story 5 ⚠️
- [X] T028 [P] [US5] Write unit tests in `rag-core/tests/test_synthesis.py` for structured summary generation alongside answer synthesis

### Implementation for User Story 5
- [X] T029 [US5] Update synthesis prompt and response parser in `rag-core/src/rag_core/generation/synthesizer.py` to produce answer + structured summary (<= 512 tokens / ~6 lines)
- [X] T030 [US5] Update `RAGService.execute_chat` and streaming generator in `backend/app/services/rag_service.py` to persist `q_and_a_summary` to session storage upon turn completion

**Checkpoint**: Incremental summary generation active and cached without runtime delay.

---

## Phase 8: User Story 6 - Qwen-Style UI Reply & Context Selection When History Is Toggled Off (Priority: P3)

**Goal**: Provide a Qwen-style "Reply / Select as Context" action button on message cards with a dismissible floating preview chip above the input box, injecting the selected turn directly as context when history is OFF.

**Independent Test**: In the UI, toggle history OFF, click the Reply button on message 3, verify the quote chip appears above the prompt textarea, submit a question, and verify the backend synthesis prompt includes Turn 3 context without invoking Decision Model 2.

### Implementation for User Story 6
- [X] T031 [P] [US6] Update `backend/app/schemas/chat.py` to accept `selected_context_turn_id` in `ChatRequest` and return `conversational_context_badge` in `ChatResponse`
- [X] T032 [US6] Update `RAGService.execute_chat` and `QueryGate` in `backend/app/services/rag_service.py` to inject `selected_context_turn_id` as direct antecedent context (bypassing Decision Model 2) when history is OFF
- [X] T033 [P] [US6] Add "Reply / Select as Context" action button on message cards in `frontend/src/components/MessageItem.tsx`
- [X] T034 [US6] Implement floating dismissible quote chip above textarea with preview, indicator badge, and clear button in `frontend/src/components/ChatInterface.tsx`
- [X] T035 [US6] Render `conversational_context_badge` ("پاسخ بر اساس گفتگوی پیشین") in `frontend/src/components/MessageItem.tsx` when citations are empty

**Checkpoint**: All 6 user stories fully implemented and independently verified.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Observability, end-to-end regression validation, and Knowledge Graph synchronization.

- [X] T036 [P] Instrument Langfuse tracing metadata for Gate 1 decisions and SubQ Decision Model 2 evaluations in `rag-core/src/rag_core/query/query_gate.py` and `backend/app/core/telemetry.py`
- [X] T037 [P] Add end-to-end integration tests in `backend/tests/test_chat_endpoints.py` validating all 6 scenarios from `quickstart.md`
- [X] T038 Run full validation suite per `specs/006-query-gate-conversational-memory/quickstart.md` (`pytest rag-core/tests/`, `pytest backend/tests/`, `npm run build`)
- [X] T039 Refresh Codebase Knowledge Graph (`.codebase-memory/graph.db.zst`) via `codebase-memory-mcp index_repository(repo_path='/Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG', persistence=True)`
- [X] T040 Verify Zero Drift (Principle X Gate: `codebase-memory-mcp detect_changes` reports `changed_count == 0`)
- [X] T041 Update `development_log.md` with Milestone 27 architectural report

---

## Dependencies & Execution Order

### Phase Dependencies
- **Setup (Phase 1)**: Can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (Phase 1). BLOCKS all user stories.
- **User Stories (Phase 3+)**: Depend on Foundational (Phase 2).
  - US1 (P1) -> US2 (P1) -> US3 (P2) -> US4 (P2) -> US5 (P3) -> US6 (P3).
- **Polish (Phase 9)**: Depends on all user stories completing.

### Parallel Execution Opportunities
- Setup: T003 in parallel with T001/T002.
- Foundational: T005, T006, T007 in parallel after T004.
- User Story 1: T010 (Test) can run before or in parallel with T011.
- User Story 2: T015 (Test) before T016.
- User Story 3: T020 (Test) before T021.
- User Story 6: T031 (Backend schema) and T033 (Frontend button) in parallel.
- Polish: T036 and T037 in parallel.

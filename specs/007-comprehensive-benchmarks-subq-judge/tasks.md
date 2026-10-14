# Tasks: Comprehensive Multi-Suite Benchmarks, Deterministic Code Gates & SubQ Decision LLM-as-a-Judge (007)

**Branch**: `007-comprehensive-benchmarks-subq-judge`  
**Input**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/007-comprehensive-benchmarks-subq-judge/spec.md), [plan.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/007-comprehensive-benchmarks-subq-judge/plan.md), [data-model.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/007-comprehensive-benchmarks-subq-judge/data-model.md), [contracts/](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/007-comprehensive-benchmarks-subq-judge/contracts/)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic directory structure

- [X] T001 Create directory structure for calibrated dataset suites in `rag-core/src/rag_core/evaluation/datasets/suites/`, `rag-core/src/rag_core/evaluation/datasets/dev/`, and `rag-core/src/rag_core/evaluation/datasets/test/`
- [X] T002 [P] Update environment configuration with SubQ judge settings (`SUBQ_ENDPOINT`, `SUBQ_API_KEY`, `SUBQ_MODEL`) in `rag-core/src/rag_core/config.py` and `.env.example`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure and Pydantic schemas that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 [P] Implement Pydantic v2 schemas (`BenchmarkItem`, `SuiteType`, `DifficultyLevel`, `HistoricalTurn`, `GateResult`, `SubQJudgeVerdict`, `ConfusionMatrixResult`, `EvaluationReport`) in `rag-core/src/rag_core/evaluation/benchmark_schema.py`
- [X] T004 [P] Implement contract validation test verifying JSON Schema conformity against `specs/007-comprehensive-benchmarks-subq-judge/contracts/benchmark_item_schema.json` in `rag-core/tests/test_benchmark_contracts.py`
- [X] T005 [P] Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)`

**Checkpoint**: Foundation ready - schemas and contracts verified. User story implementation can now begin.

---

## Phase 3: User Story 1 - Multi-Suite Calibrated Benchmark Datasets (Priority: P1) 🎯 MVP

**Goal**: Deliver a comprehensive 360-item master benchmark across 5 suites with controlled difficulty (20% Easy, 50% Medium, 30% Hard/Adversarial) and a sealed 25% dev (90 items) / 75% test (270 items) partition.

**Independent Test**: Execute `test_benchmark_suites.py` to confirm all 5 suites load, validate against `BenchmarkItem`, meet difficulty distributions, and enforce split boundaries without leakage.

### Tests for User Story 1

- [X] T006 [P] [US1] Write dataset schema validation, difficulty ratio, and split boundary tests in `rag-core/tests/test_benchmark_suites.py`

### Implementation for User Story 1

- [X] T007 [P] [US1] Generate calibrated 120-item Grade 7 Science dataset (15 chapters, multi-hop, formulas, diagrams; 20% Easy, 50% Medium, 30% Hard) in `rag-core/src/rag_core/evaluation/datasets/suites/science_curriculum.csv`
- [X] T008 [P] [US1] Generate calibrated 80-item Administrative FAQ dataset (term collisions, multi-condition circulars, colloquial phrasing) in `rag-core/src/rag_core/evaluation/datasets/suites/administrative_faq.csv`
- [X] T009 [P] [US1] Generate calibrated 60-item Conversational Memory dataset (multi-turn histories, pronoun resolution, Turn 0 empty history) in `rag-core/src/rag_core/evaluation/datasets/suites/conversational_memory.csv`
- [X] T010 [P] [US1] Generate calibrated 50-item Subquery Decomposition dataset (compound queries requiring multi-subquery split) in `rag-core/src/rag_core/evaluation/datasets/suites/subquery_decomposition.csv`
- [X] T011 [P] [US1] Generate calibrated 50-item Adversarial Refusal dataset (prompt injection, out-of-domain commercial banking) in `rag-core/src/rag_core/evaluation/datasets/suites/adversarial_refusal.csv`
- [X] T012 [US1] Implement dataset loader and partitioner establishing sealed 25% dev (90 items) and 75% test (270 items) splits in `rag-core/src/rag_core/evaluation/load_dataset.py`

**Checkpoint**: User Story 1 complete. Master benchmark dataset is generated, partitioned, sealed, and verified.

---

## Phase 4: User Story 2 - SubQ System One LLM-as-a-Judge Integration (Priority: P1)

**Goal**: Implement `SubQJudge` using the SubQ Decisions API (`subq-snap-preview`) with 4 System One rubrics (faithfulness, relevancy, citation attribution, turn relevance), 10-10 parallel batching (< 600ms), and automatic fallback to local `LLMJudge`.

**Independent Test**: Pass a test batch of query-context-response tuples to `SubQJudge` and assert typed verdicts, confidence scores, < 600ms latency, and graceful fallback when the API key is invalid.

### Tests for User Story 2

- [X] T013 [P] [US2] Write unit tests for SubQJudge batch evaluation, System One rubrics, and local LLM fallback in `rag-core/tests/test_subq_judge.py`

### Implementation for User Story 2

- [X] T014 [US2] Implement `SubQJudge` with 4 rubrics (`faithfulness`, `answer_relevancy`, `citation_attribution`, `turn_relevance`) and concurrent 10-10 batching in `rag-core/src/rag_core/evaluation/subq_judge.py`
- [X] T015 [US2] Implement automatic fallback mechanism from `SubQJudge` to `LLMJudge` (Nine Router / Gemini CLI) in `rag-core/src/rag_core/evaluation/subq_judge.py`

**Checkpoint**: User Stories 1 and 2 complete. High-speed System One judge is operational.

---

## Phase 5: User Story 3 - Expanded Layer 1 Deterministic Code Gates (Priority: P2)

**Goal**: Expand Layer 1 deterministic code gates (< 5ms, 0 tokens) to verify Persian refusal regex, domain routing keywords, retrieval Hit@k, citation brackets, and schema contracts, resolving > 35% of evaluation items deterministically.

**Independent Test**: Run `test_code_gates_expanded.py` across boundary test cases and verify that `gated=True`, execution time is < 5ms, and 0 LLM calls are triggered.

### Tests for User Story 3

- [X] T016 [P] [US3] Write unit tests for expanded deterministic code gates in `rag-core/tests/test_code_gates_expanded.py`

### Implementation for User Story 3

- [X] T017 [US3] Expand normalized Persian refusal regex patterns and fuzzy phrase matchers in `rag-core/src/rag_core/evaluation/code_gates.py`
- [X] T018 [US3] Implement deterministic domain routing, Hit@1, Hit@3, Hit@5, MRR calculation, and citation syntax validation in `rag-core/src/rag_core/evaluation/code_gates.py`

**Checkpoint**: Layer 1 deterministic defense resolves boundary items in < 5ms with zero tokens.

---

## Phase 6: User Story 4 - Decoupled Retrieval vs. Generation Diagnostic Matrix (Priority: P2)

**Goal**: Implement a 2x2 confusion matrix (TP, FP, TN, FN) to cleanly separate retrieval errors (missing chunks) from generation errors (hallucination), and calculate Hallucination Catch Rate = `TN / (TN + FP)`.

**Independent Test**: Run decoupled evaluation over synthetic retrieval/generation combinations and assert correct classification into the 4 diagnostic quadrants.

### Tests for User Story 4

- [X] T019 [P] [US4] Write unit tests for 2x2 confusion matrix classification and hallucination catch rate calculation in `rag-core/tests/test_decoupled_eval.py`

### Implementation for User Story 4

- [X] T020 [US4] Implement decoupled diagnostic evaluator and confusion matrix calculator in `rag-core/src/rag_core/evaluation/decoupled_eval.py`

**Checkpoint**: Error diagnosis is decoupled between retriever and synthesizer.

---

## Phase 7: User Story 5 - Mentor-Ready Executive Evaluation Dashboard & Langfuse Telemetry (Priority: P3)

**Goal**: Implement automated Markdown and JSON report generator comparing Ground Zero Baseline vs Current Run vs Target KPIs, with failure taxonomies and Langfuse trace links.

**Independent Test**: Execute `run_eval.py --export-report` and verify that `evaluation_report.md` compiles with the executive KPI scorecard, confusion matrix, and Langfuse trace links.

### Implementation for User Story 5

- [X] T021 [P] [US5] Implement automated Markdown report generator comparing Ground Zero Baseline vs Current Run vs Target KPIs in `rag-core/src/rag_core/evaluation/report_generator.py`
- [X] T022 [US5] Update CLI runner `rag-core/src/rag_core/evaluation/run_eval.py` to support `--split`, `--suite`, `--judge (subq|jev|llm|all)`, `--limit`, `--export-report`, and Langfuse trace score attachment
- [X] T023 [US5] Add `make eval-dev` and `make eval-report` convenience commands in `Makefile`

**Checkpoint**: End-to-end evaluation harness runs with single command and outputs publication-ready mentor reports.

---

## Phase 8: User Story 6 - Interactive Frontend Evaluation Dashboard & Live Pipeline Runner (Priority: P2)

**Goal**: Build a frontend Evaluation Dashboard section and backend REST endpoints where mentors can run any benchmark suite (or all), configure execution options (e.g. Retrieve Only vs End-to-End, judge model, split, limit), view real-time streaming progress, and explore live measurement scorecards, 2x2 confusion matrix, and failure taxonomy directly in the UI.

**Independent Test**: Open frontend Evaluation Modal, select `science_curriculum` in `retrieve_only` mode with limit 10, click "Start Benchmark Run", observe the live progress bar, and verify real-time metric cards and retrieved chunks rendered upon completion.

### Tests for User Story 6

- [X] T024 [P] [US6] Write integration test verifying `/api/v1/evaluation/run` dispatch and `/status/{run_id}` polling in `backend/tests/test_evaluation_endpoints.py`

### Implementation for User Story 6

- [X] T025 [P] [US6] Implement REST schemas (`EvaluationRunRequest`, `EvaluationRunStatus`, `EvaluationScope`) in `backend/app/schemas/evaluation.py`
- [X] T026 [US6] Implement background evaluation worker service in `backend/app/services/evaluation_service.py` delegating to `rag_core.evaluation.run_eval` with state tracking
- [X] T027 [US6] Expose REST endpoints (`POST /api/v1/evaluation/run`, `GET /status/{run_id}`, `GET /suites`, `GET /runs`) in `backend/app/api/v1/endpoints/evaluation.py` and register in `backend/app/api/v1/router.py`
- [X] T028 [P] [US6] Add TypeScript interfaces for evaluation runs, progress, and reports in `frontend/src/types/index.ts`
- [X] T029 [US6] Implement `EvaluationModal.tsx` component with suite selector, option dropdowns (scope, split, judge, limit), live progress bar, measurement scorecard cards, 2x2 confusion matrix, failure explorer, and report download button in `frontend/src/components/EvaluationModal.tsx`
- [X] T030 [US6] Add Evaluation Dashboard launcher button and state handler in `frontend/src/components/Sidebar.tsx` and integrate modal into `frontend/src/app/page.tsx`

**Checkpoint**: Full visual evaluation dashboard is live in the frontend, enabling mentors to launch runs and inspect live measurements.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation for mentors, quickstart validation, and codebase knowledge graph synchronization.

- [X] T031 [P] Create comprehensive mentor evaluation guide and KPI reference document in `docs/mentor_evaluation_framework_and_kpis.md`
- [X] T032 Execute quickstart end-to-end validation scenarios per `specs/007-comprehensive-benchmarks-subq-judge/quickstart.md`
- [X] T033 Refresh Codebase Knowledge Graph (`.codebase-memory/graph.db.zst`) via codebase-memory-mcp `index_repository(persistence=True)`
- [X] T034 Verify Zero Drift (Principle X Gate: `detect_changes` reports `changed_count: 0`)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    │
    ▼
Phase 2: Foundational (Blocking Prerequisites: Schemas & Contracts)
    │
    ├───────────────────────────────────────┐
    ▼                                       ▼
Phase 3: User Story 1 (Datasets)    Phase 4: User Story 2 (SubQ Judge)
    │                                       │
    ├───────────────────────────────────────┘
    ▼
Phase 5: User Story 3 (Deterministic Code Gates)
    │
    ▼
Phase 6: User Story 4 (Decoupled Matrix)
    │
    ▼
Phase 7: User Story 5 (Reporting & CLI Runner)
    │
    ▼
Phase 8: User Story 6 (Frontend Evaluation Dashboard)
    │
    ▼
Phase 9: Polish & Knowledge Graph Refresh
```

### Parallel Opportunities

- T001, T002 (Setup tasks) can run in parallel.
- T003, T004 (Foundational schemas and contracts) can run in parallel.
- T007, T008, T009, T010, T011 (Dataset creation for all 5 suites) can run in parallel.
- T013, T016, T019, T024 (Unit and integration tests across stories) can be prepared in parallel.
- T021, T025, T028, T031 (Report generator, backend schemas, frontend types, and mentor documentation) can run in parallel.

---

## Implementation Strategy: MVP First

1. **Step 1 (MVP Foundation)**: Complete Phase 1 (Setup) and Phase 2 (Pydantic Schemas).
2. **Step 2 (MVP Pipeline)**: Implement Phase 3 (US1 - Datasets) and Phase 4 (US2 - SubQ Judge).
3. **Step 3 (Stop & Validate)**: Run `SubQJudge` on a 10-item sample from the dev split to verify sub-500ms latency and typed verdicts.
4. **Step 4 (Incremental Delivery)**: Add Layer 1 deterministic gates (US3), decoupled matrix (US4), and the automated report generator (US5).
5. **Step 5 (Mentor Review)**: Export `docs/mentor_evaluation_framework_and_kpis.md` and present empirical deltas.

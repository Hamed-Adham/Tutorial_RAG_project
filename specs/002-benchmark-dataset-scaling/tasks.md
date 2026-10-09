# Implementation Tasks: Benchmark Dataset Scaling & Comprehensive Quality Evaluation

**Feature Branch**: `002-benchmark-dataset-scaling`  
**Specification**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/002-benchmark-dataset-scaling/spec.md)  
**Plan**: [plan.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/002-benchmark-dataset-scaling/plan.md)  
**Created**: 2026-10-08  

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Configure generator templates, synthetic client wrapper, and contract tests.

- [X] T001 Configure benchmark scaling options and model parameters in `rag-core/src/rag_core/config.py`
- [X] T002 [P] Create contract validation unit test for benchmark items and report schemas in `rag-core/tests/test_benchmark_schema.py`
- [X] T003 [P] Create synthetic generation client wrapper supporting Nine Router `gemini-cli` with temperature control and retry backoff in `rag-core/src/rag_core/evaluation/synthetic_client.py`


---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Ingest grounding catalogs and prepare curriculum section loaders.

- [X] T004 Implement canonical catalog ID mapper and loader for `data/faq_qa_pairs.csv` (`faq_000` to `faq_133`) in `rag-core/src/rag_core/evaluation/faq_catalog.py`
- [X] T005 [P] Implement curriculum section grounder reading `data/vlm_parsed/manifest.json` and stitched markdown files in `rag-core/src/rag_core/evaluation/textbook_grounder.py`
- [X] T006 Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)` to record foundational baseline


**Checkpoint**: Foundation ready. User stories can proceed.

---

## Phase 3: User Story 1 - Multi-Perspective FAQ Evaluation (Priority: P1) 🎯 MVP

**Goal**: Scale administrative FAQ benchmark to 200+ items testing colloquial phrasings, cross-category keyword clashes, multi-row combinations, and out-of-scope refusals.

**Independent Test**: Run retrieval and synthesis against the 200+ FAQ items and verify correct catalog row retrieval and response accuracy.

### Tests for User Story 1
- [X] T007 [P] [US1] Contract test for FAQ synthetic generation and catalog mapping in `rag-core/tests/test_faq_benchmark.py`

### Implementation for User Story 1
- [X] T008 [P] [US1] Implement colloquial and dialect paraphrasing generator for FAQ rows (60% quota, 120 items) in `rag-core/src/rag_core/evaluation/generators/faq_generator.py`
- [X] T009 [P] [US1] Implement cross-category lexical ambiguity generator (*ضمانت*, *تسهیلات*, *سود* clashes across departments, 15% quota, 30 items) in `rag-core/src/rag_core/evaluation/generators/faq_generator.py`
- [X] T010 [US1] Implement multi-row composite policy query generator (e.g. loan ceilings + collateral requirements, 15% quota, 30 items) in `rag-core/src/rag_core/evaluation/generators/faq_generator.py`
- [X] T011 [US1] Implement out-of-scope non-fund refusal query generator (10% quota, 20 items) in `rag-core/src/rag_core/evaluation/generators/faq_generator.py`
- [X] T012 [US1] Assemble and validate 200+ FAQ benchmark items into `rag-core/src/rag_core/evaluation/datasets/suites/administrative_faq.csv`


**Checkpoint**: User Story 1 fully functional and testable independently (MVP reached).

---

## Phase 4: User Story 2 - Full-Curriculum Science Benchmark (Priority: P2)

**Goal**: Expand science curriculum benchmark across all 15 chapters (180+ items) using pre-parsed VLM Markdown, testing factoids, multi-hop reasoning, diagrams, and experiments.

**Independent Test**: Verify that all 15 chapters have at least 10 items each, covering diagrams, experiments, and cross-chapter synthesis.

### Tests for User Story 2
- [X] T013 [P] [US2] Contract test for 15-chapter science benchmark coverage in `rag-core/tests/test_science_benchmark.py`

### Implementation for User Story 2
- [X] T014 [P] [US2] Implement curriculum factoid and multi-hop reasoning generator across Chapters 1 to 15 in `rag-core/src/rag_core/evaluation/generators/science_generator.py`
- [X] T015 [P] [US2] Implement diagram extraction query generator querying visual description blocks across all chapters in `rag-core/src/rag_core/evaluation/generators/science_generator.py`
- [X] T016 [US2] Implement laboratory experiment procedure query generator querying `<experiment>` and `<activity>` tagged sections in `rag-core/src/rag_core/evaluation/generators/science_generator.py`
- [X] T017 [US2] Assemble and validate 180+ science benchmark items with chapter and section attribution into `rag-core/src/rag_core/evaluation/datasets/suites/science_textbook.csv`

**Checkpoint**: User Stories 1 and 2 functional and testable.

---

## Phase 5: User Story 3 - Decoupled Retrieval vs. Generation Evaluation (Priority: P3)

**Goal**: Decouple intermediate retrieval metrics (`Hit@1`, `Hit@3`, `Hit@5`, `MRR`, `Context Recall`) from generation metrics (`Faithfulness`, `Answer Relevancy`), enabling zero-token retrieval benchmarks.

**Independent Test**: Execute `--retrieval-only` mode and confirm sub-second scoring without LLM judge calls.

### Tests for User Story 3
- [X] T018 [P] [US3] Contract test for decoupled retrieval scoring in `rag-core/tests/test_decoupled_eval.py`

### Implementation for User Story 3
- [X] T019 [US3] Implement deterministic retrieval evaluation engine (`Hit@1`, `Hit@3`, `Hit@5`, `MRR`, `Context Recall`) in `rag-core/src/rag_core/evaluation/retrieval_scorer.py`
- [X] T020 [US3] Refactor CLI evaluation runner `rag-core/src/rag_core/evaluation/run_eval.py` to support `--retrieval-only` mode with zero LLM judge tokens
- [X] T021 [US3] Update `rag-core/src/rag_core/evaluation/harness.py` to emit decoupled reports matching `evaluation_report_schema.json`

**Checkpoint**: Decoupled retrieval and generation evaluation fully operational.

---

## Phase 6: User Story 4 - Citation & Provenance Verification (Priority: P4)

**Goal**: Audit generated source citations against gold reference chunks to measure Citation Precision and Citation Recall.

**Independent Test**: Evaluate answers with injected valid and invalid citations, confirming proper attribution scoring.

### Tests for User Story 4
- [X] T022 [P] [US4] Contract test for citation provenance scoring in `rag-core/tests/test_citation_eval.py`

### Implementation for User Story 4
- [X] T023 [US4] Implement citation parser and ground truth chunk matcher in `rag-core/src/rag_core/postprocessing/citations.py`
- [X] T024 [US4] Integrate citation precision and recall calculation into `rag-core/src/rag_core/evaluation/harness.py`

**Checkpoint**: Citation fidelity auditing operational across all evaluation suites.

---

## Phase 7: User Story 5 - Golden Dataset Synchronization & Observability (Priority: P5)

**Goal**: Compile 400+ benchmark items into `golden_set.jsonl` with deterministic 75/25 dev/test split and synchronize idempotently to Langfuse Cloud.

**Independent Test**: Verify Langfuse Cloud datasets contain all 400+ items with complete metadata and zero duplicates.

### Tests for User Story 5
- [X] T025 [P] [US5] Contract test for master golden set schema and split purity in `rag-core/tests/test_golden_set_sync.py`

### Implementation for User Story 5
- [X] T026 [US5] Implement master dataset compiler and stratified 75/25 dev/test partitioner in `rag-core/src/rag_core/evaluation/build_standardized_benchmarks.py` emitting `golden_set.jsonl` (400+ items)
- [X] T027 [US5] Upgrade `rag-core/src/rag_core/evaluation/migrate_datasets_to_langfuse.py` to sync all 400+ items idempotently to Langfuse Cloud under `dev/*`, `test/*`, and `suites/*`
- [X] T028 [US5] Add `eval-retrieval`, `eval-full`, and `eval-gates` targets in `Makefile`

**Checkpoint**: Centralized dataset versioned, split, and synchronized to Langfuse Cloud.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Verification, documentation, and constitution compliance.

- [X] T029 [P] Run full test suite (`pytest`) and verify all unit and schema tests pass
- [X] T030 Execute quickstart validation scenarios from `specs/002-benchmark-dataset-scaling/quickstart.md`
- [X] T031 Record benchmark scaling metrics and baseline comparison in `development_log.md`
- [X] T032 Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)`
- [X] T033 Verify Zero Drift (Principle X Gate: `detect_changes` reports `changed_count == 0`)

---

## Dependencies & Execution Order

```text
Phase 1: Setup (T001..T003)
       │
       ▼
Phase 2: Foundational (T004..T006)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Phase 3: US1 FAQ Scaling (T007..T012)    Phase 4: US2 Science Scaling (T013..T017)
       │                                         │
       └────────────────────┬────────────────────┘
                            │
                            ▼
       Phase 5: US3 Decoupled Evals (T018..T021)
                            │
                            ▼
       Phase 6: US4 Citation Verification (T022..T024)
                            │
                            ▼
       Phase 7: US5 Master Sync & Langfuse (T025..T028)
                            │
                            ▼
       Phase 8: Polish & Constitution Gates (T029..T033)
```

---

## Parallel Execution Opportunities

- **Phase 1**: T002 (Schema tests) and T003 (Synthetic client) can run in parallel.
- **Phase 2**: T004 (FAQ catalog) and T005 (Textbook grounder) can run in parallel.
- **User Stories 1 & 2**: Once Phase 2 completes, US1 (FAQ generation) and US2 (Science generation) can run concurrently.
- **Phase 3**: T008 (Paraphrases) and T009 (Lexical ambiguity) can run in parallel.
- **Phase 4**: T014 (Factoids), T015 (Diagrams), and T016 (Experiments) can run in parallel.

---

## Implementation Strategy (MVP First)

1. **Step 1**: Complete Phase 1 (Setup) and Phase 2 (Foundational).
2. **Step 2**: Execute **Phase 3 (User Story 1 - FAQ Scaling)**. This delivers an immediate MVP scaling the FAQ corpus from 40 to 200+ items.
3. **Step 3**: Execute Phase 4 (Science Curriculum scaling across all 15 chapters).
4. **Step 4**: Execute Phase 5 & 6 (Decoupled Retrieval & Citation Verification).
5. **Step 5**: Execute Phase 7 (Master compile to `golden_set.jsonl` and sync to Langfuse Cloud).
6. **Step 6**: Execute Phase 8 (Verification and Constitution Knowledge Graph refresh).

---

## Phase 9: Convergence (Ratified Clarifications Alignment)

**Goal**: Align retrieval scoring and citation verification with the 5 newly ratified clarifications in `spec.md` (Q-1 through Q-5).

### Findings Summary
| ID | Area | Severity | Requirement | Description | Target File |
|---|---|---|---|---|---|
| GAP-01 | US3 Decoupled Evals | High | FR-006 / Q-1 | Out-of-scope queries (`gold_chunks == []`) currently receive artificial 1.0 hit rate instead of metric exclusion. | `rag-core/src/rag_core/evaluation/retrieval_scorer.py` |
| GAP-02 | US4 Citation Auditing | High | FR-007 / Q-3 | Citation parser currently accepts loose bracketed tokens rather than strictly requiring Markdown links (`[Title](...#page=N)`). | `rag-core/src/rag_core/postprocessing/citations.py` |
| GAP-03 | US4 Citation Tests | Medium | FR-007 / Q-3 | Unit tests must enforce strict Markdown link extraction and negative rejection of plain text brackets. | `rag-core/tests/test_citation_eval.py` |
| GAP-04 | US3 Retrieval Tests | Medium | FR-006 / Q-1 | Unit tests must verify out-of-scope exclusion from aggregate `Hit@k` and `MRR` rates. | `rag-core/tests/test_decoupled_eval.py` |
| GAP-05 | Polish & Hygiene | Low | Principle X/XII | Re-index repository into Codebase Knowledge Graph and verify zero drift. | `.codebase-memory/graph.db.zst` |

### Tasks
- [X] T034 [P] [US3/Q-1] Update `RetrievalScorer.score_query` and `aggregate` in `rag-core/src/rag_core/evaluation/retrieval_scorer.py` to tag out-of-scope queries (`gold_chunks == []`) and exclude them from `Hit@k` and `MRR` calculations per Clarification Q-1
- [X] T035 [P] [US4/Q-3] Update citation parser in `rag-core/src/rag_core/postprocessing/citations.py` to extract and validate strict Markdown link citations (`[Title](...#page=N)` / `[Title](...#row=N)`), treating unlinked bracketed text as invalid per Clarification Q-3
- [X] T036 [US4/Q-3] Add unit test cases in `rag-core/tests/test_citation_eval.py` asserting strict Markdown link extraction and negative rejection of plain bracketed references
- [X] T037 [US3/Q-1] Add unit test cases in `rag-core/tests/test_decoupled_eval.py` verifying that out-of-scope queries do not artificially inflate aggregate `Hit@1`, `Hit@3`, and `MRR`
- [X] T038 Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)` and verify zero drift

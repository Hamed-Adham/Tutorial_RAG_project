# Implementation Tasks: Enterprise Science RAG Pipeline

**Feature Branch**: `003-enterprise-science-pipeline`  
**Specification**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/spec.md)  
**Plan**: [plan.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/plan.md)  
**Created**: 2026-10-10  

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize data models, dependencies, and component contracts.

- [X] T001 Configure science pipeline dependencies (`rank-bm25`, `hazm`, `sentence-transformers`) in `rag-core/pyproject.toml`
- [X] T002 [P] Define domain schemas (`ScienceChunk`, `ParentSection`, `RetrievalCandidate`, `RerankedCandidate`, `AdvancedRAGResult`) in `rag-core/src/rag_core/indexing/science_schema.py` per data-model.md
- [X] T003 [P] Create interface contracts for `IBM25Retriever`, `IRRFFusion`, `ICrossEncoderReranker`, and `IAdvancedRAGPipeline` in `rag-core/src/rag_core/contracts.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure and serialized lexical artifacts required by all user stories.

**⚠️ CRITICAL**: No user story can begin until this phase is complete.

- [X] T004 Implement in-memory BM25 index builder with `hazm` Persian tokenization, stemming, and disk serialization in `rag-core/src/rag_core/indexing/bm25_index.py`
- [X] T005 [P] Implement CLI script to build and serialize the 15-chapter science index from `data/vlm_parsed/manifest.json` into `data_index/science_bm25.pkl` in `rag-core/src/rag_core/indexing/build_science_bm25.py`
- [X] T006 Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)` to record foundational baseline

**Checkpoint**: Foundation ready. Lexical indexer operational and Knowledge Graph synchronized.

---

## Phase 3: User Story 1 - Hybrid Retrieval (BM25 + Qdrant Dense) with RRF (Priority: P1) 🎯 MVP

**Goal**: Deliver sub-5ms BM25 search, parallel dense vector search, and Reciprocal Rank Fusion on the 15-chapter science textbook.

**Independent Test**: Execute `pytest rag-core/tests/test_hybrid_retrieval.py` and confirm that queries with exact scientific terms (e.g., *سانتریفیوژ*, *چگالی*) achieve higher `Hit@1` and `Hit@5` than dense-only retrieval without requiring downstream generation.

### Tests for User Story 1
- [X] T007 [P] [US1] Write contract unit test for BM25 retrieval, Dense vector search, and RRF rank merging in `rag-core/tests/test_hybrid_retrieval.py`

### Implementation for User Story 1
- [X] T008 [P] [US1] Implement sub-5ms in-memory `BM25Retriever` loading `data_index/science_bm25.pkl` in `rag-core/src/rag_core/retrieval/bm25.py`
- [X] T009 [P] [US1] Implement non-parametric `ReciprocalRankFusion` (`score = sum(1 / (60 + rank))`) in `rag-core/src/rag_core/fusion/rrf.py`
- [X] T010 [US1] Implement parallel `HybridRetriever` executing `asyncio.gather(dense, bm25)` with RRF candidate fusion in `rag-core/src/rag_core/retrieval/hybrid.py`
- [X] T011 [US1] Execute isolated retrieval test against `rag-core/tests/test_hybrid_retrieval.py` and verify passing MVP

**Checkpoint**: User Story 1 functional and independently testable (MVP reached).

---

## Phase 4: User Story 2 - Cross-Encoder Re-ranking & Dynamic Parent Chunk Expansion (Priority: P2)

**Goal**: Filter top 50 candidates down to top 5 using `BAAI/bge-reranker-v2-m3` (with 350ms circuit breaker) and dynamically expand child chunks to deduplicated parent sections (~2,500 token ceiling).

**Independent Test**: Execute `pytest rag-core/tests/test_cross_encoder_rerank.py` and `pytest rag-core/tests/test_parent_expansion.py` to verify re-ranking precision lift, timeout fallback, and token-bounded context expansion with `LongContextReorder`.

### Tests for User Story 2
- [X] T012 [P] [US2] Write contract unit test for Cross-Encoder re-ranking with 350ms timeout circuit breaker in `rag-core/tests/test_cross_encoder_rerank.py`
- [X] T013 [P] [US2] Write contract unit test for Parent Chunk Expansion and 2,500 context token ceiling in `rag-core/tests/test_parent_expansion.py`

### Implementation for User Story 2
- [X] T014 [US2] Implement `CrossEncoderReranker` using local `BAAI/bge-reranker-v2-m3` (CPU/MPS device selection) with 350ms timeout fallback in `rag-core/src/rag_core/fusion/reranker.py`
- [X] T015 [US2] Implement `ParentSectionExpander` resolving child chunks to parent sections (capped at max 3 unique sections / 2,500 tokens) with `LongContextReorder` in `rag-core/src/rag_core/postprocessing/expansion.py`

**Checkpoint**: High-precision neural reranking and parent context expansion operational.

---

## Phase 5: User Story 3 - Confidence-Gated Query Rewriting & Sub-query Decomposition (Priority: P3)

**Goal**: Route ambiguous or multi-turn queries through history-aware rewriting and sub-query decomposition using a TypeSafe Jev classifier while bypassing standalone queries to protect the latency budget.

**Independent Test**: Execute `pytest rag-core/tests/test_gated_rewriter.py` to verify that standalone queries bypass rewriter in <10ms with 0 tokens, while pronoun-heavy queries reconstruct full entity context.

### Tests for User Story 3
- [X] T016 [P] [US3] Write contract unit test for confidence-gated query rewriting and multi-hop sub-query decomposition in `rag-core/tests/test_gated_rewriter.py`

### Implementation for User Story 3
- [X] T017 [US3] Implement `GatedQueryRewriter` using TypeSafe Jev classifier confidence scoring to bypass standalone queries and rewrite conversational/ambiguous inquiries in `rag-core/src/rag_core/query/gated_rewriter.py`
- [X] T018 [US3] Implement sub-query decomposition for multi-concept curriculum questions in `rag-core/src/rag_core/query/decomposition.py`

**Checkpoint**: Query intelligence operational with zero latency overhead on simple prompts.

---

## Phase 6: User Story 4 - Latency-Budgeted Advanced Pipeline with Langfuse Trace Quality Metrics (Priority: P4)

**Goal**: Assemble end-to-end `AdvancedRAGPipeline` meeting p95 < 2.0s latency budget, streaming TTFT < 1.0s, strict Markdown citations, and asynchronous trace-level quality scoring in Langfuse.

**Independent Test**: Execute `pytest rag-core/tests/test_advanced_pipeline.py` and run benchmark against `science_textbook.csv` (180+ items) verifying `Hit@5 >= 0.90`, `MRR >= 0.80`, and `Faithfulness >= 0.85`.

### Tests for User Story 4
- [X] T019 [P] [US4] Write end-to-end integration test asserting latency budgets (<2.0s p95, <1.0s TTFT) and strict Markdown citations in `rag-core/tests/test_advanced_pipeline.py`

### Implementation for User Story 4
- [X] T020 [US4] Implement `AdvancedRAGPipeline` orchestrator unifying gated rewriting, parallel hybrid retrieval, cross-encoder reranking, parent expansion, and citation synthesis in `rag-core/src/rag_core/pipeline.py`
- [X] T021 [US4] Implement asynchronous background trace quality evaluators (Chunk Relevance binary 0/1, Context Adherence) dispatched via `asyncio.create_task` in `rag-core/src/rag_core/observability/trace_evaluators.py`
- [X] T022 [US4] Update evaluation CLI runner `rag-core/src/rag_core/evaluation/run_eval.py` to support `--pipeline advanced` benchmarking against `science_textbook.csv` (180+ items)

**Checkpoint**: End-to-end enterprise RAG pipeline complete, latency budgeted, and traced in Langfuse.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Automation, regression testing, and constitution governance compliance.

- [X] T023 [P] Add `eval-science-advanced` and `build-bm25` targets in root `Makefile`
- [X] T024 Execute full test suite (`pytest rag-core/tests/`) and quickstart validation guide in `specs/003-enterprise-science-pipeline/quickstart.md`
- [X] T025 Record empirical delta between Naive baseline and Advanced pipeline in `development_log.md`
- [X] T026 Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)`
- [X] T027 Verify Zero Drift (Principle X Gate: `detect_changes` reports `changed_count == 0`)

---

## Dependencies & Execution Order

```text
Phase 1: Setup (T001..T003)
       │
       ▼
Phase 2: Foundational (T004..T006)
       │
       ▼
Phase 3: User Story 1 - Hybrid Retrieval & RRF (T007..T011) 🎯 MVP
       │
       ▼
Phase 4: User Story 2 - Cross-Encoder & Parent Expansion (T012..T015)
       │
       ▼
Phase 5: User Story 3 - Confidence-Gated Rewriter (T016..T018)
       │
       ▼
Phase 6: User Story 4 - Advanced Pipeline & Observability (T019..T022)
       │
       ▼
Phase 7: Polish & Constitution Gates (T023..T027)
```

---

## Parallel Execution Opportunities

- **Phase 1**: T002 (Schemas) and T003 (Contracts) can run in parallel.
- **Phase 2**: T005 (CLI builder) can run parallel with schema verification.
- **Phase 3 (US1)**: T007 (Tests), T008 (BM25 retriever), and T009 (RRF fusion) can be authored in parallel.
- **Phase 4 (US2)**: T012 (Reranker test) and T013 (Expansion test) can run in parallel.
- **Phase 5 (US3)**: T016 (Rewriter tests) can be drafted in parallel with T014/T015.
- **Phase 7**: T023 (Makefile) can run in parallel with documentation tasks.

---

## Implementation Strategy (MVP First)

1. **Step 1**: Complete Phase 1 (Setup) and Phase 2 (Foundational - build `science_bm25.pkl`).
2. **Step 2**: Execute **Phase 3 (User Story 1 - Hybrid Retrieval & RRF)**. This delivers an immediate MVP lifting retrieval recall over dense-only search.
3. **Step 3**: Execute Phase 4 (Cross-Encoder & Parent Expansion) to maximize context precision.
4. **Step 4**: Execute Phase 5 (Gated Rewriter) to support multi-hop curriculum questions.
5. **Step 5**: Execute Phase 6 (End-to-End Orchestrator & Langfuse Trace Evaluators).
6. **Step 6**: Execute Phase 7 (Verification, Benchmark Run, and Knowledge Graph synchronization).

---

## Phase 8: Convergence

**Purpose**: Close remaining gaps between the specification, plan, and codebase implementations.

- [X] T028 [CRITICAL] Cut feature branch `003-enterprise-science-pipeline` from `002-benchmark-dataset-scaling` and stage atomic Conventional Commits referencing task IDs per Constitution XI/XII (contradicts)
- [X] T029 [HIGH] Implement streaming token generation (`query_stream`) in `AdvancedRAGPipeline` and `ResponseSynthesizer` to satisfy TTFT < 1.0s per SC-003, FR-014 (missing)
- [X] T030 [HIGH] Execute full 180-item benchmark sweep against `science_textbook` suite to measure and validate retrieval recall thresholds per SC-001, FR-013 (partial)
- [X] T031 [HIGH] Rebuild and persist Codebase Knowledge Graph artifact via codebase-memory-reindex to satisfy zero-drift invariant per Constitution X (missing)
- [X] T032 [MEDIUM] Formally audit, integrate into `Makefile`, and document science textbook Qdrant indexing script and vector cache artifact in `rag-core/src/rag_core/indexing/index_science_qdrant.py` per FR-002, plan: dense vector store (unrequested)
- [X] T033 [LOW] Document 4-flag Jev query transformation engine (`resolve_context`, `expand`, `decompose`, `bypass`, `rejected`) and dual-scorer audit (`--audit-jev`) in `quickstart.md` and feature contracts per FR-007, Constitution II (unrequested)


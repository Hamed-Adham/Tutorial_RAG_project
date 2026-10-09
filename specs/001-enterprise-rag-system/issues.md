# GitHub / Project Issues Tracker: Enterprise RAG + Text-to-SQL System

**Specification**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/spec.md) | **Plan**: [plan.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/plan.md) | **Tasks**: [tasks.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/tasks.md)

This document contains canonical, actionable issue representations for all tasks defined in `tasks.md`. It is formatted for direct manual or automated import into GitHub Issues, Jira, Linear, or GitLab.

---

## Issue Index & Summary

- **Total Issues**: 77
- **Status**: 24 Completed (`closed`), 53 Pending (`open`)
- **Phases**:
  - `phase-1-setup`: 7 issues (T001 - T006, T006b) - 100% Complete
  - `phase-2-foundations`: 7 issues (T007 - T011, T010b, T010c) - 100% Complete
  - `phase-3-us1-mvp`: 24 issues (T012 - T026, T022 - T022i) - 10 Complete, 14 Pending
  - `phase-4-us2-retrieval`: 11 issues (T027 - T037)
  - `phase-5-us3-agent-sql`: 10 issues (T038 - T047)
  - `phase-6-us4-enterprise`: 13 issues (T048 - T060)
  - `phase-7-ci-cd`: 5 issues (T061 - T065)

---

## Phase 1: Setup & Shared Infrastructure

### Issue #1: `T001: Initialize directory tree for rag-core, backend, and frontend`
- **State**: `closed` (Completed)
- **Labels**: `phase-1-setup`, `scaffolding`, `area:infra`
- **Target File**: Repository Root
- **Description**:
  Initialize directory tree for all 10 pipeline stages in `rag-core/src/rag_core/` (`manifest`, `parsing`, `chunking`, `indexing`, `query`, `retrieval`, `fusion`, `postprocessing`, `generation`, `agents`, `text_to_sql`, `evaluation`, `security`, `cache`, `observability`), `rag-core/tests/`, `backend/`, and `frontend/`.
- **Acceptance Criteria**:
  - Directory structure matches `plan.md` architectural specification.
  - All package folders contain `__init__.py`.

### Issue #2: `T002: Create rag-core pyproject.toml`
- **State**: `closed` (Completed)
- **Labels**: `phase-1-setup`, `parallelizable`, `area:rag-core`
- **Target File**: `rag-core/pyproject.toml`
- **Description**:
  Configure Python package metadata and core dependencies: LlamaIndex, LangChain, LangGraph, BGE-M3, Qdrant, RAGAS, Presidio, Langfuse, and Docling/PyMuPDF.

### Issue #3: `T003: Create backend pyproject.toml and requirements.txt`
- **State**: `closed` (Completed)
- **Labels**: `phase-1-setup`, `parallelizable`, `area:backend`
- **Target File**: `backend/pyproject.toml`
- **Description**:
  Configure FastAPI server dependencies: FastAPI, Uvicorn, SQLAlchemy, Alembic, and Pydantic v2.

### Issue #4: `T004: Create root .env.example`
- **State**: `closed` (Completed)
- **Labels**: `phase-1-setup`, `parallelizable`, `area:config`
- **Target File**: `.env.example`
- **Description**:
  Define connection parameters for Qdrant, PostgreSQL, Redis, Langfuse, Neo4j, and local/remote LLM endpoints.

### Issue #5: `T005: Create docker-compose.yml for local infrastructure`
- **State**: `closed` (Completed)
- **Labels**: `phase-1-setup`, `parallelizable`, `area:devops`
- **Target File**: `docker-compose.yml`
- **Description**:
  Provision local Docker containers for Qdrant, PostgreSQL, Redis, Langfuse, and Neo4j (`neo4j-rag`).

### Issue #6: `T006: Create root Makefile`
- **State**: `closed` (Completed)
- **Labels**: `phase-1-setup`, `parallelizable`, `area:devops`
- **Target File**: `Makefile`
- **Description**:
  Add targets: `make dev`, `make test`, `make eval`, and `make build`.

### Issue #6b: `T006b: Add --gates-only benchmark mode and test-gates / check Makefile targets`
- **State**: `closed` (Completed)
- **Labels**: `phase-1-setup`, `testing`, `area:devops`, `area:evaluation`
- **Target File**: `Makefile`, `rag-core/src/rag_core/naive/run_baseline_eval.py`
- **Description**:
  Implement `--gates-only` flag in `run_baseline_eval.py` to evaluate Layer 1 Deterministic Code Gates across all 100 benchmark items in < 2s without LLM calls. Add `make test-gates` and `make check` (`test` + `test-gates`) targets to root `Makefile`.
- **Acceptance Criteria**:
  - `make test` runs unit tests in `rag-core/tests/` (< 0.1s).
  - `make test-gates` evaluates 100 golden set items deterministically in < 2s with 0 token cost.
  - `make check` completes both checks successfully.



---

## Phase 2: Foundational Architecture

### Issue #7: `T007: Implement central configuration model`
- **State**: `closed` (Completed)
- **Labels**: `phase-2-foundations`, `area:rag-core`
- **Target File**: `rag-core/src/rag_core/config.py`
- **Description**:
  Implement central settings model using Pydantic Settings reading environment variables for Qdrant, Neo4j, LLM gateways, and evaluation thresholds.

### Issue #8: `T008: Configure structured JSON logging`
- **State**: `closed` (Completed)
- **Labels**: `phase-2-foundations`, `parallelizable`, `area:observability`
- **Target File**: `rag-core/src/rag_core/observability/logging.py`
- **Description**:
  Configure `structlog` for structured JSON output across all pipeline stages.

### Issue #9: `T009: Implement Langfuse tracing client`
- **State**: `closed` (Completed)
- **Labels**: `phase-2-foundations`, `parallelizable`, `area:observability`
- **Target File**: `rag-core/src/rag_core/observability/langfuse.py`
- **Description**:
  Implement centralized `LangfuseTracker` client and OpenInference span handlers for prompt, completion, and latency tracing across pipeline stages.

### Issue #10: `T010: Implement Qdrant vector database connection`
- **State**: `closed` (Completed)
- **Labels**: `phase-2-foundations`, `parallelizable`, `area:indexing`
- **Target File**: `rag-core/src/rag_core/indexing/vector_store.py`
- **Description**:
  Implement Qdrant client connection, HNSW collection initialization, and cosine similarity metric setup.

### Issue #11: `T010b: Implement Neo4j Knowledge Graph Store driver`
- **State**: `closed` (Completed)
- **Labels**: `phase-2-foundations`, `parallelizable`, `area:kg`
- **Target File**: `rag-core/src/rag_core/kg/neo4j_store.py`
- **Description**:
  Implement Neo4j driver with schema constraints (`Document`, `Chunk`, `Entity`, `Keyphrase`), relationships (`HAS_CHUNK`, `MENTIONS`, `SIMILAR_TO`), Cypher traversal, and offline fallback.

### Issue #11b: `T010c: Implement Knowledge Graph schema models`
- **State**: `closed` (Completed)
- **Labels**: `phase-2-foundations`, `parallelizable`, `area:kg`
- **Target File**: `rag-core/src/rag_core/kg/schema.py`
- **Description**:
  Define Knowledge Graph schema models (`DocumentNode`, `ChunkNode`, `EntityNode`, `KeyphraseNode`, `MultiHopCandidate`) and relationship enumerations.

### Issue #12: `T011: Implement BGE-M3 embedding loader`
- **State**: `closed` (Completed)
- **Labels**: `phase-2-foundations`, `parallelizable`, `area:indexing`
- **Target File**: `rag-core/src/rag_core/indexing/embeddings.py`
- **Description**:
  Implement BGE-M3 loader supporting int8 quantization and asymmetrical query prefix routing.

---

## Phase 3: User Story 1 - Core 10-Stage MVP Pipeline & Swiss Cheese Evaluation

### Issue #13: `T012: Implement PostgreSQL document catalog`
- **State**: `open`
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-1`
- **Target File**: `rag-core/src/rag_core/manifest/catalog.py`
- **Description**:
  Implement document manifest cataloging with stable `doc_id`, SHA-256 hash, tenant, and ACL tracking.

### Issue #14: `T013: Implement SHA-256 duplicate collapse & MinHash clustering`
- **State**: `open`
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-1`
- **Target File**: `rag-core/src/rag_core/manifest/deduplicator.py`
- **Description**:
  Implement exact hash collision collapse and MinHash near-duplicate clustering.

### Issue #15: `T014: Implement tiered parsing probe router & layout extractor`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-2`
- **Target File**: `rag-core/src/rag_core/parsing/router.py`, `tier0_layout.py`
- **Description**:
  Build router to probe document formats and extract layout blocks via PyMuPDF/Docling.

### Issue #16: `T015: Implement Canonical IR block schema`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-2`
- **Target File**: `rag-core/src/rag_core/parsing/schema.py`
- **Description**:
  Define Canonical IR schema (`type`, `text`, `page`, `bbox`).

### Issue #17: `T016: Implement structure-aware hierarchical chunking`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-3`
- **Target File**: `rag-core/src/rag_core/chunking/hierarchical.py`
- **Description**:
  Implement heading splitter prepending breadcrumbs (`Doc > Section > Subsection (page)`).

### Issue #18: `T017: Implement Parent-Child chunking`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-3`
- **Target File**: `rag-core/src/rag_core/chunking/parent_child.py`
- **Description**:
  Index small retrieval chunks linked to larger parent context chunks.

### Issue #19: `T018: Implement unified multi-index indexer`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-4`
- **Target File**: `rag-core/src/rag_core/indexing/indexer.py`
- **Description**:
  Orchestrate pipeline from manifest -> parsing -> chunking -> embedding -> Qdrant.

### Issue #20: `T019: Implement dense HNSW vector retriever`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-6`
- **Target File**: `rag-core/src/rag_core/retrieval/dense.py`
- **Description**:
  Implement dense vector retrieval with BGE-M3 query prefix.

### Issue #21: `T020: Implement page and bbox source citation formatter`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-8`
- **Target File**: `rag-core/src/rag_core/postprocessing/citations.py`
- **Description**:
  Format citations with source page and bounding box coordinates.

### Issue #22: `T021: Implement grounded LLM response synthesizer`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-9`
- **Target File**: `rag-core/src/rag_core/generation/synthesizer.py`, `prompts.py`
- **Description**:
  Synthesize grounded answers using GPT-4o / local Llama-3.1 with "I don't know" fallback.

### Issue #23: `T022: Create curated 100-item Golden Benchmark Dataset`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/datasets/golden_set.jsonl`, `load_dataset.py`
- **Description**:
  Curate 100 Q&A pairs across 4 suites: Science Textbook (40), Administrative FAQ (30), Domain Routing (15), and Adversarial/Refusal (15).

### Issue #24: `T022b: Implement 5-Part Judge Rubrics`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/rubrics.py`
- **Description**:
  Implement Persona, Binary Criteria, XML Data Boundaries, Persian Few-Shot Examples, and Mandatory Step-by-Step Chain-of-Thought rubrics.

### Issue #25: `T022c: Implement Meta-Evaluation Confusion Matrix`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/meta_eval.py`
- **Description**:
  Compute TP, FP, TN, FN, Precision, Recall, Hallucination Catch Rate, and Pass^k / Pass@k reliability scores.

### Issue #26: `T022d: Implement Layer 1 Deterministic Code Gates`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/code_gates.py`
- **Description**:
  Implement regex refusal patterns and domain keyword sanity checks executing in `< 5ms` with zero token cost.

### Issue #27: `T022e: Implement execution duration tracking (latency_seconds) in Judge`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/naive/judge_evaluator.py`
- **Description**:
  Measure and record judge LLM invocation wall-clock duration in `latency_seconds` for every evaluation sample.

### Issue #28: `T022f: Implement local Ragas adapter`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/ragas_adapter.py`
- **Description**:
  Bind Faithfulness, Answer Relevancy, Context Precision, and Context Recall to local OpenAI-compatible endpoint (`http://localhost:20128/v1`).

### Issue #29: `T022g: Implement benchmark evaluation runner with gate bypass`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `stage-10`
- **Target File**: `rag-core/src/rag_core/naive/run_baseline_eval.py`
- **Description**:
  Build CLI evaluation runner filtering suites, bypassing negative samples with Layer 1 gates, and outputting latency and verdict summaries.

### Issue #30: `T022h: Implement Cypher multi-hop candidate discovery query`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/kg/neo4j_store.py`
- **Description**:
  Execute Cypher multi-hop traversal to discover chunk pairs connected via entity bridges for synthetic Evol-Instruct question generation.

### Issue #31: `T022i: Instrument pipeline spans with Langfuse OpenTelemetry context managers`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `parallelizable`, `stage-10`
- **Target File**: `rag-core/src/rag_core/naive/pipeline.py`
- **Description**:
  Wrap `rag_query`, `retrieval`, and `synthesis` spans with Langfuse tracing capturing latency, token counts, and retrieval scores.

### Issue #32: `T023: Implement RAGAS EvaluationHarness`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/harness.py`
- **Description**:
  Implement evaluation harness computing Faithfulness, Answer Relevancy, Context Precision, and Context Recall over batch datasets.

### Issue #33: `T024: Implement Quality Gate threshold validator`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/quality_gate.py`
- **Description**:
  Validate evaluation results against constitutional thresholds (Faithfulness >= 0.80, Relevancy >= 0.75, Precision >= 0.75, Recall >= 0.70).

### Issue #34: `T025: Implement baseline snapshot manager`
- **State**: `open`
- **Labels**: `phase-3-us1`, `stage-10`
- **Target File**: `rag-core/src/rag_core/evaluation/baseline_manager.py`
- **Description**:
  Persist baseline evaluation runs and compute score regressions/deltas across pipeline revisions.

### Issue #35: `T026: Write automated evaluation test suite`
- **State**: `closed` (Completed)
- **Labels**: `phase-3-us1`, `stage-10`, `area:tests`
- **Target File**: `rag-core/tests/test_evaluation.py`
- **Description**:
  Automated unit test suite verifying Layer 1 Code Gate refusal/content interception, Knowledge Graph schema dataclasses, Neo4j connection fallback, LangfuseTracker instantiation, and JudgeVerdict latency metrics. All 6 tests passing.

---

## Phase 4: User Story 2 - Hybrid Search, RRF, Cross-Encoder & Query Transforms

### Issues #36 to #46 (`T027` - `T037`)
- **T027**: Implement BM25 inverted index with identifier-preserving tokenizers (`rag-core/src/rag_core/indexing/bm25_index.py`).
- **T028**: Implement lexical BM25 retriever with field boosts (`rag-core/src/rag_core/retrieval/bm25.py`).
- **T029**: Implement outlier exclusion: Distance thresholding and Autocut (`rag-core/src/rag_core/retrieval/outliers.py`).
- **T030**: Implement Reciprocal Rank Fusion (RRF) (`rag-core/src/rag_core/fusion/rrf.py`).
- **T031**: Implement cross-encoder re-ranking postprocessor (`rag-core/src/rag_core/fusion/reranker.py`).
- **T032**: Implement document deduplication & chunk capper (`rag-core/src/rag_core/fusion/capper.py`).
- **T033**: Implement query transformation techniques (Rewriting, Expansion, HyDE) (`rag-core/src/rag_core/query/`).
- **T034**: Implement sub-question decomposition & filter routing (`rag-core/src/rag_core/query/decomposition.py`).
- **T035**: Implement Parent / Sentence Window expansion (`rag-core/src/rag_core/postprocessing/expansion.py`).
- **T036**: Implement context compression & `LongContextReorder` (`rag-core/src/rag_core/postprocessing/`).
- **T037**: Run evaluation comparing Phase 1 baseline against Phase 2 optimized pipeline.

---

## Phase 5: User Story 3 - Autonomous Multi-Tool Agent & Text-to-SQL

### Issues #47 to #56 (`T038` - `T047`)
- **T038**: Wrap LlamaIndex QueryEngine as a LangChain Tool (`rag-core/src/rag_core/agents/tools/rag_tool.py`).
- **T039**: Implement database schema introspection (`rag-core/src/rag_core/text_to_sql/schema_loader.py`).
- **T040**: Implement AST safety validator for read-only SQL (`rag-core/src/rag_core/text_to_sql/validator.py`).
- **T041**: Implement Text-to-SQL generation chain (`rag-core/src/rag_core/text_to_sql/chain.py`).
- **T042**: Wrap Text-to-SQL as LangChain Tool (`rag-core/src/rag_core/agents/tools/sql_tool.py`).
- **T043**: Implement web search fallback Tool (`rag-core/src/rag_core/agents/tools/web_tool.py`).
- **T044**: Implement GraphRAG entity extraction & Leiden community detection (`rag-core/src/rag_core/indexing/graphrag.py`).
- **T045**: Implement Graph local 1-2 hop neighborhood retriever (`rag-core/src/rag_core/retrieval/graph_local.py`).
- **T046**: Build LangGraph `StateGraph` agent orchestrator (`rag-core/src/rag_core/agents/agent.py`).
- **T047**: Write unit test suite for Text-to-SQL AST validator & agent routing (`rag-core/tests/test_sql_validator.py`).

---

## Phase 6: User Story 4 - Enterprise Security, Performance Caching & UI

### Issues #57 to #69 (`T048` - `T060`)
- **T048**: Implement Microsoft Presidio PII detection and anonymizer (`rag-core/src/rag_core/security/pii_redactor.py`).
- **T049**: Implement Qdrant payload ACL filter (`rag-core/src/rag_core/security/acl_filter.py`).
- **T050**: Implement structured compliance audit logger (`rag-core/src/rag_core/security/audit.py`).
- **T051**: Implement Redis content-hash embedding cache (`rag-core/src/rag_core/cache/embedding_cache.py`).
- **T052**: Implement Redis TTL LLM response cache (`rag-core/src/rag_core/cache/llm_cache.py`).
- **T053**: Configure PostgreSQL async DB & Alembic migrations (`backend/app/db/`, `backend/alembic/`).
- **T054**: Implement FastAPI backend application & routers (`backend/app/main.py`).
- **T055**: Implement streaming `/api/v1/chat` endpoint (`backend/app/api/v1/endpoints/chat.py`).
- **T056**: Implement document management endpoints (`backend/app/api/v1/endpoints/documents.py`).
- **T057**: Implement evaluation API endpoints (`backend/app/api/v1/endpoints/evaluation.py`).
- **T058**: Scaffold Next.js 14+ App Router frontend with CopilotKit (`frontend/src/app/`).
- **T059**: Build Next.js Evaluation Dashboard (`frontend/src/app/dashboard/evaluation/page.tsx`).
- **T060**: Build manifest upload and pipeline settings pages (`frontend/src/app/documents/`).

---

## Phase 7: Polish & CI/CD Workflows

### Issues #70 to #75 (`T061` - `T065`)
- **T061**: Create GitHub Actions evaluation workflow (`.github/workflows/rag-eval.yml`).
- **T062**: Create scheduled drift detection workflow (`.github/workflows/scheduled-eval.yml`).
- **T063**: Create multi-stage production Dockerfiles (`backend/Dockerfile`, `frontend/Dockerfile`).
- **T064**: Write comprehensive project `README.md`.
- **T065**: Run end-to-end verification of all quality criteria (`SC-001` through `SC-015`).

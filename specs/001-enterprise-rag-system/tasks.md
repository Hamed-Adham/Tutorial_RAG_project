# Implementation Tasks: Enterprise RAG System

**Branch**: `001-enterprise-rag-system` | **Spec**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/spec.md) | **Plan**: [plan.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/plan.md)

---

## Phase 1: Setup & Shared Infrastructure (Phase 0)

**Purpose**: Project initialization, directory tree scaffolding for all 10 stages, packaging, and local developer environment.

- [x] T001 Initialize directory tree for `rag-core/src/rag_core/` (modules: `manifest`, `parsing`, `chunking`, `indexing`, `query`, `retrieval`, `fusion`, `postprocessing`, `generation`, `agents`, `evaluation`, `security`, `cache`, `observability`), `rag-core/tests/`, `backend/`, and `frontend/`.
- [x] T002 [P] Create `rag-core/pyproject.toml` with LlamaIndex, LangChain, LangGraph, BGE-M3, Qdrant, RAGAS, Presidio, Langfuse, Hazm, and Docling/PyMuPDF dependencies.
- [x] T003 [P] Create `backend/pyproject.toml` and `requirements.txt` with FastAPI, Uvicorn, SQLAlchemy, Alembic, and Pydantic v2.
- [x] T004 [P] Create root `.env.example` defining Qdrant, PostgreSQL, Redis, Langfuse, and LLM API keys.
- [x] T005 [P] Create `docker-compose.yml` for local services (Qdrant, PostgreSQL, Redis, Langfuse, Neo4j).
- [x] T006 [P] Create root `Makefile` with targets: `make dev`, `make test`, `make eval`, and `make build`.
- [x] T006b [P] Add `--gates-only` flag to `run_baseline_eval.py` and configure `make test-gates` and `make check` in root `Makefile` for sub-2s zero-token deterministic gate verification.

---

## Phase 2: Foundational Architecture (Blocking Prerequisites)

**Purpose**: Core configurations, logging, vector client, graph store, and tracing connectors that MUST be complete before user stories can execute.

- [x] T007 Implement central configuration model in `rag-core/src/rag_core/config.py` using Pydantic Settings.
- [x] T008 [P] Configure structured JSON logging with `structlog` in `rag-core/src/rag_core/observability/logging.py`.
- [x] T009 [P] Implement Langfuse tracing client and OpenInference callback handlers in `rag-core/src/rag_core/observability/langfuse.py`.
- [x] T010 [P] Implement Qdrant vector database connection and HNSW collection initialization in `rag-core/src/rag_core/indexing/vector_store.py`.
- [x] T010b [P] Implement Neo4j Knowledge Graph Store driver with schema constraints and offline fallback in `rag-core/src/rag_core/kg/neo4j_store.py`.
- [x] T010c [P] Implement Knowledge Graph schema models (`DocumentNode`, `ChunkNode`, `EntityNode`, `KeyphraseNode`, `MultiHopCandidate`) in `rag-core/src/rag_core/kg/schema.py`.
- [x] T011 [P] Implement BGE-M3 embedding loader (`HuggingFaceEmbedding` with int8 quantization support and query prefix) in `rag-core/src/rag_core/indexing/embeddings.py`.
- [x] T011b [P] Refresh Codebase Knowledge Graph via `codebase-memory-mcp` `index_repository(persistence=True)` and verify zero drift (Constitution Principle X).

**Checkpoint**: Foundation ready — vector store, knowledge graph store, embedding model, tracing, and logging are operable. Codebase Knowledge Graph is refreshed and persistent.

---

## Phase 3: User Story 1 - Core 10-Stage MVP Pipeline & Swiss Cheese Evaluation (Priority: P1) 🎯 MVP

**Goal**: Deliver a functional, verifiable RAG pipeline featuring manifest deduplication, canonical IR parsing, breadcrumb chunking (~512 tokens), dense vector retrieval, grounded response synthesis with page citations, Tri-Layered "Swiss Cheese" evaluation, Neo4j multi-hop synthetic dataset discovery, and baseline RAGAS scores.

**Independent Test**: Load documents into the manifest catalog, verify exact SHA-256 deduplication, chunk with breadcrumbs, retrieve via Qdrant, generate an answer with page citations, execute Layer 1 code gates (< 5ms, 0 tokens) returning standardized refusal payload, evaluate via Layer 2 judge with latency reporting, and run `run_baseline_eval.py` against `golden_set.jsonl`.

### Implementation Tasks for User Story 1

- [x] T012 [P] [US1] Implement PostgreSQL document catalog with stable `doc_id`, SHA-256 hash, tenant, ACL tracking, and synchronous cascade hard-deletion interface in `rag-core/src/rag_core/manifest/catalog.py` (Stage 1).
- [x] T013 [P] [US1] Implement exact SHA-256 duplicate collapse and MinHash near-duplicate clustering in `rag-core/src/rag_core/manifest/deduplicator.py` (Stage 1).
- [x] T014 [US1] Implement tiered parsing probe router and Tier 0 layout-aware extractor (PyMuPDF/Docling) in `rag-core/src/rag_core/parsing/router.py` and `tier0_layout.py` (Stage 2).
- [x] T015 [US1] Implement Canonical IR block schema (`type`, `text`, `page`, `bbox`) in `rag-core/src/rag_core/parsing/schema.py` (Stage 2).
- [x] T016 [US1] Implement structure-aware hierarchical chunking on headings with breadcrumb prepending (`Doc > Section > Subsection (page)`), targeting ~512 tokens per atomic chunk and parent aggregation, in `rag-core/src/rag_core/chunking/hierarchical.py` (Stage 3).
- [x] T017 [US1] Implement Parent-Child chunking (indexing small child chunks, storing parent link) in `rag-core/src/rag_core/chunking/parent_child.py` (Stage 3).
- [x] T018 [US1] Implement unified multi-index indexer orchestrating manifest -> parsing -> chunking -> embedding -> Qdrant in `rag-core/src/rag_core/indexing/indexer.py` (Stage 4).
- [x] T019 [US1] Implement core dense HNSW vector retriever with model query prefix in `rag-core/src/rag_core/retrieval/dense.py` (Stage 6).
- [x] T020 [US1] Implement page and bounding box (`bbox`) source citation formatter in `rag-core/src/rag_core/postprocessing/citations.py` (Stage 8).
- [x] T021 [US1] Implement grounded LLM response synthesizer (supporting GPT-4o and local Llama-3.1) with "I don't know" fallback in `rag-core/src/rag_core/generation/synthesizer.py` and `prompts.py` (Stage 9).
- [x] T022 [P] [US1] Create curated Golden Benchmark Dataset (100 Q&A pairs stratified 75 dev / 25 test across 4 suites: Science Textbook, Administrative FAQ, Domain Routing, Adversarial Refusal) in `rag-core/src/rag_core/evaluation/datasets/golden_set.jsonl` and `load_dataset.py` (Stage 10).
- [x] T022b [P] [US1] Implement 5-Part Judge Rubrics (Persona, Binary Criteria, XML Data Boundaries, Few-Shot Persian Examples, Mandatory Step-by-Step CoT) for Faithfulness, Relevancy, Routing, and Adversarial Refusal in `rag-core/src/rag_core/evaluation/rubrics.py` (Stage 10).
- [x] T022c [P] [US1] Implement Meta-Evaluation Confusion Matrix (TP, FP, TN, FN, Precision, Recall, Hallucination Catch Rate) and Pass^k / Pass@k Production Reliability Scoring in `rag-core/src/rag_core/evaluation/meta_eval.py` (Stage 10).
- [x] T022d [P] [US1] Implement Layer 1 Deterministic Code Gates (< 5ms, 0 tokens) with standardized Persian refusal messages and structured metadata (`is_refusal: true`, `refusal_reason`) in `rag-core/src/rag_core/evaluation/code_gates.py` (Stage 10).
- [x] T022e [P] [US1] Implement high-resolution execution duration tracking (`latency_seconds`) in LLM Judge Evaluator in `rag-core/src/rag_core/naive/judge_evaluator.py` (Stage 10).
- [x] T022f [P] [US1] Implement local Ragas adapter (`LocalRagasEvaluator`) binding Faithfulness, Answer Relevancy, Context Precision, and Context Recall to local Nine Router gateway (`http://localhost:20128/v1`) in `rag-core/src/rag_core/evaluation/ragas_adapter.py` (Stage 10).
- [x] T022g [US1] Implement benchmark evaluation runner with Layer 1 gate bypass and judge latency reporting in `rag-core/src/rag_core/naive/run_baseline_eval.py` (Stage 10).
- [x] T022h [P] [US1] Implement Cypher multi-hop candidate discovery query (`find_multi_hop_candidates()`) for synthetic testset generation in `rag-core/src/rag_core/kg/neo4j_store.py` (Stage 10).
- [x] T022i [P] [US1] Instrument pipeline spans (`rag_query`, `retrieval`, `synthesis`) with Langfuse OpenTelemetry context managers in `rag-core/src/rag_core/naive/pipeline.py` (Stage 10).
- [x] T023 [US1] Implement RAGAS `EvaluationHarness` computing Faithfulness, Answer Relevancy, Context Precision, and Context Recall in `rag-core/src/rag_core/evaluation/harness.py` (Stage 10).
- [x] T024 [US1] Implement Quality Gate threshold validator (Faithfulness >= 0.80, Relevancy >= 0.75, Precision >= 0.75, Recall >= 0.70) in `rag-core/src/rag_core/evaluation/quality_gate.py` (Stage 10).
- [x] T025 [US1] Implement CLI evaluation entrypoint `run_eval.py` and baseline snapshot manager in `rag-core/src/rag_core/evaluation/baseline_manager.py` (Stage 10).
- [x] T026 [US1] Write automated test suite for manifest cataloging, parsing, chunking, and evaluation in `rag-core/tests/test_evaluation.py`.
- [x] T066 [P] [US1] Write automated unit and integration tests for manifest catalog, deduplicator, tiered parser, hierarchical/parent-child chunkers, indexer, citations, synthesizer, and evaluation harness in `rag-core/tests/test_pipeline_stages.py`.
- [x] T067 [US1] Document Milestone 11: Modular 10-Stage Pipeline Implementation & Verification in root `development_log.md` per Constitution VIII.
- [x] T068 [P] [US1] Extract centralized prompt templates into dedicated module `rag-core/src/rag_core/generation/prompts.py` per Stage 9 and FR-023.
- [x] T069 [P] [US1] Implement document lineage, version tracking, and tombstone cascade deletion in `rag-core/src/rag_core/manifest/lineage.py` per Stage 1 and FR-003.
- [x] T070 [P] [US1] Implement dedicated Tier 1 OCR extractor (`tier1_ocr.py`) and Tier 2 VLM extractor (`tier2_vlm.py`) in `rag-core/src/rag_core/parsing/` per Stage 2 and FR-005.
- [x] T072 [US1] Add `make eval-modular` target to root `Makefile` invoking `rag_core.evaluation.run_eval` with quality gate verification per T006, T025, and FR-027.
- [x] T073 [US1] Refresh Codebase Knowledge Graph (`.codebase-memory/graph.db.zst`) via `index_repository(persistence=True)` and verify 0 unindexed files via `detect_changes` (Constitution Principle X).

**Checkpoint**: MVP is complete. Can catalog documents, deduplicate, parse to IR, chunk with breadcrumbs, query with citations, trace via Langfuse, intercept negative queries with Layer 1 code gates in < 5ms, track judge latencies, assert passing RAGAS evaluation scores, and synchronize persistent codebase knowledge graph.

---

## Phase 4: User Story 2 - Retrieval Optimization via Component Maximization (Priority: P2)

**Goal**: Maximize existing dense retrieval, chunking, and prompt engineering components to their absolute performance ceiling before introducing any new architectural component, model layer, or pipeline stage (Constitution Principle VII). Only introduce BM25, cross-encoder rerankers, or query transformations where empirical error analysis on benchmark failure modes justifies the added latency and complexity.

**Independent Test**: Benchmark existing dense pipeline across varied chunk sizes and HNSW `ef_search` parameters; perform error categorization on remaining misses; evaluate comparative metric deltas after adding each targeted component.

### Sub-Phase 4A: Maximize Current Baseline Components (Zero New Dependencies)

- [ ] T027a [P] [US2] Benchmark and tune structure-aware chunk sizing parameters (512 vs 256 vs 1024 tokens) and heading breadcrumb formats in `rag-core/src/rag_core/chunking/hierarchical.py` against `golden_set.jsonl`.
- [ ] T027b [P] [US2] Optimize Qdrant dense vector search parameters (`ef_search` beam width, `top_k` candidate multipliers) in `rag-core/src/rag_core/retrieval/dense.py`.
- [ ] T027c [P] [US2] Refine grounded synthesis prompts with Persian few-shot demonstrations and strict anti-hallucination instructions in `rag-core/src/rag_core/generation/prompts.py`.
- [ ] T027d [US2] Execute full 100-item benchmark evaluation on the maximized baseline, record metrics in `development_log.md`, and generate a categorized error log (Lexical Mismatch, Chunk Boundary Cutoff, Complex Query, or Out of Scope).
- [x] T027e [P] [US2] Implement TypeSafe Jev System One client wrapper in `rag-core/src/rag_core/evaluation/jev_client.py` with HTTP timeout, retry handling, and automatic fallback to Nine Router per FR-039.
- [x] T027f [P] [US2] Implement `JevJudge` in `rag-core/src/rag_core/evaluation/jev_judge.py` with single-request speculative fan-out evaluating Faithfulness (Noul), 3-level Answer Relevancy (Score), and Adversarial Refusal (Noul) per FR-039 and SC-016.
- [x] T027g [P] [US2] Implement `JevRouter` in `rag-core/src/rag_core/query/jev_router.py` using `Choice` primitive with confidence thresholding and automatic local LLM fallback per FR-039.
- [x] T027h [US2] Integrate `JevJudge` as the default evaluation backend in `rag-core/src/rag_core/naive/run_baseline_eval.py` and `Makefile` (`make eval` defaulting to Jev, `make eval-llm` for local generative judge).
- [x] T027i [P] [US2] Implement `JevReranker` in `rag-core/src/rag_core/fusion/jev_reranker.py` and single-call `route_and_guard` in `rag-core/src/rag_core/query/jev_router.py`, minimizing Jev calls via parallel speculative fan-out per FR-039.
- [x] T027j [P] [US2] Implement universal LangGraph `StateGraph` RAG pipeline (`RAGGraphState`, nodes, conditional edges) in `rag-core/src/rag_core/graph/` refactoring baseline execution to LangChain/LangGraph standard per FR-040.
- [x] T027k [P] [US2] Implement mandatory Langfuse tracing for LangGraph via native `CallbackHandler` with granular per-span latency and token/cost attribution per FR-041 and SC-017.
- [x] T027l [P] [US2] Implement dynamic trace tagging (`baseline`, `improved`, dataset split, suite name) across LangGraph and evaluation harnesses per FR-041 and SC-017.
- [x] T027m [P] [US2] Migrate Golden Benchmark to Langfuse Datasets (`rag-core/src/rag_core/evaluation/langfuse_dataset.py`) supporting dataset sync and experiment runs per FR-041 and SC-018.
- [x] T027n [US2] Author Langfuse Project Manual & Cheat Sheet in `docs/langfuse_manual.md` documenting setup, LangGraph tracing, latency/cost analysis, and dataset benchmarks per FR-041.



### Sub-Phase 4B: Targeted Component Addition (Empirically Justified by 4A Error Log)

- [ ] T028 [P] [US2] Implement BM25 inverted index with `hazm` / `parsivar` Persian NLP normalization, stemming, and lemmatization in `rag-core/src/rag_core/indexing/bm25_index.py` (Stage 4).
- [ ] T029 [P] [US2] Implement lexical BM25 retriever with field boosts (title^3, breadcrumb^2) in `rag-core/src/rag_core/retrieval/bm25.py` (Stage 6).
- [ ] T030 [US2] Implement Reciprocal Rank Fusion (RRF: `score = sum(1 / (60 + rank))`) and Autocut outlier exclusion in `rag-core/src/rag_core/fusion/rrf.py` and `rag-core/src/rag_core/retrieval/outliers.py` (Stage 7).
- [ ] T031 [US2] Implement cross-encoder re-ranking postprocessor using `BAAI/bge-reranker-v2-m3` in `rag-core/src/rag_core/fusion/reranker.py` (Stage 7).
- [ ] T032 [US2] Implement query transformation techniques (Query Rewriting, HyDE, Sub-question decomposition) in `rag-core/src/rag_core/query/rewriting.py` and `decomposition.py` (Stage 5).
- [ ] T033 [US2] Implement parent chunk expansion with dynamic parent aggregation over N child chunks in `rag-core/src/rag_core/postprocessing/expansion.py` (Stage 8).
- [ ] T034 [US2] Run comparative evaluation harness against the maximized baseline and record the empirical delta in `development_log.md` per Constitution VII.

**Checkpoint**: Retrieval optimization complete. All added components are justified by proven empirical delta over the maximized baseline.

---

## Phase 5: User Story 3 - Autonomous Multi-Tool Agent with Document RAG, GraphRAG & Web Search (Priority: P3)

**Goal**: Build a LangGraph StateGraph agent that autonomously routes queries between unstructured Document RAG, GraphRAG multi-hop entity graphs, and external Web search (with Text-to-SQL explicitly descoped).

**Independent Test**: Submit document questions (Document RAG), multi-hop entity questions (GraphRAG), and external/current information questions (Web search), confirming proper tool routing, multi-source synthesis, and state checkpointing.

### Implementation Tasks for User Story 3

- [ ] T038 [P] [US3] Wrap LlamaIndex QueryEngine as a LangChain Tool in `rag-core/src/rag_core/agents/tools/rag_tool.py` (Stage 9).
- [ ] T039 [P] [US3] Implement GraphRAG entity and typed relationship extraction with Leiden community detection in `rag-core/src/rag_core/indexing/graphrag.py` (Stage 4).
- [ ] T040 [P] [US3] Implement Graph local 1-2 hop neighborhood retriever in `rag-core/src/rag_core/retrieval/graph_local.py` (Stage 6).
- [ ] T041 [P] [US3] Wrap GraphRAG local retriever as a LangChain Tool in `rag-core/src/rag_core/agents/tools/graph_tool.py` (Stage 9).
- [ ] T042 [P] [US3] Implement web search fallback Tool in `rag-core/src/rag_core/agents/tools/web_tool.py` (Stage 9).
- [ ] T043 [US3] Build LangGraph `StateGraph` agent orchestrator coordinating Document RAG, GraphRAG, and Web Search with conditional decision branches and checkpointing in `rag-core/src/rag_core/agents/agent.py` (Stage 9).
- [ ] T044 [US3] Write unit and integration test suite for GraphRAG tools and LangGraph multi-tool agent routing in `rag-core/tests/test_agent.py`.

**Checkpoint**: Multi-tool agent operational. Successfully coordinates Document RAG, GraphRAG, and Web Search.

---

## Phase 6: User Story 4 - Enterprise Security, Performance Caching, FastAPI & Full-Stack UI (Priority: P4)

**Goal**: Harden the solution with Microsoft Presidio PII redaction, Qdrant payload ACL filtering, Redis caching, FastAPI REST streaming server, and a Next.js CopilotKit UI with evaluation dashboards.

**Independent Test**: Ingest a document containing PII (verify redaction), query with restricted user role (verify ACL filtering), repeat identical query (verify Redis cache hit), delete document (verify immediate cascade hard-deletion across Postgres, Qdrant, BM25, and Neo4j), and interact with the Next.js chat interface and evaluation metrics dashboard.

### Implementation Tasks for User Story 4

- [ ] T048 [P] [US4] Implement Microsoft Presidio PII detection and anonymizer in `rag-core/src/rag_core/security/pii_redactor.py` (Stage 10).
- [ ] T049 [P] [US4] Implement Qdrant payload-based Access Control List (ACL) filter in `rag-core/src/rag_core/security/acl_filter.py` (Stage 10).
- [ ] T050 [P] [US4] Implement structured compliance audit logger in `rag-core/src/rag_core/security/audit.py` (Stage 10).
- [ ] T051 [P] [US4] Implement Redis content-hash embedding cache in `rag-core/src/rag_core/cache/embedding_cache.py` (Stage 10).
- [ ] T052 [P] [US4] Implement Redis TTL query/context response cache in `rag-core/src/rag_core/cache/llm_cache.py` (Stage 10).
- [ ] T053 [US4] Configure PostgreSQL async database engine and Alembic migrations for manifest catalog (`manifest_documents`), conversations, and evaluation runs in `backend/app/db/` and `backend/alembic/`.
- [ ] T054 [US4] Implement FastAPI backend application, CORS middleware, and routers adhering to OpenAPI contract in `backend/app/main.py` and `backend/app/api/v1/router.py`.
- [ ] T055 [US4] Implement streaming `/api/v1/chat` endpoint and RAG bridge service in `backend/app/api/v1/endpoints/chat.py` and `backend/app/services/rag_service.py`.
- [ ] T056 [US4] Implement document management endpoints (upload, manifest list, and immediate synchronous cascade hard-delete `DELETE /api/v1/documents/{doc_id}`) in `backend/app/api/v1/endpoints/documents.py`.
- [ ] T057 [US4] Implement evaluation API endpoints (`/run`, `/results`, `/baseline`) in `backend/app/api/v1/endpoints/evaluation.py`.
- [ ] T058 [US4] Scaffold Next.js 14+ App Router frontend with CopilotKit provider in `frontend/src/app/layout.tsx` and `frontend/src/app/chat/page.tsx`.
- [ ] T059 [US4] Build Next.js Evaluation Dashboard components (`MetricsChart`, `EvalRunTable`, `QualityGateStatus`) in `frontend/src/app/dashboard/evaluation/page.tsx`.
- [ ] T060 [US4] Build document manifest upload and pipeline settings pages in `frontend/src/app/documents/page.tsx` and `frontend/src/app/settings/page.tsx`.

**Checkpoint**: Full-stack application ready with enterprise security, low-latency caching, backend API with cascade hard-deletion, and CopilotKit UI.

---

## Phase 7: Polish & CI/CD Workflows

**Purpose**: Continuous evaluation automation, containerization, and final verification.

- [ ] T061 [P] Create GitHub Actions workflow `.github/workflows/rag-eval.yml` to trigger RAGAS evaluation on PRs affecting RAG files.
- [ ] T062 [P] Create scheduled GitHub Actions workflow `.github/workflows/scheduled-eval.yml` for weekly drift detection.
- [ ] T063 [P] Create multi-stage production Dockerfiles for `backend/Dockerfile` and `frontend/Dockerfile`.
- [ ] T064 Write comprehensive project `README.md` documenting the 10-stage architecture, setup commands, evaluation workflow, and API reference.
- [ ] T065 Run end-to-end verification of all quality criteria (SC-001 through SC-015, excluding descoped SC-006).
- [ ] T065b Refresh Codebase Knowledge Graph (`.codebase-memory/graph.db.zst`) via `index_repository(persistence=True)` and execute Codebase Memory Graph Gate (verify zero drift via `detect_changes`, Constitution Principle X).

---

## Dependencies & Execution Sequence

```text
Phase 1: Setup ───► Phase 2: Foundational ───► Phase 3: US1 (MVP RAG + Eval)
                                                       │
                                                       ├───► Phase 4: US2 (Optimization)
                                                       │
                                                       ├───► Phase 5: US3 (Agent + GraphRAG)
                                                       │
                                                       └───► Phase 6: US4 (Enterprise & UI)
                                                                    │
                                                                    ▼
                                                             Phase 7: CI/CD & Polish
```

### Phase Dependencies
- **Setup (Phase 1)**: No dependencies — completed.
- **Foundational (Phase 2)**: Completed — core storage and model loading operational.
- **User Story 1 (Phase 3)**: Completed — end-to-end MVP, Swiss Cheese evaluation, Neo4j multi-hop discovery, and modular pipeline verified.
- **User Story 2 (Phase 4)**: Ready for execution — depends on US1. Adds BM25 with Persian NLP, RRF, cross-encoder, and query transformations.
- **User Story 3 (Phase 5)**: Depends on US1; integrates GraphRAG and Web search into LangGraph StateGraph (NL2SQL descoped).
- **User Story 4 (Phase 6)**: Depends on US1/US2/US3; implements Presidio PII, Qdrant ACL, Redis caching, FastAPI service with cascade deletion, and Next.js UI.
- **Polish (Phase 7)**: Depends on desired user stories being complete.

---

## Parallel Execution Opportunities

- In **Phase 4 (US2)**: T027 (BM25 inverted index) and T028 (BM25 retriever) can be implemented in parallel with T033 (Query Rewriting/Expansion/HyDE).
- In **Phase 5 (US3)**: T038 (RAG tool wrapper), T039 (GraphRAG extraction), and T042 (Web search tool) can be implemented in parallel before T043 (StateGraph orchestrator).
- In **Phase 6 (US4)**: Security tasks (T048, T049, T050) and Caching tasks (T051, T052) can be implemented in parallel. Frontend pages (T058, T059, T060) can be developed in parallel against `backend-api.yaml` OpenAPI contract.

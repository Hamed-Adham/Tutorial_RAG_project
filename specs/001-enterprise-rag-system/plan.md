# Implementation Plan: Enterprise RAG System

**Branch**: `001-enterprise-rag-system` | **Date**: 2026-10-05 (Updated with Clarified 10-Stage Architecture) | **Spec**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/spec.md)

**Input**: Feature specification synthesizing `PROJECT_DOCUMENTATION (1).pdf`, 100k-document production explainer (Ardya Dipta), Advanced RAG Techniques, and ratified clarifications (Session 2026-10-05).

---

## Summary

Build an enterprise-grade Retrieval-Augmented Generation (RAG) system organized into a **Unified 10-Stage Pipeline Architecture** adhering to a 3-layer decoupled design (with Text-to-SQL descoped and deferred):
1. `rag-core`: Framework-independent, modular Python package encapsulating all 10 stages:
   - **Stage 1**: Ingestion & Manifest Catalog (Postgres catalog, SHA-256 hash collapse, MinHash near-duplicate clustering, immediate synchronous cascade hard-deletion across Qdrant, BM25, and Neo4j).
   - **Stage 2**: Tiered Parsing into Canonical IR (Tier 0 text-native layout, Tier 1 OCR confidences, Tier 2 VLM captions, Canonical IR blocks with `type`, `text`, `page`, `bbox`).
   - **Stage 3**: Chunking Strategies (Structure-aware headings with breadcrumbs, ~512-token atomic chunks, cohesive semantic sections up to 1024 tokens, dynamic parent aggregation over N child chunks).
   - **Stage 4**: Enrichment, Embedding & Multi-Index Views (Rebuildable views: Qdrant HNSW vector store with int8 BGE-M3, BM25 inverted index, GraphRAG entity/community graph).
   - **Stage 5**: Pre-Retrieval & Query Understanding (Query rewriting, query expansion, HyDE, sub-question decomposition, closed-schema filter extraction, query routing).
   - **Stage 6**: Multi-Pronged Retrieval & Outlier Handling (Session ACL pre-filtering, BM25 with Hazm/Parsivar Persian NLP normalization & identifier tokenization, Dense HNSW with query prefix, Graph local expansion, Autocut & distance thresholding).
   - **Stage 7**: Fusion & Post-Retrieval Re-ranking (Reciprocal Rank Fusion [RRF], cross-encoder re-ranking [BGE-Reranker v2 M3 / Zerank 1 Small], document chunk capping).
   - **Stage 8**: Context Post-Processing & Refinement (Parent / Sentence Window expansion, contextual compression, `LongContextReorder`, citation assembly with `bbox`).
   - **Stage 9**: Generation, Prompt Engineering & Agent Orchestration (Grounded prompts, CoT/ToT, LangGraph StateGraph agent coordinating Document RAG, GraphRAG, and Web search; Text-to-SQL explicitly descoped).
   - **Stage 10**: Evaluation, Observability & Guardrails:
     - **Option 10A**: Tri-Layered "Swiss Cheese" Evaluation: Layer 1 Deterministic Code Gates (< 5ms, 0 tokens) with standardized Persian refusal messages and structured metadata (`is_refusal: true`); Layer 2 LLM-as-a-Judge with 5-part CoT rubrics, execution latency tracking (`latency_seconds`), and local gateway-backed Ragas metrics (Faithfulness >= 0.80, Relevancy >= 0.75, Precision >= 0.75, Recall >= 0.70); Layer 3 Human Meta-Evaluation (confusion matrix, Hallucination Catch Rate).
     - **Option 10B**: Langfuse OpenTelemetry tracing (spans for `rag_query`, `retrieval`, `synthesis`, latency profiling, tokens, costs).
     - **Option 10C**: Neo4j Knowledge Graph Store (`rag_core.kg.neo4j_store`) for document-chunk-entity hierarchies, relationship extraction, and Cypher multi-hop candidate discovery for synthetic testset generation (Evol-Instruct).
     - **Option 10D**: Enterprise security guardrails (Microsoft Presidio PII redaction, structured audit logging).
     - **Option 10E**: Performance caching (Redis content-hash embedding cache, TTL LLM response cache).
2. `backend`: FastAPI API server providing typed REST endpoints (`/api/v1`), streaming chat, session persistence, document management, and evaluation triggers with Alembic database migrations.
3. `frontend`: Next.js 14+ App Router interface featuring CopilotKit interactive chat, document management views, and real-time RAGAS evaluation dashboards.

---

## Technical Context

- **Languages & Runtimes**: Python 3.11+, TypeScript, Node.js 18+
- **Core Frameworks**: LangChain & LangGraph (mandatory universal pipeline standard; compiled `StateGraph` for both baseline and multi-tool agent pipelines), LlamaIndex (ingestion & indexing utilities)
- **Vector, Graph & Storage**:
  - Qdrant (HNSW vector store with payload ACL filtering; migration path to Milvus)
  - Neo4j 5.x / 6.x (`neo4j` Python driver 6.3.1, Bolt port 7687, HTTP port 7474; document-chunk-entity knowledge graph for multi-hop synthesis)
  - PostgreSQL (document manifest catalog, evaluation history, metadata)
  - Redis (embedding & LLM response caching)
- **Embedding & Re-ranking**: `BAAI/bge-m3` (dense multilingual embeddings, int8 quantization), `BAAI/bge-reranker-v2-m3` / `Zerank 1 Small` (cross-encoder re-ranking)
- **Lexical Search & Persian NLP**: BM25 inverted index with `hazm` / `parsivar` Persian text normalization, stemming, and lemmatization, plus identifier-preserving tokenizers (retaining codes like `ERR-4092`)
- **LLM Providers & Gateway**: Local OpenAI-compatible gateway (Nine Router on `http://localhost:20128/v1`), GPT-4o (production reasoning), Llama-3.1-8B via Ollama / Xinference / vLLM (local private execution), TypeSafe Jev (System One AI for sub-300ms typed decisions)
- **Evaluation & Full Langfuse Ecosystem**:
  - **Mandatory Langfuse Tracing**: Active by default across all workflow runs and evaluations with zero-overhead fallback. Native `CallbackHandler` on LangGraph compiles execution spans with granular per-node latency and token/cost attribution.
  - **Dynamic Trace Tags**: Strict tagging taxonomy (`baseline`, `improved`, `dev`, `test`, `smoke`, suite names).
  - **Langfuse Datasets Migration**: Golden Benchmark synced to Langfuse Dataset (`unified-rag-golden-benchmark`), logging evaluation runs directly as scored dataset experiments.
  - **Tier 1 Code Gates**: Zero-token heuristic checks, regex refusal, domain lexical matching, standardized refusal payload.
  - **Tier 2.5 Jev as Judge**: Calibrated single-call fan-out for Faithfulness, 3-level Relevancy, Context Coverage, and Refusal.
  - **Tier 2 LLM Judge & RAGAS**: 5-part CoT rubrics, execution latency tracking (`latency_seconds`), and local gateway-backed Ragas metrics.
  - `structlog` (structured JSON logging)

- **Security & Privacy**: Microsoft Presidio (PII redaction & anonymization), Qdrant Payload Filters (retrieval-level ACL)
- **API & Web Interface**: FastAPI, Uvicorn, Pydantic v2, Next.js 14+ (App Router), Tailwind CSS / Vanilla CSS, CopilotKit
- **Database Migrations**: Alembic for PostgreSQL schema lifecycle

---

## Constitution Check

- [x] **Principle I: Three-Layer Decoupled Architecture**: `rag-core` has zero dependencies on FastAPI or Next.js. `backend` imports `rag-core`. `frontend` connects strictly via HTTP/WebSocket API contracts.
- [x] **Principle II: Evaluation-Driven Quality**: RAGAS Evaluation Harness and Golden Test Set are established in Phase 1 before optimization. Quality gates block PRs that drop below thresholds.
- [x] **Principle III: Granular Observability & Open Inference Standards (Constitution v1.2.0)**: Langfuse OpenTelemetry spans wrap all discrete pipeline stages (parse, chunk, embed, retrieve, rerank, synthesize) with user/session/release tags, token & cost tracking, latency_seconds, and trace-level evaluation score ingestion. Fallbacks guarantee zero-crash execution when offline.
- [x] **Principle IV: Independent Multi-Tool Agent**: StateGraph multi-tool agent cleanly decouples tools (Document RAG, GraphRAG, Web search). Text-to-SQL is explicitly descoped for this milestone to focus on document/graph RAG.
- [x] **Principle V: Defense-in-Depth Security**: Presidio PII redaction and Qdrant payload ACL filtering are enforced at retrieval time.
- [x] **Principle VI: Performance Caching**: Redis content-hash embedding and TTL LLM response caching are integrated.
- [x] **Principle VII: Law of Parsimony & Baseline-First Engineering**: Naive baseline implemented, Layer 1 deterministic gates (< 5ms, 0 tokens) bypass expensive LLM calls, benchmarked on 100-item golden test set to set Ground Zero.
- [x] **Principle VIII: Continuous Development Logging**: Progress, architectural choices, and evaluation metrics continuously logged in `development_log.md`.
- [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance (Constitution v1.4.0)**: The codebase dependency and symbol graph is continuously maintained as a persistent Knowledge Graph (`.codebase-memory/graph.db.zst`) via `codebase-memory-mcp`. Pre-planning queries the KG to ground decisions; post-implementation hooks (`.specify/extensions.yml`) and phase checkpoints trigger `index_repository(persistence=True)` to refresh the index. All milestones require zero drift (`changed_count: 0`). The `evaluation/` directory is the canonical path for all eval logic; `development_log.md` is updated at every structural change, and `make check` must pass.


---


## Detailed Directory Structure (Mapped to the 10 Stages)

```text
Unified_RAG/
├── .agents/skills/                    # Spec Kit & Antigravity skills
├── .specify/                          # Spec Kit memory, templates, scripts
├── specs/001-enterprise-rag-system/   # Feature specification & implementation plan
│   ├── spec.md                        # Feature specification
│   ├── plan.md                        # Implementation plan (this file)
│   ├── tasks.md                       # Phase-by-phase actionable tasks
│   └── checklists/
│       └── requirements.md            # Spec quality checklist
│
├── rag-core/                          # 🧠 Independent, Modular RAG Package
│   ├── src/
│   │   └── rag_core/
│   │       ├── __init__.py
│   │       ├── config.py              # Central Pydantic pipeline settings
│   │       │
│   │       ├── manifest/              # 📋 Stage 1: Manifest & Deduplication
│   │       │   ├── catalog.py         # Postgres document catalog (stable doc_id, SHA-256, ACL, tenant)
│   │       │   ├── deduplicator.py    # Exact SHA-256 collapse & MinHash near-duplicate clustering
│   │       │   └── lineage.py         # Version tracking, tombstones & incremental re-processing
│   │       │
│   │       ├── parsing/               # 📄 Stage 2: Tiered Parsing into Canonical IR
│   │       │   ├── router.py          # Probe & route (text-native vs scan vs image)
│   │       │   ├── tier0_layout.py    # Docling / PyMuPDF layout-aware extractor (reading order, tables)
│   │       │   ├── tier1_ocr.py       # Textract / PaddleOCR with confidence scores
│   │       │   ├── tier2_vlm.py       # Vision-language model structured captions for figures/tables
│   │       │   └── schema.py          # Canonical IR: Block (type, text, page, bbox)
│   │       │
│   │       ├── chunking/              # 🧩 Stage 3: Structure-Aware Chunking Strategies
│   │       │   ├── hierarchical.py    # Heading hierarchy splitter with breadcrumbs (Doc > Sec > Subsec)
│   │       │   ├── parent_child.py    # Parent-child chunking (small retrieval nodes -> parent doc)
│   │       │   ├── semantic.py        # Semantic cosine distance shift splitter
│   │       │   └── contextual.py      # Anthropic-style 1-2 sentence contextual prefix generator
│   │       │
│   │       ├── indexing/              # 📇 Stage 4: Multi-Index Derived Views & Embedding
│   │       │   ├── embeddings.py      # BGE-M3 loader (with int8 quantization & query prefix)
│   │       │   ├── vector_store.py    # Qdrant HNSW vector store with payload ACL filtering
│   │       │   ├── bm25_index.py      # Inverted index with identifier-preserving tokenizers
│   │       │   ├── graphrag.py        # Entity/relationship extraction & Leiden community detection
│   │       │   └── indexer.py         # Unified multi-index build orchestrator from Canonical IR
│   │       │
│   │       ├── query/                 # 🔍 Stage 5: Pre-Retrieval & Query Understanding
│   │       │   ├── rewriting.py       # Rewrite-Retrieve-Read conversation query normalizer
│   │       │   ├── expansion.py       # Multi-query expansion generator
│   │       │   ├── hyde.py            # Hypothetical Document Embeddings generator
│   │       │   ├── decomposition.py   # Sub-question decomposition for multi-part queries
│   │       │   └── routing.py         # Closed-schema filter extraction & intent classification router
│   │       │
│   │       ├── retrieval/             # 📊 Stage 6: Multi-Pronged Retrieval & Outlier Handling
│   │       │   ├── dense.py           # HNSW semantic vector search with ef_search tuning
│   │       │   ├── bm25.py            # BM25 lexical retriever with field boosts (title^3, breadcrumb^2)
│   │       │   ├── graph_local.py     # GraphRAG 1-2 hop entity neighborhood retriever
│   │       │   └── outliers.py        # Outlier exclusion: Top-k, Distance thresholding, Autocut
│   │       │
│   │       ├── fusion/                # ⚡ Stage 7: Fusion & Post-Retrieval Re-ranking
│   │       │   ├── rrf.py             # Reciprocal Rank Fusion (score = sum(1 / (60 + rank)))
│   │       │   ├── reranker.py        # Cross-encoder re-ranker (BGE-Reranker v2 M3, Zerank 1 Small)
│   │       │   └── capper.py          # Document deduplication & chunk capper (max 2-3 per doc)
│   │       │
│   │       ├── postprocessing/        # ✨ Stage 8: Context Post-Processing & Refinement
│   │       │   ├── expansion.py       # Child to Parent expansion & Sentence Window expansion
│   │       │   ├── compression.py     # Contextual token compression
│   │       │   ├── reorder.py         # LongContextReorder (mitigating lost-in-the-middle)
│   │       │   └── citations.py       # Page & bounding box (bbox) citation formatter
│   │       │
│   │       ├── generation/            # 💬 Stage 9: Generation & Prompt Engineering
│   │       │   ├── llm.py             # LLM client (GPT-4o / Llama-3.1-8B)
│   │       │   ├── prompts.py         # Grounded system prompts, CoT, ToT templates
│   │       │   └── synthesizer.py     # Compact, Refine, Tree Summarize synthesis modes
│   │       │
│   │       ├── agents/                # 🤖 Stage 9: LangGraph Agent Orchestrator
│   │       │   ├── agent.py           # StateGraph multi-tool agent with checkpointing
│   │       │   └── tools/
│   │       │       ├── rag_tool.py    # Document RAG QueryEngine wrapped as LangChain Tool
│   │       │       ├── graph_tool.py  # GraphRAG multi-hop traversal wrapped as LangChain Tool
│   │       │       └── web_tool.py    # Web search fallback Tool
│   │       │
│   │       ├── text_to_sql/           # [DESCOPED] Deferred to future milestones
│   │       │
│   │       ├── kg/                    # 🕸 Stage 4/10: Knowledge Graph Store
│   │       │   ├── neo4j_store.py     # Neo4j driver, Cypher multi-hop query, schema constraints, offline fallback
│   │       │   └── schema.py          # Node (Document, Chunk, Entity, Keyphrase) and Relationship models
│   │       │
│   │       ├── evaluation/            # 📏 Stage 10: Automated Evaluation & Swiss Cheese Harness
│   │       │   ├── code_gates.py        # Layer 1 Deterministic Code Gates (<5ms, 0 tokens)
│   │       │   ├── judge_evaluator.py   # Layer 2 LLM-as-judge CoT [CANONICAL — edit here]
│   │       │   ├── jev_judge.py         # Layer 2.5 Jev speculative fan-out judge
│   │       │   ├── jev_client.py        # Jev HTTP API client
│   │       │   ├── rubrics.py           # 5-part Persian CoT judge rubrics
│   │       │   ├── ragas_adapter.py     # RAGAS metric bindings to local gateway
│   │       │   ├── harness.py           # EvaluationHarness (RAGAS + quality gate)
│   │       │   ├── meta_eval.py         # Confusion matrix + Pass^k / Pass@k
│   │       │   ├── quality_gate.py      # Pass/fail threshold validator
│   │       │   ├── baseline_manager.py  # Baseline snapshots + drift detection
│   │       │   ├── langfuse_dataset.py  # Sync golden set to Langfuse Dataset
│   │       │   ├── run_eval.py          # Modular eval entrypoint (--gates-only, --pipeline)
│   │       │   ├── run_baseline_eval.py # Full baseline eval runner [CANONICAL — make eval → here]
│   │       │   ├── baselines/           # Saved JSON baseline reports
│   │       │   └── datasets/
│   │       │       ├── golden_set.jsonl # 100-item benchmark (75 dev / 25 test)
│   │       │       └── load_dataset.py  # Dataset loader & JSONL integrity validator
│   │       │
│   │       ├── graph/                   # 🔗 Stage 9/10: LangGraph Compiled Pipeline
│   │       │   ├── state.py             # RAGGraphState TypedDict
│   │       │   ├── nodes.py             # Node factories (guard, retrieve, rerank, synth, eval)
│   │       │   └── pipeline_graph.py    # LangGraphRAGPipeline (compiled StateGraph)
│   │       │
│   │       ├── naive/                   # 🧪 Baseline components (simplified e2e)
│   │       │   ├── document_loader.py   # Simple doc loader
│   │       │   ├── chunker.py           # Fixed-size chunker
│   │       │   ├── embedder.py          # BgeM3Embedder wrapper
│   │       │   ├── retriever.py         # NaiveRetriever (numpy cosine)
│   │       │   ├── qdrant_retriever.py  # QdrantRetriever
│   │       │   ├── synthesizer.py       # NaiveSynthesizer
│   │       │   ├── pipeline.py          # NaiveRAGPipeline (full e2e baseline)
│   │       │   ├── judge_evaluator.py   # LEGACY COPY (canonical at evaluation/)
│   │       │   └── run_baseline_eval.py # LEGACY COPY (canonical at evaluation/)
│   │       │
│   │       ├── security/              # 🔒 Stage 10: Enterprise Privacy & Security Guardrails
│   │       │   ├── pii_redactor.py    # Microsoft Presidio PII detection & anonymization
│   │       │   ├── acl_filter.py      # Qdrant payload-based access control filters
│   │       │   └── audit.py           # Structured compliance audit logger
│   │       │
│   │       ├── cache/                 # ⚡ Stage 10: Performance Caching
│   │       │   ├── embedding_cache.py # Redis content-hash embedding cache
│   │       │   └── llm_cache.py       # Redis TTL query/context response cache
│   │       │
│   │       └── observability/         # 👁 Stage 10: Tracing & Logging
│   │           ├── langfuse.py        # Langfuse OpenTelemetry spans (rag_query, retrieval, synthesis)
│   │           └── logging.py         # structlog structured JSON logging configuration
│   │
│   ├── tests/
│   │   ├── test_pipeline_stages.py    # Stage 1-9 component tests
│   │   ├── test_evaluation.py         # Stage 10 evaluation harness tests
│   │   ├── test_graph.py              # LangGraph pipeline tests
│   │   └── test_jev.py                # Jev judge & router tests
│   └── pyproject.toml
│
├── docs/                              # 📚 Project Documentation
│   ├── references/                    # PDFs, images, external reference materials
│   ├── langfuse_manual.md             # Langfuse usage guide & cheatsheet
│   └── *.md                           # Other design & strategy docs
│
├── backend/                           # ⚙ FastAPI REST API Server
│   ├── app/
│   │   ├── main.py                    # Entrypoint, CORS, routers, lifecycle
│   │   ├── api/v1/
│   │   │   ├── router.py              # Aggregated v1 router
│   │   │   └── endpoints/
│   │   │       ├── chat.py            # POST /chat (streaming + RAG)
│   │   │       ├── rag.py             # /retrieve, /index endpoints
│   │   │       ├── documents.py       # Upload, manifest list, delete documents
│   │   │       ├── evaluation.py      # Trigger eval, view history, baselines
│   │   │       └── health.py          # System health checks
│   │   ├── core/
│   │   │   ├── config.py              # Pydantic Settings
│   │   │   ├── logging.py             # structlog configuration
│   │   │   ├── security.py            # JWT, API keys
│   │   │   └── deps.py                # Dependency injection
│   │   ├── services/
│   │   │   ├── chat_service.py        # Conversation management
│   │   │   ├── rag_service.py         # Bridge to rag-core
│   │   │   ├── agent_service.py       # LangGraph agent orchestration
│   │   │   ├── document_service.py    # Ingestion & manifest lifecycle
│   │   │   └── evaluation_service.py  # Evaluation trigger & persistence
│   │   ├── db/
│   │   │   ├── base.py                # SQLAlchemy Base
│   │   │   ├── session.py             # Async database session
│   │   │   └── models.py              # ORM: manifest, eval_runs, conversations
│   │   └── schemas/                   # Pydantic request/response models
│   ├── alembic/                       # Database Migrations
│   │   ├── alembic.ini
│   │   ├── env.py
│   │   └── versions/
│   ├── tests/
│   └── pyproject.toml
│
├── frontend/                          # 🖥 Next.js 14+ CopilotKit Interface
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx             # Root layout & providers
│   │   │   ├── page.tsx               # Landing & navigation
│   │   │   ├── chat/page.tsx          # CopilotKit chat interface
│   │   │   ├── dashboard/evaluation/  # Evaluation metrics dashboard
│   │   │   │   ├── page.tsx
│   │   │   │   └── components/        # MetricsChart, EvalRunTable, QualityGateStatus
│   │   │   ├── documents/page.tsx     # Manifest & document management
│   │   │   └── settings/page.tsx      # RAG pipeline config UI
│   │   ├── components/                # Reusable UI & chat components
│   │   └── lib/api/                   # Typed API client for FastAPI
│   └── package.json
│
├── .github/workflows/
│   ├── rag-eval.yml                   # Automated PR evaluation on RAG changes
│   ├── scheduled-eval.yml             # Weekly drift evaluation
│   └── deploy.yml                     # Production deployment
├── docker-compose.yml                 # Local cluster: Qdrant, Postgres, Redis, Langfuse, API, UI
├── .env.example
└── Makefile
```

---

## Phased Implementation Roadmap

### Phase 0: Project Setup & Environment Isolation (1-2 Days)
- Scaffolding `rag-core/`, `backend/`, and `frontend/` directory trees per the 10-stage architecture.
- Initializing `pyproject.toml` configurations for `rag-core` and `backend`.
- Configuring root `.env.example`, `docker-compose.yml`, and `Makefile`.

### Phase 0.5: Naive Baseline & Ground Zero Evaluation (Completed / Operational)
- Minimalist Naive RAG pipeline (`rag_core.naive.pipeline.NaiveRAGPipeline`): PyMuPDF text extraction, fixed-size chunking, BGE-M3 dense embeddings in Qdrant, and standard synthesis.
- Curated 100-item Golden Benchmark Dataset across 4 suites (`golden_set.jsonl`): Science Textbook, Admin FAQ, Domain Routing, and Adversarial/Refusal.
- Tri-Layered "Swiss Cheese" Evaluation: Layer 1 Deterministic Code Gates (`< 5ms`, 0 tokens) and Layer 2 LLM Judge with 5-part rubrics and `latency_seconds` execution tracking.
- Neo4j Knowledge Graph Store (`rag_core.kg.neo4j_store`): Docker container `neo4j-rag`, schema constraints, and Cypher multi-hop candidate discovery.
- Observability: Langfuse OpenTelemetry span instrumentation (`rag_query`, `retrieval`, `synthesis`) with token, cost, and latency attribution.

### Phase 1: MVP - Core 10-Stage Pipeline & Advanced Evaluation (2-4 Weeks)
- **Stage 1**: Manifest cataloging with SHA-256 deduplication and immediate synchronous cascade hard-deletion across Qdrant, BM25, and Neo4j.
- **Stage 2**: Tiered parsing into Canonical IR (`type`, `text`, `page`, `bbox`) with initial Persian text normalization hooks.
- **Stage 3**: Structure-aware chunking on headings with breadcrumb prepending (`Doc > Section > Subsection (page)`), ~512-token atomic chunks (up to 1024 tokens for cohesive semantic sections), and parent aggregation over N child chunks.
- **Stage 4**: BGE-M3 embedding loader and Qdrant HNSW vector store with payload ACL.
- **Stage 6**: Dense vector retrieval with query prefix.
- **Stage 8**: Bounding box and page-level source citations.
- **Stage 9**: Grounded answer synthesis with "I don't know" fallback.
- **Stage 10**: Local Ragas adapter (`LocalRagasEvaluator` computing Faithfulness, Relevancy, Precision, Recall via local gateway), Layer 1 Code Gate standardized refusal contract (`is_refusal: true`), Neo4j Evol-Instruct synthetic testset generation, and baseline delta verification.

### Phase 2: Optimization - Hybrid Search, Re-ranking & Query Transforms (4-6 Weeks)
- **Stage 4**: BM25 inverted index with `hazm` / `parsivar` Persian NLP normalization, lemmatization, and identifier-preserving tokenizers.
- **Stage 5**: Query rewriting, expansion, HyDE, and sub-question decomposition.
- **Stage 6**: Autocut outlier exclusion and distance thresholding.
- **Stage 7**: Reciprocal Rank Fusion (RRF: `score = sum(1 / (60 + rank))`), BGE-Reranker v2 M3, and document chunk capping.
- **Stage 8**: Parent / Sentence Window expansion and `LongContextReorder`.

### Phase 3: Agent & Orchestration - Multi-Tool & GraphRAG (4-6 Weeks)
- **Stage 4**: GraphRAG entity and typed relationship extraction with Leiden community detection.
- **Stage 9**: LangGraph StateGraph agent orchestrator with checkpointing coordinating Document RAG, GraphRAG, and Web Search (Text-to-SQL descoped).
- **Stage 6**: Graph local 1-2 hop expansion integrated into multi-retriever fusion.

### Phase 4: Enterprise - Security, Caching, API & Full-Stack Deployment (6-12 Weeks)
- **Stage 10**: Microsoft Presidio PII redaction, Qdrant payload ACL filtering, Redis content-hash embedding and TTL response caches.
- **Backend API**: FastAPI REST endpoints with streaming `/chat`, document lifecycle, evaluation endpoints, and Alembic migrations.
- **Frontend UI**: Next.js 14+ CopilotKit chat interface, manifest table, and evaluation dashboard.
- **CI/CD Quality Gates**: GitHub Actions `rag-eval.yml` blocking regressions in CI.

---

## 5. Verification & Testing Strategy: The 3 Tiers of AI Engineering

To maximize developer velocity while enforcing strict empirical quality gates, verification is decoupled into a 3-tier hierarchy:

### Tier 1: Unit & Component Tests (`make test`)
- **Execution Speed**: Sub-second (< 1.0s).
- **Token Cost**: 0 tokens ($0.00).
- **Scope**: Pure Python unit tests in `rag-core/tests/` verifying isolated dataclasses, config validation, AST SQL rules, regex patterns, and component interfaces.
- **Cadence**: Run continuously during local development, on file save, and in git pre-commit hooks.

### Tier 2: Deterministic Dataset Gates (`make test-gates`)
- **Execution Speed**: Fast (1 to 2 seconds for all 100 benchmark items).
- **Token Cost**: 0 tokens ($0.00).
- **Scope**: Executes Layer 1 Deterministic Code Gates (`rag_core.evaluation.code_gates`) across the entire 100-item Golden Benchmark (`golden_set.jsonl`) using `--gates-only`.
- **Target Assertions**:
  - Adversarial Refusal Catch Rate >= 90% (SC-012)
  - Domain Routing Accuracy >= 95% (SC-011)
  - Per-item Execution Duration < 5ms (SC-013)
- **Cadence**: Run before git commit/push, PR validation, and CI smoke gates.

### Tier 2.5: Fast Calibrated System One Evaluation (`make eval` with Jev default)
- **Execution Speed**: Very Fast (~20-30 seconds for 75 dev items, ~300ms per item).
- **Token Cost**: 0 completion tokens (typed probabilistic decisions).
- **Scope**: Evaluates Faithfulness (Noul), 3-level Answer Relevancy (Score), and Adversarial Refusal (Noul) using TypeSafe Jev System One via single-request speculative fan-out.
- **Architectural Fallback**: Automatically degrades to Tier 3 LLM-as-a-Judge (`gemini-cli`) if Jev is unreachable, times out, or reports low confidence (< 0.65).
- **Cadence**: Run after every retrieval or chunking parameter iteration for rapid hill-climbing feedback.

### Tier 3: Deep Generative LLM Evaluation (`make eval-llm`)
- **Execution Speed**: Slower (5 to 10 minutes for 75 items).
- **Token Cost**: Consumes LLM API tokens via Nine Router gateway (`http://localhost:20128/v1`).
- **Scope**: Executes full pipeline retrieval, grounded generation, Layer 2 LLM-as-a-Judge with 5-part rubrics (generating detailed Persian Chain of Thought explanations for human forensics), and Ragas metric adapters (`LocalRagasEvaluator`).
- **Cadence**: Nightly evaluation runs, pre-release tagging, and deep forensic failure debugging.

### Developer Unified Gate: `make check`
Runs `make test` followed by `make test-gates` in sequence (< 2.0s total, $0.00 cost) ensuring zero code regressions and 100% deterministic dataset gate compliance.


---
name: codebase-memory
description: >
  Canonical module map and structural rules for the Unified RAG codebase.
  Agents MUST read this before adding or moving files, and MUST update it
  after every implementation milestone that changes the project layout.
  This is the single source of truth for project organization.
---

# Codebase Memory — Unified RAG

**CRITICAL**: Every time you add, move, rename, or delete a module, update this
file before ending your session. It is the living map of the project.

---

## Project Root Layout

    Unified_RAG/
    ├── .agents/skills/              # SpecKit & custom agent skills
    ├── .specify/                    # SpecKit memory & templates
    ├── specs/001-enterprise-rag-system/  # Feature spec, plan, tasks
    ├── rag-core/                    # Core Python package (10-stage RAG)
    │   ├── src/rag_core/            # Source package
    │   └── tests/                   # Unit & integration tests
    ├── backend/                     # FastAPI REST server
    ├── frontend/                    # Next.js UI
    ├── data/                        # Raw source documents
    ├── data_index/                  # Numpy vector cache
    ├── docs/                        # All project documentation
    │   ├── references/              # PDFs, images, external docs
    │   └── langfuse_manual.md       # Langfuse usage guide
    ├── architecture/                # Architecture diagrams
    ├── docker-compose.yml
    ├── Makefile
    ├── .env / .env.example
    └── development_log.md           # Chronological milestone log

---

## rag_core Module Map (10 Stages)

### Stage 1 — Manifest
    manifest/catalog.py, deduplicator.py, lineage.py

### Stage 2 — Parsing
    parsing/router.py, tier0_layout.py, tier1_ocr.py, tier2_vlm.py, extract_textbook_vlm.py, schema.py

### Stage 3 — Chunking
    chunking/hierarchical.py, parent_child.py, semantic.py, contextual.py

### Stage 4 — Indexing
    indexing/embeddings.py, vector_store.py, bm25_index.py, graphrag.py, indexer.py, science_schema.py, build_science_bm25.py, index_science_qdrant.py

### Stage 5 — Query Understanding
    query/rewriting.py, expansion.py, hyde.py, decomposition.py, routing.py
    query/jev_router.py   <- PRIMARY routing via TypeSafe Jev
    query/gated_rewriter.py <- Flag-driven 4-mode Jev query transformation engine

### Stage 6 — Retrieval
    retrieval/dense.py, bm25.py, hybrid.py, graph_local.py, outliers.py

### Stage 7 — Fusion & Reranking
    fusion/rrf.py, reranker.py, jev_reranker.py, capper.py
    jev_reranker.py does single-call batch scoring via Jev.
    reranker.py does cross-encoder neural reranking (BAAI/bge-reranker-v2-m3).

### Stage 8 — Post-processing
    postprocessing/expansion.py, compression.py, reorder.py, citations.py

### Stage 9 — Generation & Enterprise Orchestrator
    generation/llm.py, prompts.py, synthesizer.py
    pipeline.py              # AdvancedRAGPipeline (Enterprise Science Engine)
    contracts.py             # Interfaces (IBM25Retriever, IRRFFusion, ICrossEncoderReranker, IAdvancedRAGPipeline)
    agents/agent.py + tools/{rag_tool.py, graph_tool.py, web_tool.py}

### Stage 9 — LangGraph Pipeline  <-- CANONICAL EXECUTION ENGINE
    graph/state.py           # RAGGraphState TypedDict
    graph/nodes.py           # Node factories: guard, retrieve, rerank, synth, eval
    graph/pipeline_graph.py  # LangGraphRAGPipeline (compiled StateGraph)

### Stage 10 — Evaluation  <-- CANONICAL: make eval / make eval-dev
    evaluation/
    ├── benchmark_schema.py  # Pydantic v2 schemas: BenchmarkItem, GateResult, SubQJudgeVerdict, ConfusionMatrixResult, EvaluationReport
    ├── code_gates.py        # Layer 1 deterministic gates (<5ms, 0 tokens: refusal, routing, hit@k, citation)
    ├── subq_judge.py        # Layer 2 SubQ Decisions API System One LLM-as-a-Judge with LLMJudge fallback
    ├── decoupled_eval.py    # 2x2 Decoupled Confusion Matrix (TP, FP, TN, FN, Hallucination Catch Rate)
    ├── report_generator.py  # Automated Markdown (Scorecard, Confusion Matrix, Taxonomy) & JSON report generator
    ├── judge_evaluator.py   # Layer 2 LLM-as-judge local fallback (Nine Router / Gemini)
    ├── jev_judge.py         # Layer 2.5 Jev speculative fan-out judge
    ├── jev_client.py        # Jev API client
    ├── rubrics.py           # 5-part Persian CoT rubrics
    ├── ragas_adapter.py     # RAGAS metric bindings
    ├── harness.py           # EvaluationHarness (dual-scorer audit: Jev vs URI)
    ├── meta_eval.py         # Confusion matrix + Pass^k
    ├── quality_gate.py      # Pass/fail threshold validator
    ├── baseline_manager.py  # Baseline snapshots + drift detection
    ├── retrieval_scorer.py  # Approach 2 OCR-tolerant retrieval scorer
    ├── generate_suite_datasets.py # Master 360-item generator across 5 calibrated suites
    ├── build_standardized_benchmarks.py  # Master benchmark generator, corpus aligner & Step 7 validator
    ├── migrate_datasets_to_langfuse.py   # Cloud dataset migration with virtual subfolder hierarchy
    ├── langfuse_dataset.py  # Sync to Langfuse Dataset
    ├── run_eval.py          # Multi-suite runner (--suite, --split, --judge, --scope, --export-report)
    ├── run_baseline_eval.py # Full baseline eval runner
    ├── baselines/           # Saved JSON reports
    └── datasets/
        ├── master_golden_set.jsonl # 360-item calibrated golden benchmark across 5 suites
        ├── dev/dev_set.jsonl       # Sealed 25% dev partition (90 items)
        ├── test/test_set.jsonl     # Sealed 75% test partition (270 items)
        ├── suites/                 # Calibrated CSV suites: science, faq, memory, decomposition, refusal
        └── golden_set.jsonl        # Legacy benchmark mirror

### Stage 10 — Observability
    observability/langfuse.py, logging.py, trace_evaluators.py
    ├── build_standardized_benchmarks.py  # Master benchmark generator, corpus aligner & Step 7 validator
    ├── migrate_datasets_to_langfuse.py   # Cloud dataset migration with virtual subfolder hierarchy
    ├── langfuse_dataset.py  # Sync to Langfuse Dataset
    ├── run_eval.py          # Modular eval (--gates-only, --pipeline)
    ├── run_baseline_eval.py # Full baseline eval runner  [EDIT HERE]
    ├── baselines/           # Saved JSON reports
    └── datasets/
        ├── golden_set.jsonl # 110-item benchmark (81 dev / 29 test across 4 suites)
        ├── suites/          # Master CSV suites (input, expected_output, metadata)
        ├── dev/ & test/     # Partitioned CSV splits
        └── langfuse/        # Mirror for cloud dataset virtual namespaces

### Naive Baseline (simplified baseline components)
    naive/
    ├── document_loader.py, chunker.py, embedder.py
    ├── retriever.py, qdrant_retriever.py, synthesizer.py
    ├── pipeline.py           # NaiveRAGPipeline (e2e baseline)
    ├── judge_evaluator.py    # LEGACY COPY — canonical at evaluation/
    └── run_baseline_eval.py  # LEGACY COPY — canonical at evaluation/

    WARNING: Only edit the evaluation/ versions.
    naive/ copies kept only for regression comparison.

### Stage 10 — Observability
    observability/langfuse.py, logging.py

### Stage 10 — Security & Cache
    security/pii_redactor.py, acl_filter.py, audit.py
    cache/embedding_cache.py, llm_cache.py
    kg/neo4j_store.py, schema.py
    text_to_sql/  [DESCOPED]

---

## Naming & Structure Conventions

| Convention          | Rule                                                              |
|---------------------|-------------------------------------------------------------------|
| Module naming       | snake_case for all files and directories                          |
| One class per file  | Each module exposes one primary class/function                    |
| Stage isolation     | Stage N must NOT import from Stage N+2 or later                   |
| naive/ policy       | Do NOT add new stage-level logic to naive/                        |
| Tests mirror source | tests/test_<module>.py for each src/rag_core/<module>/            |
| Eval canonical path | evaluation/run_baseline_eval.py — NOT naive/run_baseline_eval.py |

---

## Make Targets Quick Reference

| Target           | Tier | What it does                          |
|------------------|------|---------------------------------------|
| make test        | 1    | Unit tests (<1s, 0 tokens)            |
| make test-gates  | 2    | Deterministic gate sweep              |
| make check       | 1+2  | Pre-commit gate (test + test-gates)   |
| make eval        | 2.5  | Jev full baseline (75 dev items)      |
| make eval-smoke  | 2.5  | Quick smoke (5 items)                 |
| make eval-force  | 2.5  | Forced live evaluation (no cache)     |
| make eval-llm    | 3    | LLM judge (Persian CoT)               |
| make eval-test   | 2.5  | Held-out test split (25 items)        |
| make eval-graph  | 2.5  | LangGraph compiled pipeline           |
| make eval-modular| 2    | Gates only                            |
| make eval-legacy | 2.5  | Legacy naive/ (regression comparison) |

---

## Agent Update Obligation

After every session that changes the layout:
1. Update the relevant section above
2. Add a dated entry in development_log.md
3. Run `make check` to verify unit tests and code gates pass
4. Refresh Knowledge Graph via `make kg-status` / `codebase-memory-mcp index_repository(persistence=True)`
5. If tests were added, update the tests section above

Never let this file become stale. It is the FIRST file agents read.

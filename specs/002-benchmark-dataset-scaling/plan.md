# Implementation Plan: Benchmark Dataset Scaling & Comprehensive Quality Evaluation

**Branch**: `002-benchmark-dataset-scaling` | **Date**: 2026-10-08 | **Spec**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/002-benchmark-dataset-scaling/spec.md)

**Input**: Feature specification from [`specs/002-benchmark-dataset-scaling/spec.md`](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/002-benchmark-dataset-scaling/spec.md)

## Summary

Scale our evaluation benchmark from 110 items to **400+ high-quality items** across the administrative FAQ and 15-chapter science curriculum corpora. The technical approach:
1. Assign canonical reference IDs (`faq_000` to `faq_133`) to [faq_qa_pairs.csv](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/data/faq_qa_pairs.csv) and batch-generate colloquial paraphrases, cross-category keyword clashes, multi-row combinations, and out-of-scope refusals via `gemini-cli`.
2. Extract curriculum units from the 15 pre-parsed VLM Markdown files in `data/vlm_parsed/stitched/` and synthesize factoids, multi-hop reasoning, diagram queries, and laboratory experiment items across all 412 sections in [manifest.json](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/data/vlm_parsed/manifest.json).
3. Refactor [run_eval.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/evaluation/run_eval.py) to decouple intermediate evaluation, providing zero-token deterministic retrieval metrics (`Hit@k`, `MRR`) alongside LLM-as-a-judge generation metrics and citation provenance verification.
4. Export to standardized 3-column CSVs and [golden_set.jsonl](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/evaluation/datasets/golden_set.jsonl), synchronizing idempotently to Langfuse Cloud under virtual namespaces.

---

## Technical Context

**Language/Version**: Python 3.11.15 in `.venv`  
**Primary Dependencies**: `ragas` (0.4.3), `langfuse` (>=4.17.0), `pydantic`, `pandas`, `jsonschema`, `structlog`  
**Storage**: Filesystem JSONL/CSV under `rag-core/src/rag_core/evaluation/datasets/`, remote dataset sync on Langfuse Cloud  
**Testing**: `pytest`, `make check`, JSON schema contract tests  
**Target Platform**: macOS / Linux local runner with Nine Router gateway (`http://localhost:20128/v1`)  
**Project Type**: Evaluation harness and synthetic generation pipeline within `rag-core`  
**Performance Goals**: Decoupled retrieval evaluation over 300+ items completes in < 3.0 seconds with 0 LLM tokens consumed; synthetic generation batch runs with retry backoff and rate-limit smoothing  
**Constraints**: Zero NaN/null fields in benchmark items; strict 75/25 dev/test split isolation; 100% chunk attribution required  
**Scale/Scope**: 400+ total items across 4 suites (Administrative FAQ >= 200, Science Textbook >= 180, Domain Routing = 25, Adversarial Refusal = 25)

---

## Constitution Check

*GATE: All checks verified compliant with Enterprise RAG Constitution v1.4.0.*

- [x] **Principle I: 3-Layer Decoupled Architecture**: All evaluation logic, schemas, and generators reside strictly inside `rag-core`. Zero coupling to backend (FastAPI) or frontend (Next.js).
- [x] **Principle II: Evaluation-Driven Quality**: Expands the golden benchmark from 110 items to 400+ items to establish a statistically rigorous 95% confidence interval, directly advancing constitutional quality governance.
- [x] **Principle III: Granular Observability**: All generated datasets adhere to Langfuse dataset specifications and synchronize with trace score logging.
- [x] **Principle VII: Law of Parsimony & Component Maximization**: Decouples retrieval evaluation into a deterministic code gate (< 5ms, 0 tokens), avoiding unnecessary LLM judge invocations for pure search quality measurement.
- [x] **Principle VIII: Continuous Development Logging**: Milestone results and metric updates will be recorded in `development_log.md`.
- [x] **Principle IX: Workspace Hygiene**: Dataset files organized into standard `suites/`, `dev/`, and `test/` subdirectories with no dangling artifacts.
- [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance**:
  - [x] Checked codebase memory graph status prior to planning (`detect_changes` confirmed `changed_count: 0`).
  - [x] Post-implementation tasks include mandatory `codebase-memory-reindex`.

---

## Project Structure

### Documentation (this feature)

```text
specs/002-benchmark-dataset-scaling/
├── plan.md              # This file
├── research.md          # Architectural decisions & query taxonomies
├── data-model.md        # Schemas for BenchmarkItem, RetrievalScoreRecord, CitationScoreRecord
├── quickstart.md        # End-to-end generation and validation commands
├── contracts/
│   ├── benchmark_item_schema.json
│   └── evaluation_report_schema.json
└── checklists/
    └── requirements.md  # Quality validation checklist
```

### Source Code

```text
rag-core/
├── src/rag_core/
│   ├── evaluation/
│   │   ├── build_standardized_benchmarks.py  # Master generator orchestrating FAQ & Science scaling
│   │   ├── run_eval.py                       # CLI runner with decoupled --retrieval-only support
│   │   ├── harness.py                        # Decoupled retrieval and generation scoring harness
│   │   ├── code_gates.py                     # Deterministic Layer 1 retrieval & refusal gates
│   │   ├── migrate_datasets_to_langfuse.py   # Langfuse Cloud virtual namespace synchronizer
│   │   └── datasets/
│   │       ├── golden_set.jsonl              # 400+ item master benchmark
│   │       ├── golden_set.csv
│   │       ├── dev/                          # 75% stratified dev split
│   │       ├── test/                         # 25% sealed test split
│   │       └── suites/                       # Isolated suites per domain
│   └── postprocessing/
│       └── citations.py                      # Provenance parser and attribution evaluator
└── tests/
    └── test_benchmark_schema.py             # Schema contract and split integrity unit tests
```

---

## Complexity Tracking

No constitutional violations or unnecessary complexity introduced. Deterministic retrieval scoring directly reduces token overhead.

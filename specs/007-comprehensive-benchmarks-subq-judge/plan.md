# Implementation Plan: Comprehensive Multi-Suite Benchmarks, Deterministic Code Gates & SubQ Decision LLM-as-a-Judge (007)

**Branch**: `007-comprehensive-benchmarks-subq-judge` | **Date**: 2026-10-14 | **Spec**: [specs/007-comprehensive-benchmarks-subq-judge/spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/007-comprehensive-benchmarks-subq-judge/spec.md)

**Input**: Feature specification from `specs/007-comprehensive-benchmarks-subq-judge/spec.md`

---

## Summary

This feature establishes an enterprise-grade evaluation and benchmarking framework for Unified RAG to provide undeniable empirical rigor for academic and technical mentors:
1. **5-Suite Calibrated Benchmark Datasets**: 360 validated items partitioned across `science_curriculum`, `administrative_faq`, `conversational_memory`, `subquery_decomposition`, and `adversarial_refusal` with a realistic 3-tier difficulty distribution (20% Easy, 50% Medium, 30% Hard/Adversarial) and a sealed 25% `dev` (90 items) / 75% `test` (270 items) split.
2. **Layer 1 Deterministic Code Gates**: High-speed, zero-token Python checks (Persian refusal regex, domain routing keyword distribution, retrieval Hit@k / MRR, and citation formatting) that resolve over 35% of evaluation items in under 5ms without invoking any LLM.
3. **Layer 2 SubQ Decisions API LLM-as-a-Judge**: Elevating `subq-snap-preview` into a primary high-throughput System One judge evaluating Faithfulness, Relevancy, Citation Attribution, and Turn Relevance via typed schemas, 384K context window, and 10-10 parallel batching (< 600ms per batch), with seamless fallback to local Gemini `LLMJudge`.
4. **Decoupled 2x2 Attribution Matrix & Mentor-Ready Reporting**: Disentangling Retrieval Failures from Generation Failures, computing Hallucination Catch Rate (`TN / (TN + FP)`), and automatically compiling an executive scorecard and failure taxonomy report in Markdown and JSON.

---

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: `pydantic` (v2), `httpx` / `urllib`, `langfuse`, `qdrant-client`, `rank_bm25`, `hazm` (Persian NLP), `structlog`, `pytest`

**Storage**:
- Version-controlled JSONL / CSV dataset files in `rag-core/src/rag_core/evaluation/datasets/` (`suites/`, `dev/`, `test/`)
- Vector embeddings and payloads in Qdrant (`data_index/qdrant_db`)
- Observability and evaluation traces in Langfuse (`http://localhost:3000` or Cloud)

**Testing**: `pytest` (`rag-core/tests/test_code_gates_expanded.py`, `rag-core/tests/test_subq_judge.py`, `rag-core/tests/test_evaluation_suites.py`)

**Target Platform**: macOS / Linux (sandboxed and production server environments)

**Project Type**: Core Python Package (`rag-core/src/rag_core/evaluation/`) with CLI runner (`run_eval.py`)

**Performance Goals**:
- Layer 1 Deterministic Code Gates: < 5ms per item (0 LLM tokens consumed).
- Layer 2 SubQ System One Judge: < 600ms per batch of 10 items.
- Full Dev-Split Benchmark Evaluation (90 items): < 45 seconds.

**Constraints**:
- Must not artificially saturate at 100% (at least 30% hard/adversarial test items).
- Strict, seamless fallback to local `LLMJudge` if SubQ Decisions API is unreachable or rate-limited.
- Strict Pydantic v2 validation at all pipeline boundaries (`BenchmarkItem`, `GateResult`, `SubQJudgeVerdict`).
- No raw LaTeX formulas in reports or chat outputs.
- Mandatory Codebase Memory Knowledge Graph re-indexing on milestone completion (`detect_changes` reports 0 drift).

**Scale/Scope**: 360 validated items across 5 suites, 15 Grade 7 Science textbook chapters, administrative circulars, and multi-turn conversational traces.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Principle I: Three-Layer Decoupled Architecture**: All evaluation logic, datasets, judges, and code gates reside strictly within `rag-core` and do not depend on FastAPI or Next.js.
- [x] **Principle II: Evaluation-Driven Quality (Non-Negotiable)**: Implements continuous RAGAS and SubQ metrics (Faithfulness >= 0.85, Relevancy >= 0.80, Science Hit@5 >= 0.80, FAQ Hit@5 >= 0.85).
- [x] **Principle III: Granular Observability & Open Inference Standards**: Every evaluation run logs execution traces and spans to Langfuse, attaching judge verdicts and decoupled scores as trace scores.
- [x] **Principle V: Defense-in-Depth Security**: Validates adversarial prompt injections and PII redactions without leaks.
- [x] **Principle VII: Law of Parsimony & Component Maximization**: Deterministic code gates precede LLM judges. Every score is benchmarked against our verified Naive Baseline ("Ground Zero").
- [x] **Principle VIII: Continuous Development Logging**: Milestone updates recorded in `development_log.md`.
- [x] **Principle IX: Workspace Hygiene**: All code in `.venv`, secrets in `.env`, datasets in `rag-core/src/rag_core/evaluation/datasets/`.
- [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance**:
  - [x] Checked codebase memory graph status prior to planning (`list_projects`, `index_status` ready with 4606 nodes, 12228 edges).
  - [x] Implementation plan includes post-phase KG refresh checkpoint (`index_repository(persistence=True)`).
  - [x] Verified zero unindexed drift requirement before milestone completion.
- [x] **Principle XI: Software Engineering & Change Discipline**: Schema-first Pydantic v2 input validation, no commented-out dead code, human approval before git commit.
- [x] **Principle XII: Enterprise Rigor & The Tool Trinity**: Full operation of Git, Spec-Kit, and Langfuse trinity.

---

## Project Structure

### Documentation (this feature)

```text
specs/007-comprehensive-benchmarks-subq-judge/
├── spec.md                                     # Feature specification
├── plan.md                                     # Implementation plan (this file)
├── research.md                                 # Phase 0 architectural decisions
├── data-model.md                               # Phase 1 Pydantic v2 schemas
├── quickstart.md                               # Phase 1 verification and run guide
├── contracts/                                  # Phase 1 interface contracts
│   ├── benchmark_item_schema.json              # Dataset item JSON Schema
│   ├── subq_judge_contract.json                # SubQ API protocol contract
│   └── evaluation_report_schema.json           # Evaluation report JSON Schema
├── checklists/
│   └── requirements.md                         # Quality validation checklist
└── tasks.md                                    # Phase 2 task checklist (generated by /speckit-tasks)
```

### Source Code (repository root)

```text
rag-core/
├── src/rag_core/
│   ├── evaluation/
│   │   ├── benchmark_schema.py                 # Pydantic v2 schemas: BenchmarkItem, GateResult, SubQJudgeVerdict
│   │   ├── code_gates.py                       # Expanded Layer 1 Deterministic Code Gates
│   │   ├── subq_judge.py                       # Layer 2 SubQ Decisions API System One Judge
│   │   ├── judge_evaluator.py                  # Layer 2 Generative LLMJudge (Gemini Fallback)
│   │   ├── jev_judge.py                        # Layer 2.5 Jev Evaluator
│   │   ├── decoupled_eval.py                   # Decoupled 2x2 Matrix & Hallucination Catch Rate
│   │   ├── run_eval.py                         # Unified CLI runner & report exporter
│   │   └── datasets/
│   │       ├── suites/                         # Master CSV/JSONL datasets for all 5 suites
│   │       │   ├── science_curriculum.csv      # 120 items
│   │       │   ├── administrative_faq.csv      # 80 items
│   │       │   ├── conversational_memory.csv   # 60 items
│   │       │   ├── subquery_decomposition.csv  # 50 items
│   │       │   └── adversarial_refusal.csv     # 50 items
│   │       ├── dev/                            # 90-item active development split
│   │       └── test/                           # 270-item sealed milestone split
│   └── query/
│       └── subq_client.py                      # SubQDecisionClient interface
backend/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   └── evaluation.py                   # REST endpoints for benchmark runs & status polling
│   │   └── router.py                           # Sub-router registration
│   ├── schemas/
│   │   └── evaluation.py                       # Pydantic v2 API schemas
│   └── services/
│       └── evaluation_service.py               # Background evaluation execution service
frontend/
├── src/
│   ├── components/
│   │   ├── EvaluationModal.tsx                 # Interactive evaluation dashboard & measurement visualizer
│   │   └── Sidebar.tsx                         # Navigation launcher button
│   └── types/
│       └── index.ts                            # Evaluation TypeScript interfaces
└── tests/
    ├── test_code_gates_expanded.py             # Layer 1 deterministic gate tests
    ├── test_subq_judge.py                      # Layer 2 SubQ judge tests with fallback
    └── test_evaluation_suites.py               # Dataset validation and suite execution tests
```

**Structure Decision**: Implements core RAG evaluation logic inside `rag-core/src/rag_core/evaluation/` adhering to Principle I (Three-Layer Decoupled Architecture). The FastAPI `backend/` exposes REST endpoints (`/api/v1/evaluation`) to trigger background runs, and the Next.js `frontend/` provides an interactive `EvaluationModal` with live progress bars and comprehensive visual scorecards for mentors.

---

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| :--- | :--- | :--- |
| *None* | All designs strictly adhere to parsimony and the 3-layer decoupled architecture. | N/A |

---

## Planned Execution Phases

### Phase 1: Core Schemas & Expanded Layer 1 Deterministic Gates
- Create updated Pydantic v2 models in `rag-core/src/rag_core/evaluation/benchmark_schema.py`.
- Expand `rag-core/src/rag_core/evaluation/code_gates.py` to cover:
  - `evaluate_refusal_gate`: Exact & regex Persian pattern matching.
  - `evaluate_domain_routing_gate`: Keyword & category distribution.
  - `evaluate_retrieval_hit_gate`: Deterministic Hit@1, Hit@3, Hit@5, and MRR.
  - `evaluate_citation_gate`: Inline citation format checking.
- Write unit tests in `rag-core/tests/test_code_gates_expanded.py`.

### Phase 2: SubQ Decisions API System One LLM-as-a-Judge
- Implement `SubQJudge` in `rag-core/src/rag_core/evaluation/subq_judge.py`:
  - 4 rubrics: `faithfulness`, `answer_relevancy`, `citation_attribution`, `turn_relevance`.
  - 10-10 concurrent batch execution via `asyncio.gather`.
  - Automatic fallback to `LLMJudge` (Nine Router / Gemini) on timeout or HTTP failure.
- Write unit tests in `rag-core/tests/test_subq_judge.py`.

### Phase 3: Calibrated 5-Suite Dataset Generation & Sealing
- Build and standardize 360 benchmark items across:
  - `science_curriculum.csv` (120 items)
  - `administrative_faq.csv` (80 items)
  - `conversational_memory.csv` (60 items)
  - `subquery_decomposition.csv` (50 items)
  - `adversarial_refusal.csv` (50 items)
- Partition into sealed 25% `dev` (90 items) and 75% `test` (270 items).
- Write dataset schema validation tests in `rag-core/tests/test_evaluation_suites.py`.

### Phase 4: Decoupled Attribution Matrix & Mentor-Ready Reporting
- Implement decoupled 2x2 matrix calculation (`TP`, `FP`, `TN`, `FN`, and Hallucination Catch Rate) in `rag-core/src/rag_core/evaluation/decoupled_eval.py`.
- Upgrade `rag-core/src/rag_core/evaluation/run_eval.py` to:
  - Parse CLI flags (`--split`, `--suite`, `--judge`, `--limit`, `--export-report`).
  - Stream progress and export executive Markdown (`evaluation_report.md`) and JSON reports.
  - Log evaluation trace scores to Langfuse.

### Phase 5: Backend API Endpoints for Evaluation Runs
- Create Pydantic API schemas in `backend/app/schemas/evaluation.py`.
- Implement `backend/app/services/evaluation_service.py` managing background evaluation execution and state polling.
- Expose REST endpoints in `backend/app/api/v1/endpoints/evaluation.py` (`POST /run`, `GET /status/{run_id}`, `GET /suites`, `GET /runs`).

### Phase 6: Interactive Frontend Evaluation Dashboard
- Build `frontend/src/components/EvaluationModal.tsx`:
  - Benchmark Suite Selector: dropdown or pill buttons (`All Suites`, `Science`, `FAQ`, `Memory`, `Decomposition`, `Refusal`).
  - Configuration Controls: Scope (`End-to-End`, `Retrieve Only`, `Synthesis Only`, `Gates Only`), Split (`dev` / `test`), Judge Model (`SubQ`, `Jev`, `Gemini`), and Item Limit (5 to 100).
  - Live Progress Indicator: Animated progress bar, elapsed time timer, and current query display.
  - Comprehensive Measurement Cards: Pass Rate, Hit@1, Hit@5, MRR, Faithfulness, Relevancy, Latency, and Cost.
  - 2x2 Decoupled Confusion Matrix: Interactive visual grid with Hallucination Catch Rate.
  - Failure Taxonomy Drill-down: Table of failing items with clickable queries, retrieved chunks, judge reasoning, and Langfuse trace links.
  - "Download Mentor Report" button exporting `evaluation_report.md`.
- Add evaluation launcher button in `frontend/src/components/Sidebar.tsx`.

### Phase 7: Codebase Memory Knowledge Graph Refresh & Verification
- Execute `call_mcp_tool` `index_repository(persistence=True)` to rebuild `.codebase-memory/graph.db.zst`.
- Run `detect_changes` to verify zero drift.

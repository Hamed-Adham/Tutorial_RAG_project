# Feature Specification: Comprehensive Multi-Suite Benchmarks, Deterministic Code Gates & SubQ Decision LLM-as-a-Judge (007)

**Feature Branch**: `007-comprehensive-benchmarks-subq-judge`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "the biggest issue is we dont have right benchmark and evaluation for what we have done . also I dont how can I represent this project to my mentor who insist on evals . we need some determinics code evals and some benchmarks ( we have some but they are outdated and we should build more case by case ones , ( we shouldnt creater such a benchmarks that we always hit 100% onthem ) what suit cases do u think we have to have and how many data per suit case is suitable . what KPI and targets we have set for our project and where we are . do we have implemented LLM as judge ? specially since we have the # SubQ Decisions API and the subq-snap-preview model we could use it as the judge in the LLM as judge . so help me to have this perfect evaluation section ."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-Suite Calibrated Benchmark Datasets with Realistic Difficulty Distribution (Priority: P1)

As an evaluation lead presenting to academic and engineering mentors, I want comprehensive, multi-suite benchmark datasets (spanning science curriculum, administrative policy, conversational memory, multi-hop subqueries, and adversarial injections) structured with a realistic 3-tier difficulty distribution (20% Easy, 50% Medium, 30% Hard/Adversarial), so that our evaluation reflects real-world operational stress rather than an artificially inflated 100% pass rate.

**Why this priority**: A benchmark where the system achieves 100% is untrustworthy and uninformative to mentors and stakeholders. Calibrated difficulty with difficult negative boundaries, term clashes, and complex multi-hop questions exposes genuine bottlenecks and empirically proves optimization gains.

**Independent Test**: Can be validated by executing schema validation and running the evaluation harness across all suites, verifying that baseline models achieve realistic discriminative scores (e.g. 40-70%) while advanced configurations achieve targeted target improvements without saturation.

**Acceptance Scenarios**:

1. **Given** a 350+ item master benchmark partitioned into 5 dedicated test suites, **When** inspected for difficulty distribution, **Then** at least 30% of items are tagged as `hard` or `adversarial`, testing complex multi-hop retrieval, cross-chapter terminology clashes, and prompt injection boundaries.
2. **Given** a conversational memory benchmark item with up to 10 historical turns, **When** evaluated, **Then** the benchmark tests whether the system selectively retrieves only the turns necessary for disambiguation while ignoring distracting irrelevant turns.
3. **Given** a multi-hop curriculum query requiring information from two distinct chapters (e.g. chemical energy in food linked to digestive absorption), **When** evaluated, **Then** ground truth specifies multiple required chunks and checks if partial retrieval is flagged.

---

### User Story 2 - SubQ System One LLM-as-a-Judge Integration (Priority: P1)

As a RAG system developer, I want to deploy the SubQ Decisions API (`subq-snap-preview`) as a high-speed, calibrated System One LLM-as-a-Judge for evaluating response Faithfulness, Answer Relevancy, Citation Precision, and Turn Relevance, so that we have fast (< 500ms), typed, cost-efficient, and objective evaluation verdicts that correlate strongly with human judgements and scale to hundreds of test items.

**Why this priority**: Traditional generative LLM judges (calling generalist 70B/chat models) are slow (3-10s per call), expensive, prone to verbose extraction errors, and bottleneck local test runs. SubQ's 384K context window, typed decision schema, and sub-second latency provide high-throughput evaluation directly aligned with our architecture.

**Independent Test**: Can be verified by passing a test batch of query-context-response tuples to the SubQ judge client, verifying typed verdicts (pass/fail, continuous score, reasoning), sub-500ms latency, and graceful fallback to local judge if unreachable.

**Acceptance Scenarios**:

1. **Given** a generated response with a subtle hallucination (e.g. wrong chemical formula or unauthorized loan percentage), **When** evaluated by the SubQ Faithfulness Judge, **Then** it returns verdict = 0 (Fail), a low groundedness score, and an explicit explanation citing the ungrounded clause.
2. **Given** a perfectly grounded response that accurately answers the user's question, **When** evaluated, **Then** both Faithfulness and Answer Relevancy return verdict = 1 (Pass) with confidence >= 0.85.
3. **Given** a batch of 20 evaluation samples, **When** dispatched via SubQ parallel batching, **Then** all 20 verdicts resolve concurrently in under 2 seconds.

---

### User Story 3 - Expanded Layer 1 Deterministic Code Gates (Priority: P2)

As a performance and quality engineer, I want an expanded suite of deterministic Python code gates that execute before any LLM judge call, verifying refusal syntax, domain routing, retrieval Hit@k, citation presence, and schema compliance in under 5ms with zero token cost, so that over 35% of evaluation items are resolved deterministically without wasting model inferences.

**Why this priority**: Deterministic gates provide 100% reproducible, objective ground truth for boundary conditions (refusals, routing, retrieval hits) and eliminate redundant LLM calls, saving time and money.

**Independent Test**: Can be tested by running the gate suite over adversarial refusal, domain routing, and empty retrieval cases, verifying that `gated=True`, latency is < 1ms, and LLM judge invocation count is 0 for those cases.

**Acceptance Scenarios**:

1. **Given** an adversarial prompt injection or out-of-domain query, **When** the system outputs a refusal, **Then** the Refusal Code Gate verifies Persian regex patterns and exact formulas in < 1ms, scoring it as Pass (1) without invoking an LLM judge.
2. **Given** an answer generated with missing or invalid citation brackets, **When** passed through the Citation Code Gate, **Then** it immediately flags missing citations deterministically.
3. **Given** a benchmark query with known gold chunk IDs, **When** retrieval completes, **Then** the Retrieval Code Gate computes Hit@1, Hit@3, Hit@5, and MRR deterministically without model evaluation.

---

### User Story 4 - Decoupled Retrieval vs. Generation Diagnostic Matrix (Priority: P2)

As an engineer preparing progress reports for my mentor, I want the evaluation harness to produce a decoupled 2x2 diagnostic confusion matrix (Retrieval Success/Failure vs Generation Faithfulness/Hallucination), so that I can clearly demonstrate whether any performance gap is caused by retrieval failure (missing chunks) or LLM synthesis failure (hallucination, claim omission, or instruction drift).

**Why this priority**: Mentors demand scientific attribution of errors. A single composite score hides whether the embedding/retrieval model or the generator failed. A decoupled matrix provides undeniable empirical clarity.

**Independent Test**: Can be verified by running the harness over 50 items and verifying the generation of the 4 diagnostic quadrants:
- True Positive (TP): Retrieval Success + Faithful Generation
- False Negative (FN): Retrieval Failure + Hallucination / Omission
- False Positive (FP): Retrieval Success + Hallucination (Pure LLM Synthesis Failure)
- True Negative (TN): Retrieval Failure + Clean Refusal (Correct Safety Behavior)

**Acceptance Scenarios**:

1. **Given** a query where retrieval fetches the wrong chunk and the LLM invents facts, **When** analyzed, **Then** the system classifies it as Retrieval Failure + Synthesis Hallucination (FN).
2. **Given** a query where retrieval fetches the correct chunk but the LLM hallucinates an outside fact, **When** analyzed, **Then** it classifies it as Retrieval Success + Synthesis Hallucination (FP), isolating the error specifically to prompt/synthesis.
3. **Given** a complete benchmark run, **When** the harness finishes, **Then** it calculates Hallucination Catch Rate = TN / (TN + FP) and Retrieval Coverage = (TP + FP) / Total.

---

### User Story 5 - Mentor-Ready Executive Evaluation Dashboard & Langfuse Telemetry (Priority: P3)

As a mentee reporting to an advisor, I want an automated summary report generator that exports a polished, publication-ready Markdown/HTML dashboard comparing our historical Naive RAG Baseline ("Ground Zero") against current pipeline performance across all KPIs, alongside deep Langfuse trace links, so that I can present an irrefutable, empirical defense of our engineering progress.

**Why this priority**: Advisors and mentors evaluate projects based on quantifiable progress from baseline to state-of-the-art, transparency in failure modes, and industrial-grade observability, not subjective demos.

**Independent Test**: Can be tested by running `make eval-report` or calling the report generator, verifying the output of an executive table with baseline vs current deltas, error taxonomy, and clickable Langfuse trace references.

**Acceptance Scenarios**:

1. **Given** an evaluation run across all suites, **When** the report compiles, **Then** it generates an executive scorecard table comparing Naive Baseline vs Current Score vs Target KPI for every metric.
2. **Given** any failed item in the report, **When** inspected, **Then** the report includes the exact query, gold answer, retrieved chunks, generated response, judge reasoning, and Langfuse trace ID.

---

### User Story 6 - Interactive Frontend Evaluation Dashboard & Live Pipeline Runner (Priority: P2)

As a mentor or system evaluator, I want an interactive, visual Evaluation Dashboard in the web frontend where I can select any benchmark suite (or all), configure evaluation options (e.g. retrieval-only vs end-to-end, split selection, judge model, sample limit), trigger live evaluation runs with a streaming progress indicator, and explore comprehensive measurements (scorecards, 2x2 confusion matrix, and failure taxonomy) directly in the UI, so that I can rigorously audit pipeline performance and reproducibility without requiring command-line access.

**Why this priority**: Mentors and non-developer stakeholders need immediate visual access to test suites, interactive configuration, and live measurements rather than reading static log outputs or terminal scripts.

**Independent Test**: Can be validated by opening the frontend Evaluation Dashboard, selecting `science_curriculum` with `Retrieve Only` mode and a limit of 10 items, clicking "Run Evaluation", observing the live progress bar, and inspecting the resulting Hit@k scores and chunk retrieval breakdown in the UI.

**Acceptance Scenarios**:

1. **Given** a user on the frontend, **When** they click the "Evaluation Dashboard" button in the sidebar, **Then** an interactive evaluation interface opens displaying available suites, past run history, and configuration controls.
2. **Given** configuration options for suite (`All Suites`, `science_curriculum`, `administrative_faq`, etc.), split (`dev` vs `test`), scope (`End-to-End`, `Retrieve Only`, `Synthesis Only`, `Gates Only`), judge backend (`SubQ`, `Jev`, `Gemini LLM`), and limit (e.g. 10 items), **When** the user clicks "Start Benchmark Run", **Then** the UI dispatches the run to `POST /api/v1/evaluation/run` and displays a real-time progress bar with elapsed time and current item queries.
3. **Given** a completed evaluation run, **When** results are rendered, **Then** the UI displays:
   - Executive Scorecard cards: Pass Rate, Hit@1, Hit@5, MRR, Faithfulness, Relevancy, Latency, and Cost.
   - Interactive 2x2 Decoupled Confusion Matrix: TP, FP, TN, FN, and Hallucination Catch Rate.
   - Drill-down Failure Explorer: Table of failing items with query text, expected vs actual answers, retrieved chunks, judge reasoning, and clickable Langfuse trace link.
4. **Given** a user viewing results, **When** they click "Download Mentor Report", **Then** the system exports `evaluation_report.md` directly to their local browser.

---

### Edge Cases

- **Mixed Intent Queries**: Queries that contain conversational pleasantries combined with curriculum questions ("هوا خوبه ولی بطن‌های قلب چیستند؟"). The benchmark must verify that the gate isolates the core question rather than triggering a false refusal.
- **SubQ API Downtime or Rate Limiting**: If the SubQ Decisions API is unreachable, times out, or returns 429, the judge harness must gracefully fall back to the local `LLMJudge` without crashing the evaluation run.
- **Sealed Test Split Contamination**: Benchmark data must maintain a strict, sealed 70/30 split between `test` and `dev` sets. Development tuning must strictly run on `dev` to prevent test-set overfitting.
- **Zero-Retrieval Refusal Correctness**: An adversarial query should ideally retrieve nothing and produce a refusal. If retrieval brings irrelevant chunks but the generator still correctly refuses, the system must score this as a successful refusal rather than a generation failure.
- **Frontend Long-Running Evaluation Timeout**: If an evaluation run on 270 items takes 2 minutes, the frontend must not block HTTP connections; it must poll background status idempotently.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST define five calibrated evaluation suites with standardized schemas:
  1. `science_curriculum`: Factual, conceptual, diagrammatic, and multi-hop science questions across physics, chemistry, biology, and geology.
  2. `administrative_faq`: Administrative, financial, regulatory, and fund policies with terminology collisions.
  3. `conversational_memory`: Multi-turn dialogues (2 to 10 turns) testing coreference resolution, antecedent tracking, and summary backfill.
  4. `subquery_decomposition`: Complex compound queries requiring multi-step decomposition and synthesis.
  5. `adversarial_refusal`: Out-of-domain, unanswerable, ambiguous, and prompt-injection queries requiring standard Persian refusal.
- **FR-002**: Benchmark suites MUST incorporate a controlled difficulty profile: approximately 20% Easy, 50% Medium, and 30% Hard/Adversarial to ensure the benchmark does not saturate at 100%.
- **FR-003**: The total master benchmark MUST contain between 350 and 500 validated items, partitioned into a 25% `dev` split (for active development) and a 75% sealed `test` split (for formal milestone verification).
- **FR-004**: Each benchmark item MUST conform to a strict Pydantic v2 contract (`BenchmarkItem`) containing: `item_id`, `suite`, `split`, `difficulty`, `query`, `ground_truth`, `expected_route`, `gold_chunk_ids`, `gold_doc_ids`, `history_turns` (if conversational), and `metadata`.
- **FR-005**: The system MUST implement an expanded Layer 1 Deterministic Code Gate suite evaluating:
  - `refusal_gate`: Persian exact and regex pattern matching against standardized refusal formulas (< 1ms, 0 tokens).
  - `domain_routing_gate`: Keyword and semantic category alignment between query and retrieved chunks.
  - `retrieval_hit_gate`: Deterministic Hit@1, Hit@3, Hit@5, and Mean Reciprocal Rank (MRR) based on `gold_chunk_ids`.
  - `citation_format_gate`: Validation of inline citation markers (e.g. `[doc:page]`) and reference list existence.
- **FR-006**: The system MUST integrate the SubQ Decisions API (`https://api.subq.ai/v1/systemone` using model `subq-snap-preview`) as a high-speed, typed System One LLM-as-a-Judge (`SubQJudge`).
- **FR-007**: `SubQJudge` MUST implement four standardized judgment rubrics:
  1. `faithfulness`: Checks whether every factual statement in the answer is strictly substantiated by the context.
  2. `answer_relevancy`: Checks whether the answer directly addresses the user query without omitting critical conditions or adding superfluous tangents.
  3. `citation_attribution`: Checks whether cited references actually contain the claims attributed to them.
  4. `history_turn_relevance`: Checks whether a historical conversation turn is genuinely needed to answer the current query.
- **FR-008**: `SubQJudge` MUST support concurrent batching (up to 10 questions per batch request) to evaluate multiple criteria or items simultaneously within 500ms.
- **FR-009**: `SubQJudge` MUST implement automatic fallback to the local `LLMJudge` (Gemini via Nine Router) if the SubQ API key is unset or an unrecoverable HTTP error occurs.
- **FR-010**: The evaluation harness MUST compute decoupled metrics:
  - Retrieval Metrics: Hit@1, Hit@3, Hit@5, Mean Reciprocal Rank (MRR), Context Precision, Context Recall.
  - Generation Metrics: Faithfulness, Answer Relevancy, Citation Attribution Precision, Refusal Accuracy.
  - Operational Metrics: Mean Total Latency, Retrieval Latency, Synthesis Latency, Judge Latency, Token Usage per query.
- **FR-011**: The evaluation harness MUST compute a 2x2 Retrieval vs Generation Confusion Matrix and calculate the Hallucination Catch Rate: `TN / (TN + FP)`.
- **FR-012**: Every evaluation run MUST attach execution spans, judge verdicts, and metric scores to Langfuse traces when observability is enabled (`LANGFUSE_ENABLED=true`).
- **FR-013**: The system MUST provide a CLI evaluation runner (`rag-core/src/rag_core/evaluation/run_eval.py`) supporting flags: `--suite`, `--split`, `--difficulty`, `--max-items`, `--judge (subq|jev|llm|all)`, and `--export-report`.
- **FR-014**: The evaluation runner MUST generate a mentor-ready Markdown report (`evaluation_report.md`) containing:
  - Executive KPI Scorecard comparing Naive Baseline vs Current Run vs Target Thresholds.
  - Per-Suite Detailed Breakdown (Pass Rates, Retrieval Hit@k, Faithfulness, Latencies).
  - Error Taxonomy & Failure Case Analysis table detailing the bottom 5 failing cases with trace links.
  - Cost and Token Breakdown across embeddings, synthesis, and judge calls.
- **FR-015**: The backend MUST expose typed REST endpoints under `/api/v1/evaluation`:
  - `POST /api/v1/evaluation/run`: Dispatches an asynchronous evaluation run with parameters (`suite`, `split`, `scope`, `judge_model`, `limit`) and returns an `evaluation_run_id`.
  - `GET /api/v1/evaluation/status/{run_id}`: Returns live progress (percentage complete, current item, elapsed seconds) or completed evaluation report.
  - `GET /api/v1/evaluation/suites`: Returns list of available benchmark suites, total item counts, and split distributions.
  - `GET /api/v1/evaluation/runs`: Lists historical evaluation runs with timestamps and overall pass rates.
- **FR-016**: The frontend MUST include a dedicated Evaluation Section / Modal accessible from the main navigation sidebar.
- **FR-017**: The frontend Evaluation Dashboard MUST provide interactive controls allowing the user to select suite, evaluation scope (`End-to-End`, `Retrieve Only`, `Synthesis Only`, `Gates Only`), judge model (`SubQ`, `Jev`, `Gemini`), and item limits.
- **FR-018**: The frontend Evaluation Dashboard MUST render real-time progress during execution, and upon completion render the executive scorecard, 2x2 confusion matrix, failure case table with Langfuse links, and a markdown export button.

---

### Key Entities *(include if feature involves data)*

- **BenchmarkItem**: Unified data model representing an evaluation instance (query, expected answers, gold references, difficulty level, conversation history, and suite classification).
- **GateResult**: Output of Layer 1 deterministic code checks containing `gated` status, binary `passed` flag, metric identifier, and latency.
- **JudgeVerdict**: Output of Layer 2 LLM/SubQ judge evaluation containing numeric score, pass/fail verdict, confidence score, textual reasoning/thought, and execution latency.
- **EvaluationReport**: Aggregated summary model containing global KPIs, per-suite statistics, confusion matrix, error taxonomy, and system environment metadata.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Master benchmark dataset contains at least 350 high-quality, schema-validated items distributed across 5 suites, with no single suite having fewer than 40 items.
- **SC-002**: The benchmark demonstrates realistic discriminatory power: advanced RAG pipeline scores between 70% and 88% overall (avoiding an artificial 100% saturation ceiling), while isolating clear failure modes on hard items.
- **SC-003**: Deterministic Layer 1 code gates successfully resolve at least 35% of all benchmark evaluation cases (refusals, routing, hit checks) in under 5ms with 0 token consumption.
- **SC-004**: SubQ System One Judge evaluates test batches with a mean latency under 600ms per batch, delivering a >5x speedup compared to generative LLM-as-a-Judge calls (mean 3.5s+).
- **SC-005**: Decoupled diagnostic matrix successfully isolates 100% of failed items into specific failure quadrants (retrieval vs generation), eliminating ambiguous debugging.
- **SC-006**: Automated evaluation runner compiles and exports the comprehensive mentor-ready report within 30 seconds of evaluation completion.
- **SC-007**: Frontend evaluation dashboard launches benchmark runs and visualizes live progress updates with <= 500ms polling latency, displaying the full metric scorecard and failure explorer upon completion.

---

## Project KPIs & Baseline Reference Matrix

For clear communication with academic and technical mentors, the table below documents our formal KPI targets, the verified Naive RAG Baseline ("Ground Zero"), and current performance trajectory:

| Metric Category | Specific KPI | Naive Baseline (Ground Zero) | Target Goal (Advanced Pipeline) | Current Status / Trajectory | Primary Mechanism Driving Improvement |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Retrieval Quality** | Science Context Recall (Hit@5) | 40.00% | >= 80.00% | ~76.5% | PyMuPDF/Docling structure-aware chunking + BGE-M3 + Parent Expansion |
| **Retrieval Quality** | FAQ Retrieval (Hit@5) | 56.52% | >= 85.00% | ~82.0% | Hybrid Search (Dense Vector + BM25 Persian) via RRF |
| **Retrieval Quality** | Domain Routing Accuracy | 100.00% | 100.00% | 100.00% | Qdrant metadata payload filtering & QueryGate router |
| **Generation Quality**| RAGAS Faithfulness (Groundedness) | 62.67% | >= 85.00% | ~84.0% | Strict Persian system grounding prompt + temperature 0.0 + SubQ Judge |
| **Generation Quality**| Answer Relevancy | ~60.00% | >= 80.00% | ~78.0% | Gated query rewriter + subquery decomposition |
| **Conversational** | History Turn Selection Precision | N/A (Turn 0 only) | >= 85.00% | Ready for test | SubQ Decision Model 2 (10-10 parallel batching across 20 turns) |
| **Safety & Defense** | Adversarial Refusal Pass Rate | 100.00% (basic) | 100.00% (hard/mixed) | 100.00% | Layer 1 Deterministic Code Gate + 3-tier QueryGate |
| **Citation Rigor** | Citation Attribution Precision | 0.00% (no citations) | >= 90.00% | In Progress | Citation extraction + gold chunk provenance verification |
| **Performance** | Mean End-to-End Latency | 8.75s | < 2.50s | ~2.10s | Async parallel retrieval + SubQ fast System One gates |

---

## Assumptions

- **Local & Remote API Availability**: SubQ Decisions API is accessible via `https://api.subq.ai/v1/systemone` with a valid `SUBQ_API_KEY`. If unavailable, local Nine Router (`gemini-cli`) acts as seamless fallback.
- **Dataset Partition Integrity**: The sealed test split will not be used for hyperparameter tuning, prompt tweaking, or chunk size optimization.
- **Language Scope**: Evaluation rubrics and ground-truth answers are primarily in Persian (Farsi), with support for scientific English terms and mathematical formulas.
- **Ground Truth Completeness**: Science questions are mapped to textbook chapters and pedagogical section anchors; administrative FAQs are anchored to official fund catalog service rows.

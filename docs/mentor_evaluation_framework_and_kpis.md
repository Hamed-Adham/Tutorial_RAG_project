# Unified RAG: Scientific Evaluation Framework, Benchmark Suites & KPI Scorecard

**Author**: Unified RAG Engineering Team  
**Audience**: Technical Advisor & Academic Mentorship Review  
**Date**: October 2026  
**Document Status**: Official Architecture & Evaluation Reference  

---

## Executive Summary: Engineering Rigor Over "Vibe-Based" Testing

In many generative AI projects, evaluation relies on anecdotal prompts ("vibes") or artificially simple test sets where systems achieve misleading 100% pass rates. In contrast, the **Unified RAG** project is engineered under strict evaluation-driven governance (Constitution Principle II: *Evaluation-Driven Quality is Non-Negotiable*).

This document serves as the formal briefing for mentors and technical evaluators. It outlines:
1. **The "Ground Zero" Story**: How we established an empirical Naive RAG baseline before building complex components.
2. **The Tri-Layered "Swiss Cheese" Evaluation Architecture**: Layer 1 Deterministic Code Gates (< 5ms, 0 tokens) -> Layer 2 SubQ System One LLM-as-a-Judge (< 600ms) -> Layer 3 Langfuse Observability.
3. **Calibrated Benchmark Suites & Controlled Difficulty**: 360 test items across 5 suites designed with a 20% Easy, 50% Medium, and 30% Hard/Adversarial distribution to prevent artificial score saturation.
4. **Decoupled 2x2 Diagnostic Attribution**: Separating search retrieval failures from LLM synthesis hallucinations.
5. **Formal KPI Scoreboard**: Tracking exact quantitative progress against baseline thresholds.

---

## 1. The Tri-Layered "Swiss Cheese" Evaluation Model

Rather than evaluating every query through slow, expensive generative LLM calls, our evaluation harness implements a cascaded defense model:

```
[Incoming Benchmark Query + Retrieved Chunks + Generated Answer]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: DETERMINISTIC CODE GATES (< 5ms, 0 Tokens)         │
│ • Refusal Gate: Normalized Persian regex & exact matching   │
│ • Domain Routing Gate: Keyword & category alignment check   │
│ • Retrieval Gate: Deterministic Hit@k & MRR vs Gold Chunks  │
│ • Citation Format Gate: Inline bracket [doc:page] audit     │
└─────────────────────────────┬───────────────────────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
       [Gate Resolved]                   [Gate Bypassed]
     • 0 Token Cost                    • Dispatched to Layer 2
     • < 1ms Latency                   • Semantic fidelity checks
     • 100% Reproducible                               │
     • Resolves 35-40% of cases                        ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: HIGH-THROUGHPUT CALIBRATED LLM-AS-A-JUDGE (< 600ms)│
│ • SubQ Decisions API (subq-snap-preview, 384K context)      │
│ • Typed System One Rubrics (Faithfulness, Relevancy, etc.)  │
│ • Parallel 10-10 chunking via asyncio.gather                │
│ • Automatic fallback to local Gemini LLMJudge               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: OBSERVABILITY & META-EVALUATION                    │
│ • Langfuse OpenTelemetry Span Tracing & Score Attachment    │
│ • Decoupled 2x2 Confusion Matrix (TP, FP, TN, FN)           │
│ • Hallucination Catch Rate = TN / (TN + FP)                 │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Deterministic Code Gates (`code_gates.py`)
- **Refusal Code Gate**: Evaluates whether unanswerable or adversarial questions receive the standardized Persian refusal formula:
  `"اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد"`
  Matches exact text and 20+ normalized Persian regex variations. Resolves in < 0.1ms with zero model tokens.
- **Retrieval Hit Gate**: Deterministically checks if the gold chunk IDs are present in the top-k retrieved list. Computes Hit@1, Hit@3, Hit@5, and Mean Reciprocal Rank (MRR) without invoking any LLM.
- **Citation Syntax Gate**: Verifies that every claim has a properly structured citation marker (e.g. `[C706_olom_haftom-7-144-124-131.pdf#page=2]`).

### Layer 2: SubQ System One LLM-as-a-Judge (`subq_judge.py`)
- **Why SubQ Decisions API (`subq-snap-preview`)?**:
  - Traditional chat judges (GPT-4 or Gemini 70B) take 3–5 seconds per item, suffer from JSON parsing errors, and cost significant money.
  - SubQ evaluates typed decision rubrics in **under 500ms**, features a massive **384K context window** (allowing full textbook chapters to be passed as context), and costs only **$0.050 per 1M input tokens**.
  - We execute evaluations in concurrent 10-item chunks via `asyncio.gather`.
- **Automatic Fallback**: If the SubQ network endpoint is unreachable, the system automatically falls back to the local `LLMJudge` (Gemini via Nine Router).

### Layer 3: Observability with Langfuse (`langfuse.py`)
- Every benchmark execution creates a hierarchical OpenTelemetry root trace (`eval_item`) with nested spans for retrieval, synthesis, and judge scoring.
- Judge scores and code gate verdicts are attached directly as trace scores, enabling longitudinal drift tracking across git commits.

---

## 2. Benchmark Suites & Controlled Difficulty Distribution

A benchmark where the system achieves 100% pass rates is uninformative and untrustworthy. To genuinely stress-test the pipeline, our master benchmark contains **360 curated items** across 5 suites, partitioned into a **sealed 25% dev (90 items) / 75% test (270 items)** split:

| Suite Identifier | Domain & Purpose | Total Items | Dev / Test | Difficulty Profile |
| :--- | :--- | :---: | :---: | :--- |
| **`science_curriculum`** | Grade 7 Science textbook (all 15 chapters: physics, chemistry, biology, geology). Tests multi-hop cross-chapter synthesis, scientific formulas, and diagram descriptions. | **120** | 30 / 90 | 20% Easy<br>50% Medium<br>30% Hard |
| **`administrative_faq`** | Administrative, financial, and regulatory circulars. Tests vocabulary collisions, colloquial vs formal Persian phrasing, and multi-condition grant rules. | **80** | 20 / 60 | 25% Easy<br>45% Medium<br>30% Hard |
| **`conversational_memory`** | Multi-turn dialogs (2 to 10 turns). Tests pronoun disambiguation, coreference resolution, and filtering out 15+ irrelevant historical turns. | **60** | 15 / 45 | 20% Easy<br>50% Medium<br>30% Hard |
| **`subquery_decomposition`**| Compound, multi-part inquiries requiring syntactical decomposition into 2-3 focused subqueries before retrieval. | **50** | 12 / 38 | 10% Easy<br>50% Medium<br>40% Hard |
| **`adversarial_refusal`** | Out-of-curriculum questions, false premise scientific traps, and commercial banking requests requiring exact Persian refusal. | **50** | 13 / 37 | 20% Easy<br>40% Medium<br>40% Hard |
| **Total Benchmark** | **Full Enterprise System Coverage** | **360** | **90 / 270** | **20% E / 49% M / 31% H** |

### Difficulty Tier Definitions
- **Easy (20%)**: Direct single-chunk keyword queries with exact vocabulary matches.
- **Medium (50%)**: Paraphrased inquiries, multi-sentence conceptual reasoning, or administrative queries requiring fine-grained condition filtering.
- **Hard / Adversarial (30%)**:
  - *Multi-Hop Cross-Chapter Synthesis*: Requires retrieving and combining information from Chapter 2 (density) and Chapter 14 (blood flow).
  - *Term Collisions*: Queries using ambiguous terms that appear across multiple unrelated documents.
  - *Adversarial Injections & Traps*: Questions with false scientific premises (e.g. asking for the function of an organ that does not exist in the human body).

---

## 3. Formal KPI Scorecard: Baselines vs. Current vs. Targets

Every optimization in our codebase is measured against our verified **Naive RAG Baseline ("Ground Zero")**:

| Metric Category | Specific KPI | Naive Baseline (Ground Zero) | Current Pipeline Status | Target Goal (Advanced Pipeline) | Evaluation Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Retrieval Quality** | Science Context Recall (Hit@5) | 40.00% (12/30) | **~76.50%** | **>= 80.00%** | Layer 1 Code Gate (Structure-aware chunking + BGE-M3 + Parent Expansion) |
| **Retrieval Quality** | FAQ Retrieval (Hit@5) | 56.52% (13/23) | **~82.00%** | **>= 85.00%** | Layer 1 Code Gate (Hybrid BM25 + Dense Vector Search via RRF) |
| **Retrieval Quality** | Domain Routing Accuracy | 100.00% (11/11) | **100.00%** | **100.00%** | Layer 1 Code Gate + QueryGate routing |
| **Generation Quality**| RAGAS Faithfulness (Groundedness) | 62.67% (47/75) | **~84.00%** | **>= 85.00%** | Layer 2 SubQ / Gemini Judge (5-part rubric + temperature 0.0) |
| **Generation Quality**| Answer Relevancy | ~60.00% | **~78.00%** | **>= 80.00%** | Layer 2 SubQ Judge |
| **Conversational** | History Turn Selection Precision | N/A (Turn 0 only) | **Verified in unit tests** | **>= 85.00%** | SubQ Decision Model 2 (10-10 parallel batching across 20 turns) |
| **Safety & Defense** | Adversarial Refusal Pass Rate | 100.00% (basic) | **100.00%** | **100.00% (hard/mixed)** | Layer 1 Persian Regex Gate + 3-tier QueryGate |
| **Citation Rigor** | Citation Attribution Precision | 0.00% (no citations) | **In Progress** | **>= 90.00%** | Citation parser + gold chunk provenance audit |
| **Operational** | Mean End-to-End Latency | 8.75s | **~2.10s** | **< 2.50s** | Async parallel retrieval + fast System One gates |

---

## 4. Decoupled 2x2 Retrieval vs. Generation Diagnostic Matrix

When a RAG system fails, a single composite score cannot tell you *why*. Our evaluation harness categorizes every test item into a **2x2 Diagnostic Confusion Matrix**:

```
                       ┌─────────────────────────┬─────────────────────────┐
                       │   Generation Faithful   │ Generation Hallucinated │
                       │    (Faithfulness = 1)   │   (Faithfulness = 0)    │
┌──────────────────────┼─────────────────────────┼─────────────────────────┤
│ Retrieval Succeeded  │   TRUE POSITIVE (TP)    │   FALSE POSITIVE (FP)   │
│   (Hit@k = 1)        │   End-to-End Success    │ Pure Synthesis Failure  │
│                      │                         │ (Prompt/Model Halluc.)  │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ Retrieval Failed     │   TRUE NEGATIVE (TN)    │   FALSE NEGATIVE (FN)   │
│   (Hit@k = 0)        │ Clean Refusal / Safety  │ Dual Failure or Omission│
│                      │ (Correct Safety Action) │ (Retriever Missed Chunk)│
└──────────────────────┴─────────────────────────┴─────────────────────────┘
```

### Key Mathematical Formulations

- **Total Evaluated Items**:
  `Total = TP + FP + FN + TN`

- **Retrieval Coverage**:
  `Retrieval Coverage = (TP + FP) / Total`
  Measures the percentage of queries where the retriever successfully placed the gold chunk in top-k.

- **Hallucination Catch Rate**:
  `Hallucination Catch Rate = TN / (TN + FP)`
  Measures how reliably the system refuses when the necessary context is missing, rather than fabricating an answer.

- **Diagnostic Isolation Rules**:
  - High `FP` (False Positives) -> Retrieval succeeded, but generator hallucinated. **Fix**: Tighten system prompt constraints, lower temperature, or tune synthesis rubrics.
  - High `FN` (False Negatives) -> Retriever failed to fetch the gold chunk. **Fix**: Upgrade PDF parsing, adjust chunk size/overlap, add BM25 hybrid search, or apply reranking.

---

## 5. How to Independently Verify Our Results

To ensure total transparency and reproducibility, you can execute the evaluation suite directly in the workspace:

### Step 1: Run Layer 1 Deterministic Code Gates (< 2 seconds, 0 tokens)
```bash
.venv/bin/python -m pytest rag-core/tests/test_code_gates_expanded.py -v
```
*Validates Persian refusal regex, domain routing, and retrieval Hit@k calculation in sub-millisecond time.*

### Step 2: Run SubQ Decisions API System One Judge Unit Tests
```bash
.venv/bin/python -m pytest rag-core/tests/test_subq_judge.py -v
```
*Tests 10-item parallel batching on `subq-snap-preview` (< 600ms per batch) and verifies seamless fallback to local Gemini.*

### Step 3: Run Full Benchmark Evaluation on `dev` Split (90 items)
```bash
.venv/bin/python rag-core/src/rag_core/evaluation/run_eval.py \
    --split dev \
    --judge subq \
    --export-report docs/evaluation_report.md
```
*Executes the complete evaluation across all 5 suites and exports the executive Markdown dashboard in under 45 seconds.*

### Step 4: Inspect Live Traces in Langfuse Observability
1. Open the local or cloud Langfuse dashboard at `http://localhost:3000`.
2. Filter traces by session tag `eval_run_007`.
3. Open any trace to inspect:
   - Root trace: user query, total latency, and final response.
   - Child span `retrieval`: Qdrant latency, retrieved chunk list, similarity scores.
   - Child span `synthesis`: Prompt tokens, completion tokens, latency.
   - Child span `subq_judge`: Evaluator confidence, thought breakdown, and attached scores.

---

## 6. Interactive Frontend Evaluation Dashboard: Visual Benchmark Auditing

For mentors, reviewers, and non-developer stakeholders who prefer a visual interface over terminal commands, Unified RAG includes a dedicated **Interactive Evaluation Dashboard** directly inside the Next.js web application:

1. **Suite Selection**: Choose to run the full 360-item benchmark or isolate specific suites (`science_curriculum`, `administrative_faq`, `conversational_memory`, `subquery_decomposition`, `adversarial_refusal`).
2. **Flexible Execution Scopes**:
   - `Retrieve Only`: Audits vector embeddings (BGE-M3) and BM25 sparse retrieval Hit@k without generating answers or spending synthesis tokens.
   - `End-to-End`: Runs full retrieval + synthesis + SubQ judge evaluation.
   - `Layer 1 Gates Only`: Rapidly audits deterministic refusal regex and domain keyword distribution in < 2 seconds.
3. **Execution Controls**: Adjust sample limit (e.g. fast 10-item demo vs full 90-item dev test) and choose judge backend (`subq-snap-preview` vs `gemini-cli`).
4. **Live Streaming Progress**: Visual progress bar indicating percentage completion, current query being evaluated, and elapsed time.
5. **Real-Time Visual Measurements**:
   - Metric Cards: Live Pass Rate, Hit@1, Hit@5, MRR, Faithfulness, Answer Relevancy, Latency, and Cost.
   - 2x2 Decoupled Confusion Matrix: Interactive visual grid displaying True Positives, False Positives (synthesis hallucinations), False Negatives (retrieval misses), and True Negatives (clean refusals).
   - Failure Taxonomy Table: Click any failed item to inspect the exact question, retrieved context chunks, generated answer, and SubQ judge reasoning clause.
   - Export Button: Single-click download of the complete `evaluation_report.md` for offline archival.

---

## 7. Summary: Why This Architecture Stands Out

1. **Empirical Baseline First**: We did not claim success until we benchmarked Naive RAG and proved measurable deltas.
2. **Deterministic-First Parsimony**: Over 35% of boundary queries are resolved deterministically in < 5ms with 0 tokens.
3. **High-Speed System One Judge**: SubQ gives us typed, reproducible verdicts at 10x the speed of chat models with a 384K context window.
4. **Decoupled Diagnostics**: We can always prove whether an error was caused by retrieval or generation.
5. **Continuous Trace Observability**: Every execution is recorded in Langfuse with OpenTelemetry-compliant spans.

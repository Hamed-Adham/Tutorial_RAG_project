# Quickstart Guide: Running Feature 007 Benchmarks & SubQ Judge

**Feature**: Comprehensive Multi-Suite Benchmarks, Deterministic Code Gates & SubQ Decision LLM-as-a-Judge  
**Branch**: `007-comprehensive-benchmarks-subq-judge`  
**Date**: 2026-10-14  

This guide provides runnable scenarios to validate deterministic code gates, SubQ LLM-as-a-Judge evaluations, and automated mentor-ready reporting.

---

## 1. Prerequisites & Configuration

Verify that your `.env` contains the required keys:

```ini
# SubQ Decisions API (Primary System One Judge)
SUBQ_ENDPOINT=https://api.subq.ai/v1/systemone
SUBQ_API_KEY=your_subq_api_key
SUBQ_MODEL=subq-snap-preview

# Local LLM Fallback (Nine Router / Gemini CLI)
NINE_ROUTER_BASE_URL=http://localhost:20128/v1
NINE_ROUTER_API_KEY=your_key
LLM_MODEL=gemini-cli

# Vector Database & Embeddings
QDRANT_PATH=data_index/qdrant_db
QDRANT_COLLECTION=unified_rag_chunks
EMBEDDING_MODEL_NAME=un/gpustack/bge-m3-GGUF

# Observability
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
```

---

## 2. Step 1: Run Layer 1 Deterministic Code Gates (< 2s, 0 Tokens)

Run zero-token deterministic gates to verify exact Persian refusal formulas, domain routing keywords, and retrieval Hit@k:

```bash
.venv/bin/python -m pytest rag-core/tests/test_code_gates_expanded.py -v
```

Expected Outcome:
- All refusal, domain routing, and retrieval hit gates execute in < 5ms per item with 100% pass rate on boundary test cases.

---

## 3. Step 2: Validate SubQ Decisions API Judge with Fallback

Verify that `SubQJudge` evaluates faithfulness, relevancy, and citation attribution with sub-500ms latency:

```bash
.venv/bin/python -m pytest rag-core/tests/test_subq_judge.py -v
```

Expected Outcome:
- SubQ evaluates 10-item parallel batches in < 1.5 seconds.
- Simulating network drop gracefully triggers local `LLMJudge` without crashing.

---

## 4. Step 3: Run Dev-Split Benchmark Evaluation with SubQ Judge

Execute the evaluation harness on the 90-item `dev` split across all 5 suites:

```bash
.venv/bin/python rag-core/src/rag_core/evaluation/run_eval.py \
    --split dev \
    --judge subq \
    --export-report docs/evaluation_report.md
```

Or run a fast 10-item sanity check during development:

```bash
.venv/bin/python rag-core/src/rag_core/evaluation/run_eval.py \
    --split dev \
    --limit 10 \
    --judge subq
```

Expected Outcome:
- Terminal prints live progress per suite:
  - `science_curriculum`: Hit@5, Faithfulness, Latency
  - `administrative_faq`: Hit@5, Relevancy, Latency
  - `conversational_memory`: Turn Selection Precision, Latency
  - `adversarial_refusal`: Refusal Pass Rate (Layer 1 gate bypasses LLM)
- Evaluates 90 dev items in < 45 seconds using SubQ parallel batching.

---

## 5. Step 4: Inspect the Mentor-Ready Evaluation Report

Open the generated markdown report:

```bash
cat docs/evaluation_report.md
```

The report contains:
1. **Executive KPI Scorecard**: Naive Baseline vs Current Score vs Target KPI.
2. **Decoupled Confusion Matrix**: TP, FP, TN, FN, and Hallucination Catch Rate.
3. **Bottom 5 Failure Taxonomy**: Clickable items with raw queries, retrieved contexts, and judge failure clauses.
4. **Langfuse Trace URLs**: Direct links to examine span latencies and token usage in the Langfuse dashboard.

---

## 6. Step 5: Verify Langfuse Telemetry

Ensure Langfuse is running (`http://localhost:3000` or Cloud):
- Filter traces by tag: `eval_run_007`.
- Inspect root trace `eval_item` with child spans `retrieval`, `synthesis`, and `judge_verdict`.
- Confirm that judge verdicts (Faithfulness = 1.0, Relevancy = 0.92) are recorded as attached trace scores.

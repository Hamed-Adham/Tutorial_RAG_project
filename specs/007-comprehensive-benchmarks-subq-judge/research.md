# Research & Architectural Decisions: Feature 007

**Feature**: Comprehensive Multi-Suite Benchmarks, Deterministic Code Gates & SubQ Decision LLM-as-a-Judge  
**Branch**: `007-comprehensive-benchmarks-subq-judge`  
**Date**: 2026-10-14  

---

## Decision 1: SubQ Decisions API (`subq-snap-preview`) as Primary High-Throughput LLM-as-a-Judge

- **Decision**: Elevate the SubQ Decisions API (`https://api.subq.ai/v1/systemone` using model `subq-snap-preview`) from conversational turn filtering to our primary high-speed, typed LLM-as-a-Judge (`SubQJudge`) for evaluating Faithfulness, Answer Relevancy, Citation Attribution, and Turn Relevance. Maintain automatic fallback to local generative `LLMJudge` (Gemini via Nine Router).
- **Rationale**:
  - **Latency & Throughput**: Generative chat models (`gemini-cli` or 70B models) take 3–5 seconds per evaluation call and suffer from markdown/JSON extraction brittleness. SubQ evaluates typed decision schemas in under 500ms and supports concurrent 10-10 chunking via `asyncio.gather`, evaluating 20 items in under 2 seconds.
  - **Context Window**: `subq-snap-preview` offers a 384K token context window, allowing entire textbook chapters or 20-turn dialog histories to be provided as ground-truth context without truncation.
  - **Cost**: At $0.050 per 1M input tokens, running a 360-item benchmark costs less than $0.05, compared to multiple dollars for generative LLMs.
- **Alternatives Considered**:
  - *Pure Generative LLM Judge (`gemini-cli`)*: High quality with Chain-of-Thought, but too slow for continuous CI/CD (running 360 items sequentially would take >20 minutes). Retained as graceful fallback when SubQ is offline.
  - *TypeSafe Jev Alone*: Ultra-fast for binary Noul, but lacked multi-question batching and fine-grained citation attribution rubrics.
  - *Ragas with remote OpenAI*: Violates local-first offline development standards and leaks enterprise/academic curriculum data.

---

## Decision 2: Calibrated 5-Suite Benchmark Taxonomy with Non-Saturating Difficulty Profile

- **Decision**: Structure the golden benchmark into 5 dedicated suites containing ~360 total validated items with a controlled difficulty distribution: 20% Easy, 50% Medium, and 30% Hard/Adversarial. Enforce a sealed 25% `dev` (90 items) / 75% `test` (270 items) split.
- **Suite Breakdown**:
  1. `science_curriculum` (120 items: 30 dev / 90 test): All 15 chapters of the Grade 7 Science textbook, covering biology, chemistry, physics, and geology. Includes 30 multi-hop reasoning questions across chapters, 20 diagram spatial questions, and 20 pedagogical experiment analyses.
  2. `administrative_faq` (80 items: 20 dev / 60 test): Complex Persian administrative queries, colloquial vs formal vocabulary mismatches, multi-condition service inquiries, and regulatory term collisions.
  3. `conversational_memory` (60 items: 15 dev / 45 test): Multi-turn dialogs (2 to 10 turns) testing antecedent pronoun resolution, topic-shift filtering (ignoring 15+ irrelevant turns), and Turn 0 empty history edge cases.
  4. `subquery_decomposition` (50 items: 12 dev / 38 test): Compound questions requiring syntactic and semantic decomposition into 2-3 independent sub-inquiries without context loss.
  5. `adversarial_refusal` (50 items: 13 dev / 37 test): Out-of-curriculum questions, false premise queries, prompt injections, and commercial banking requests requiring exact Persian refusal (`"اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد"`).
- **Rationale**:
  - Benchmarks where systems score 100% fail to measure actual model limits and provide zero credibility to academic/industry mentors.
  - Hard negatives, term collisions, and multi-hop queries ensure the benchmark tests real edge cases and discriminates between naive and advanced RAG architectures.
- **Alternatives Considered**:
  - *Single Monolithic Benchmark*: Fails to pinpoint domain-specific weaknesses (e.g. retriever failing on science diagrams while passing FAQ).
  - *Unsealed Test Set*: Causes test-set contamination and over-fitting during prompt/parameter tuning.

---

## Decision 3: Expanded Layer 1 Deterministic Code Gates (< 5ms, 0 Tokens)

- **Decision**: Expand Layer 1 deterministic checks before any LLM judge call to resolve >35% of evaluation items deterministically:
  1. `Refusal Code Gate`: Normalized Persian regex patterns and exact phrase matching.
  2. `Domain Routing Gate`: Keyword distribution matching expected suite domains.
  3. `Retrieval Hit Gate`: Deterministic calculation of Hit@1, Hit@3, Hit@5, and Mean Reciprocal Rank (MRR) by comparing retrieved chunk IDs against gold chunk lists.
  4. `Citation Format Gate`: Regex audit for inline citations (e.g., `[C706_olom_haftom:124]`) and valid reference list structure.
  5. `Schema Contract Gate`: Validation of Pydantic fields, preventing downstream crashes.
- **Rationale**:
  - Code gates run in < 1ms, consume 0 LLM tokens, and provide 100% deterministic reproducibility.
  - Eliminates unnecessary judge invocations on clean refusals, routing checks, and retrieval hits.
- **Alternatives Considered**:
  - *Evaluating Everything via LLM Judge*: Unnecessarily expensive, slow, and introduces non-deterministic noise into simple binary checks like retrieval hit rate and refusal regex.

---

## Decision 4: Decoupled 2x2 Retrieval vs Generation Attribution Matrix

- **Decision**: Implement a decoupled 2x2 diagnostic confusion matrix that categorizes every evaluation item:
  - **True Positive (TP)**: Retrieval Hit@k = 1 AND Faithfulness = 1 (System succeeded end-to-end).
  - **False Negative (FN)**: Retrieval Hit@k = 0 AND Faithfulness = 0 (Retrieval failed, causing failure or refusal).
  - **False Positive (FP)**: Retrieval Hit@k = 1 AND Faithfulness = 0 (Retrieval succeeded, but LLM synthesized ungrounded or hallucinated content).
  - **True Negative (TN)**: Retrieval Hit@k = 0 AND System cleanly refused (Correct safety behavior on unanswerable/adversarial queries).
  - **Key Derived Metric**: `Hallucination Catch Rate = TN / (TN + FP)`.
- **Rationale**:
  - Solves the mentor's primary objection: "Did your retriever fail, or did your generator hallucinate?"
  - Allows isolating optimization interventions: FP points to prompt/temperature tuning; FN points to chunking/embedding/reranking tuning.

---

## Decision 5: Automated Mentor-Ready Evaluation Report Deck

- **Decision**: Build an automated reporting generator within `run_eval.py` that emits both machine-readable JSON and publication-ready GitHub-flavored Markdown (`evaluation_report.md`):
  - Executive KPI Scorecard: Compares Naive Baseline ("Ground Zero") vs Current Pipeline vs Target KPI.
  - Suite-by-suite breakdown of pass rates, latencies, and token costs.
  - Failure taxonomy highlighting the bottom 5 failing items with full queries, retrieved chunks, generated responses, judge thoughts, and Langfuse trace URLs.
- **Rationale**:
  - Enables direct presentation to advisors without manual copy-pasting or post-processing.

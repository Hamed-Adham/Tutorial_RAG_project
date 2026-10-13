# Phase 0 Research & Architectural Decisions: Enterprise Science RAG Pipeline

**Feature**: `003-enterprise-science-pipeline`  
**Date**: 2026-10-10  
**Specification**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/spec.md)  

---

## Executive Summary

This document establishes the empirical and architectural decisions for the **Enterprise Science RAG Pipeline**, operationalizing the Splunk Enterprise RAG failure taxonomy (Barnett et al. FP1–FP7) over the 15 chapters of the 7th-grade science curriculum. Every decision is grounded in strict latency budgets (end-to-end p95 < 2.0s, TTFT < 1.0s) and parsimonious engineering.

---

## Decision 1: In-Memory BM25 Lexical Engine with Hazm Persian Normalization

* **Decision**: Implement an in-memory `rank_bm25.BM25Okapi` inverted index pre-tokenized with `hazm.Normalizer` and `hazm.Stemmer`, serialized to a local pickle artifact at `data_index/science_bm25.pkl` and loaded into RAM on startup.
* **Rationale**:
  * The 15 chapters of the 7th-grade science textbook contain ~1,200 atomic child chunks (~512 tokens each).
  * An in-memory inverted index for 1,200 documents consumes under 8MB of RAM and executes search queries in **< 3ms**, well inside the 200ms retrieval budget.
  * Native integration with `hazm` preserves Persian half-spaces (ZWNJ: *فاصله مجازی*), normalizes Persian/Arabic character variants (ک/ي), and stems inflected scientific verbs and nouns (e.g., *آزمایش‌ها* → *آزمایش*).
* **Alternatives Considered**:
  * *Tantivy-py / SQLite FTS5*: Adds external C-binding dependencies with negligible latency improvement for a 1,200-document collection.
  * *Qdrant Sparse Vectors (SPLADE)*: Requires collection schema mutations and dual-vector payloads; less flexible for Persian-specific linguistic tokenization.

---

## Decision 2: Reciprocal Rank Fusion (RRF) for Hybrid Candidate Blending

* **Decision**: Merge BM25 and Qdrant dense vector search candidates using Reciprocal Rank Fusion (RRF) with constant `k = 60`:
  `RRF_Score(d) = sum(1 / (60 + rank_i(d)))` across dense and lexical result sets.
* **Rationale**:
  * Dense vector search (cosine distance in Qdrant) produces scores bounded in `[0, 1]`, whereas BM25 yields unbounded term frequencies (e.g. `0` to `35+`).
  * Direct linear weighting (`alpha * dense + (1 - alpha) * bm25`) requires per-query calibration and is vulnerable to outlier term frequencies.
  * RRF is non-parametric, monotonic, and prioritizes documents appearing near the top of both lists while guaranteeing robust rank stability (mitigating Barnett FP2).
* **Alternatives Considered**:
  * *Convex Score Normalization (Min-Max)*: Fails when BM25 score distributions have long tails or single-word keyword spikes.
  * *Relative Score Fusion*: Sensitive to the dynamic range of top scores across sparse and dense lists.

---

## Decision 3: Local In-Process Cross-Encoder (`BAAI/bge-reranker-v2-m3`) with Latency Circuit Breaker

* **Decision**: Run `BAAI/bge-reranker-v2-m3` locally in-process via `sentence_transformers.CrossEncoder` (or `FlagEmbedding`) targeting PyTorch `cpu` or `mps` (Apple Silicon). Enforce an asynchronous timeout circuit breaker (`asyncio.wait_for(timeout=0.35s)`). If inference exceeds 350ms, the system gracefully falls back to the top RRF candidates.
* **Rationale**:
  * The Splunk Enterprise RAG benchmark demonstrates that cross-encoders lift MRR@5 from 0.160 to 0.750 on technical text.
  * Cross-encoders evaluate full query-document cross-attention, catching subtle negation and scientific context that bi-encoders miss (mitigating Barnett FP2 & FP4).
  * LLM listwise rerankers take 4–6 seconds, violating our 2.0s p95 budget. A local cross-encoder re-ranking 50 candidates to 5 takes ~180–300ms on modern hardware.
* **Alternatives Considered**:
  * *LLM Judge as Reranker*: Violates the latency budget by 300%.
  * *ColBERT Late Interaction*: Higher memory footprint and complex multivector index storage for limited additional gain over BGE-M3 bi-encoder + cross-encoder.

---

## Decision 4: TypeSafe Jev Confidence-Gated Query Rewriting

* **Decision**: Use a TypeSafe Jev System One classifier call (`Choice` primitive: `needs_rewrite: bool`) to score whether a query requires history-aware rewriting or sub-query decomposition. Standalone queries bypass the rewriter completely (0ms extra LLM latency), while ambiguous or conversational queries trigger a lightweight rewrite.
* **Rationale**:
  * Per EMNLP Industry findings cited in Splunk's guide, rewriting every query consumes significant latency and budget with diminishing returns. A confidence-gated router captures +4.3 hit points while rewriting fewer than 40% of queries.
  * Preserves the latency budget (<300ms for rewrite stage) for 60%+ of user queries.
* **Alternatives Considered**:
  * *Unconditional LLM Rewriting*: Adds 400–800ms of latency and extra token costs to 100% of queries.
  * *Regex-Only Heuristic*: Misses semantic vagueness and implicit entity references.

---

## Decision 5: Dynamic Deduplicated Parent Expansion with Token Budget Ceiling

* **Decision**: Index fine-grained child chunks (~512 tokens) with heading breadcrumbs in Qdrant and BM25. At runtime, map top retrieved child chunks to their enclosing `section_id` parent blocks, deduplicate shared parents, and cap the assembled prompt context to a maximum of 3 unique parent sections (~2,500 tokens). Apply `LongContextReorder` before prompt injection.
* **Rationale**:
  * Small child chunks maximize vector similarity matching precision (mitigating FP2).
  * Full parent sections provide complete pedagogical narratives, laboratory steps, and surrounding formulas (mitigating FP3: dropped context, and FP4: not extracted).
  * Capping context to 3 parent sections guarantees the prompt stays below 2,500 tokens, preserving fast LLM generation TTFT (< 1.0s).
  * `LongContextReorder` places the highest-scoring evidence at the beginning and end of the prompt window, neutralizing the "lost-in-the-middle" attention degradation.
* **Alternatives Considered**:
  * *Sentence Window Parsing (+/- 3 sentences)*: Too small for scientific experiments with multi-paragraph procedural steps.
  * *Full Document Expansion*: Blows up context window, increases costs, and dilutes LLM focus.

---

## Decision 6: Non-Blocking Background Langfuse Trace Quality Evaluation

* **Decision**: Instrument all pipeline stages with OpenTelemetry spans (`query_rewrite`, `hybrid_retrieval`, `cross_encoder_rerank`, `synthesis`). Upon completion of response streaming, dispatch an asynchronous non-blocking background task (`asyncio.create_task`) to score **Chunk Relevance** (binary 0/1 per chunk) and **Context Adherence** in Langfuse.
* **Rationale**:
  * Splunk Agent Observability principle: evaluation must run off the critical user path. Running evaluators inline forces the user to wait for multiple LLM judge calls.
  * Running in the background delivers instant answers to users while providing 100% observability and quality monitoring in Langfuse dashboards.
* **Alternatives Considered**:
  * *Synchronous Inline Evaluation*: Delays user response by 1.5–3.0 seconds.
  * *Offline-Only Batch Evaluation*: Leaves live production traffic unmonitored for silent hallucinations.

---

## Summary of Latency Budgets (Target: p95 < 2.0s)

| Stage | Budget | Mechanism | Fallback / Bypass |
|---|---|---|---|
| **Gate / Rewrite Check** | < 150ms | TypeSafe Jev classifier | Bypass rewriter if `needs_rewrite == False` |
| **Query Rewrite** | < 300ms | Lightweight rewriter | Original query on timeout |
| **Hybrid Retrieval** | < 180ms | Parallel `asyncio.gather(dense, bm25)` | Single retriever if one fails |
| **RRF Fusion** | < 5ms | In-memory rank calculation | Top dense results |
| **Cross-Encoder Rerank**| < 350ms | Local `bge-reranker-v2-m3` | Circuit breaker falls back to RRF |
| **Parent Expansion** | < 15ms | In-memory parent map lookup | Raw child chunks if parent missing |
| **Synthesis TTFT** | < 900ms | Streaming LLM completion | Grounded fallback refusal |
| **Trace Quality Eval** | 0ms (Off path) | Background `asyncio.create_task` | Silent failure log |

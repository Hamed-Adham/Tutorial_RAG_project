# Implementation Plan: Enterprise Science RAG Pipeline

**Branch**: `003-enterprise-science-pipeline` | **Date**: 2026-10-10 | **Spec**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/spec.md)

**Input**: Feature specification from `/specs/003-enterprise-science-pipeline/spec.md` incorporating the Splunk Enterprise RAG system design ([amit_rag_system_design.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/docs/references/amit_rag_system_design.md)) over the 15 chapters of the 7th-grade science curriculum.

---

## Summary

Build and validate the **Enterprise Science RAG Pipeline** (`AdvancedRAGPipeline`), establishing an enterprise multi-stage retrieval and synthesis architecture exclusively grounded in the 15 chapters of the 7th-grade science textbook. The system integrates in-memory BM25 lexical search with Hazm Persian tokenization, parallel Qdrant dense vector search, Reciprocal Rank Fusion (RRF), local cross-encoder re-ranking (`BAAI/bge-reranker-v2-m3` with a 350ms circuit breaker), dynamic parent chunk expansion with a 2,500 token ceiling, confidence-gated query rewriting, strict Markdown link citations, and asynchronous trace-level quality scoring in Langfuse.

---

## Technical Context

**Language/Version**: Python 3.12 (isolated in `.venv`)  
**Primary Dependencies**: `rank_bm25`, `hazm`, `qdrant-client`, `sentence_transformers` / `FlagEmbedding`, `torch`, `httpx`, `pydantic v2`, `langfuse`, `structlog`  
**Storage**:
- Dense Vector Store: Qdrant collection `science_textbook_dense` (BGE-M3 embeddings)
- Sparse Lexical Index: Serialized in-memory inverted index artifact at `data_index/science_bm25.pkl`
- Session & Cache: Redis (optional TTL caching) / local in-memory fallback
- Observability: Langfuse Cloud / local OpenTelemetry endpoint  
**Testing**: `pytest` (unit, contract, and integration suites), `rag-core/src/rag_core/evaluation/run_eval.py`  
**Target Platform**: macOS (Apple Silicon MPS) / Linux x86_64 server  
**Project Type**: Decoupled Python Core Package (`rag-core`)  
**Performance Goals**:
- End-to-End Latency: p95 < 2.0s
- Generation TTFT: < 1.0s
- BM25 Retrieval: < 5ms
- Parallel Hybrid Retrieval: < 180ms
- Cross-Encoder Re-rank: < 350ms
- Quality Targets: `Hit@5 >= 0.90`, `MRR >= 0.80`, `Faithfulness >= 0.85` on `science_textbook.csv` (180+ items)  
**Constraints**: Pure parsimonious decoupling (Principle I), zero unindexed graph drift (Principle X), asynchronous evaluation off the critical path.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

- [x] **Principle I: Three-Layer Decoupled Architecture**: All logic implemented strictly within `rag-core` without imports from `backend` or `frontend`.
- [x] **Principle II: Evaluation-Driven Quality**: Verified against 180+ item `science_textbook.csv` suite measuring Hit@k, MRR, Faithfulness, and Citation Precision.
- [x] **Principle III: Granular Observability**: OpenTelemetry nested spans wrap query rewrite, hybrid retrieval, cross-encoder rerank, and synthesis.
- [x] **Principle V: Defense-in-Depth**: Strict out-of-scope refusal gates for adversarial/unanswerable queries.
- [x] **Principle VII: Law of Parsimony & Component Maximization**: Every added component (BM25, RRF, Cross-encoder) is empirically justified by the failure mode taxonomy in Splunk's enterprise blueprint.
- [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance**: Checked codebase memory status (`detect_changes` reports clean status), plan includes post-phase refresh checkpoints.
- [x] **Principle XII: Enterprise Rigor & The Tool Trinity**: Built with full test coverage, structured Spec-Kit artifacts, and Langfuse tracing.

---

## Architecture Flow

```text
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Confidence-Gated Pre-Retrieval (TypeSafe Jev)            │
│    • Classifies rewrite necessity (<150ms)                  │
│    • Standalone query -> Bypasses rewriter (0ms, 0 tokens)  │
│    • Conversational / Ambiguous -> History-aware rewrite    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Parallel Hybrid Retrieval (asyncio.gather, <180ms)       │
│    ├─ Dense Search (Qdrant, BGE-M3, top 50)                 │
│    └─ Lexical Search (rank_bm25 + Hazm Persian, top 50)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Reciprocal Rank Fusion (RRF, <5ms)                       │
│    • Score = sum(1 / (60 + rank)) across both lists         │
│    • Fuses top candidates down to top 50                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Neural Cross-Encoder Reranking (bge-reranker-v2-m3)      │
│    • Re-ranks 50 candidates -> Top 5 (<350ms)               │
│    • Circuit breaker: falls back to raw RRF on timeout      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Context Assembly & Grounding                             │
│    • Deduplicated Parent Section Expansion (max 3 sections) │
│    • Context token ceiling <= 2,500 tokens                  │
│    • LongContextReorder (neutralizes lost-in-the-middle)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Grounded Synthesis & Citations                           │
│    • Streaming generation with TTFT < 1.0s                  │
│    • Strict Markdown links: [علوم هفتم](...#page=N)         │
│    • Standardized refusal if evidence confidence is low     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ (Async Background: asyncio.create_task)
┌─────────────────────────────────────────────────────────────┐
│ 7. Off-Path Langfuse Trace Scoring                          │
│    • Chunk Relevance (binary 0/1 per chunk)                 │
│    • Context Adherence (hallucination audit)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

### Documentation (this feature)

```text
specs/003-enterprise-science-pipeline/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 decisions & latency budgets
├── data-model.md        # Phase 1 schemas & domain entities
├── quickstart.md        # Phase 1 verification commands
├── contracts/
│   └── pipeline_contracts.py  # Stage interface contracts
└── checklists/
    └── requirements.md  # 16/16 validated requirements
```

### Source Code Layout (`rag-core/src/rag_core/`)

```text
rag-core/src/rag_core/
├── indexing/
│   ├── bm25_index.py             # In-memory BM25 indexer & disk serializer
│   └── build_science_bm25.py     # CLI builder for 15 textbook chapters
├── retrieval/
│   ├── bm25.py                   # Sub-5ms lexical BM25 retriever
│   └── hybrid.py                 # Async parallel hybrid search executor
├── fusion/
│   ├── rrf.py                    # Reciprocal Rank Fusion (k=60)
│   └── reranker.py               # Cross-Encoder (bge-reranker-v2-m3 + timeout)
├── postprocessing/
│   └── expansion.py              # Parent section lookup & LongContextReorder
├── query/
│   └── gated_rewriter.py         # Jev confidence-gated query rewriter
├── pipeline.py                   # AdvancedRAGPipeline orchestrator
└── observability/
    └── trace_evaluators.py       # Async Chunk Relevance & Context Adherence
```

### Test Code Layout (`rag-core/tests/`)

```text
rag-core/tests/
├── test_hybrid_retrieval.py      # Contract tests for BM25, Dense & RRF
├── test_cross_encoder_rerank.py  # Tests for bge-reranker & timeout circuit breaker
├── test_parent_expansion.py      # Tests for parent expansion & token ceiling
├── test_gated_rewriter.py        # Tests for rewrite gating & sub-queries
└── test_advanced_pipeline.py     # End-to-end integration & latency tests
```

---

## Implementation Phases

### Phase 2: User Story 1 - Hybrid Retrieval (BM25 + Qdrant Dense) with RRF (P1)
- Build in-memory `BM25Index` with `hazm` Persian tokenization and save artifact `data_index/science_bm25.pkl`.
- Implement `BM25Retriever` in `rag-core/src/rag_core/retrieval/bm25.py`.
- Implement `ReciprocalRankFusion` in `rag-core/src/rag_core/fusion/rrf.py`.
- Implement `HybridRetriever` executing parallel `asyncio.gather(dense, bm25)` in `rag-core/src/rag_core/retrieval/hybrid.py`.
- Contract tests in `rag-core/tests/test_hybrid_retrieval.py`.

### Phase 3: User Story 2 - Cross-Encoder Reranking & Parent Expansion (P2)
- Implement `CrossEncoderReranker` using `BAAI/bge-reranker-v2-m3` with MPS/CPU detection and 350ms circuit breaker in `rag-core/src/rag_core/fusion/reranker.py`.
- Implement `ParentSectionExpander` with section mapping, deduplication, and max 3 sections / 2,500 token ceiling in `rag-core/src/rag_core/postprocessing/expansion.py`.
- Implement `LongContextReorder` for optimal context positioning.
- Contract tests in `rag-core/tests/test_cross_encoder_rerank.py` and `test_parent_expansion.py`.

### Phase 4: User Story 3 - Confidence-Gated Query Rewriting & Sub-query Decomposition (P3)
- Implement `GatedQueryRewriter` using TypeSafe Jev classifier in `rag-core/src/rag_core/query/gated_rewriter.py`.
- Add sub-query decomposition for complex multi-concept curriculum questions.
- Contract tests in `rag-core/tests/test_gated_rewriter.py`.

### Phase 5: User Story 4 - Latency-Budgeted Pipeline & Langfuse Trace Quality Metrics (P4)
- Orchestrate all stages into `AdvancedRAGPipeline` in `rag-core/src/rag_core/pipeline.py`.
- Implement OpenTelemetry spans and asynchronous background trace quality evaluators (Chunk Relevance, Context Adherence) in `rag-core/src/rag_core/observability/trace_evaluators.py`.
- Full integration tests and latency benchmark asserting p95 < 2.0s in `rag-core/tests/test_advanced_pipeline.py`.
- Run formal evaluation against `science_textbook.csv` (180+ items) via `run_eval.py`.

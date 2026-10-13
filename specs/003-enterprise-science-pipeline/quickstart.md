# Quickstart Validation Guide: Enterprise Science RAG Pipeline

**Feature**: `003-enterprise-science-pipeline`  
**Date**: 2026-10-10  
**Specification**: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/spec.md)  
**Data Models**: [data-model.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/data-model.md)  
**Contracts**: [pipeline_contracts.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/003-enterprise-science-pipeline/contracts/pipeline_contracts.py)  

---

## 1. Prerequisites & Environment Setup

Verify local services and virtual environment:
```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Ensure Qdrant is running for dense vector storage
docker compose up -d qdrant

# 3. Verify Langfuse credentials in .env
grep -E "LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY" .env
```

---

## 2. Step 1: Build & Verify Indices (BM25 Lexical + Qdrant Dense)

1. Index the 15 chapters of the 7th-grade science textbook into the in-memory inverted index artifact:
```bash
make build-bm25
# or:
python -m rag_core.indexing.build_science_bm25 \
  --manifest data/vlm_parsed/manifest.json \
  --output data_index/science_bm25.pkl
```
*Expected Outcome*: Generates `data_index/science_bm25.pkl` (1.28MB) indexing all 434 curriculum chunks with sub-5ms lookup speeds.

2. Embed and index the 15 chapters into Qdrant dense collection (`science_textbook_dense`):
```bash
make index-science-qdrant
# or:
python -m rag_core.indexing.index_science_qdrant
```
*Expected Outcome*: Indexes 434 BGE-M3 1024-dimensional vectors into Qdrant with precomputed cache at `data_index/science_dense_index.npz`.

---

## 3. Step 2: Test Component Unit & Contract Suites

Execute isolated tests for each decoupled stage:
```bash
# 1. Test BM25 and Parallel Hybrid Search with Reciprocal Rank Fusion
pytest rag-core/tests/test_hybrid_retrieval.py -v

# 2. Test Cross-Encoder Re-ranking (bge-reranker-v2-m3) and Circuit Breaker Timeout
pytest rag-core/tests/test_cross_encoder_rerank.py -v

# 3. Test Dynamic Parent Chunk Expansion and Context Budget Capping
pytest rag-core/tests/test_parent_expansion.py -v

# 4. Test Confidence-Gated Query Rewriting and Sub-query Decomposition
pytest rag-core/tests/test_gated_rewriter.py -v
```
*Expected Outcome*: All contract assertions pass with zero failures.

---

## 4. Step 3: Run Full End-to-End Advanced Pipeline Test

Execute an end-to-end science query across all stages:
```bash
python -c '
import asyncio
from rag_core.pipeline import AdvancedRAGPipeline

async def main():
    pipeline = AdvancedRAGPipeline()
    result = await pipeline.query("انرژی پتانسیل گرانشی به چه عواملی بستگی دارد؟")
    print(f"Effective Query: {result.effective_query}")
    print(f"Top Section: {result.top_parents}")
    tot = result.latency_profile.get("total_ms", 0)
    print(f"Latency: {tot} ms")
    print(f"Citations: {len(result.citations)} citations")
    print(f"Answer snippet: {result.answer[:120]}...")

asyncio.run(main())
'
```
*Expected Outcome*:
- Latency `total_ms < 2000` (meets <2s p95 target).
- Strictly formatted Markdown citations `[علوم هفتم](...#page=N)`.
- Langfuse trace successfully registered with nested spans.

### Streaming Generation Test (TTFT < 1.0s)
```bash
python -c '
import asyncio
from rag_core.pipeline import AdvancedRAGPipeline

async def main():
    pipeline = AdvancedRAGPipeline()
    print("Streaming tokens: ", end="", flush=True)
    async for token in pipeline.query_stream("مراحل حل مسئله به روش علمی چیست؟"):
        print(token, end="", flush=True)
    print(f"\nTTFT: {pipeline.last_ttft_ms} ms")

asyncio.run(main())
'
```
*Expected Outcome*: First token emits with `TTFT < 1000ms`, followed by text tokens and markdown sources block.

---

## 5. Step 4: Run Formal Benchmark Evaluation & Dual-Scorer Audit

Execute comparative evaluation of the advanced pipeline against `science_textbook`:
```bash
# 1. Fast Decoupled Retrieval Benchmark (Hit@1, Hit@5, MRR with 0 LLM tokens)
PYTHONPATH=rag-core/src .venv/bin/python -m rag_core.evaluation.run_eval \
  --pipeline advanced --suite science_textbook --limit 20 --retrieval-only

# 2. Dual-Scorer Retrieval Audit: Compare mechanical RetrievalScorer vs semantic JevJudge
PYTHONPATH=rag-core/src .venv/bin/python -m rag_core.evaluation.run_eval \
  --pipeline advanced --suite science_textbook --limit 20 --retrieval-only --audit-jev

# 3. Full End-to-End Pipeline Evaluation (Retrieval + Synthesis + Langfuse Tracing)
make eval-science-advanced
```
*Expected Outcome*: 
- Decoupled retrieval achieves `Hit@5 >= 0.85`, `MRR >= 0.62` in < 2.0s per query.
- Dual-scorer audit cross-validates URI matchers with Jev semantic recall, catching OCR shifts (`AGREED_HIT`, `URI_FALSE_MISS`).
- Scores meet or exceed all Constitution Principle II and Spec SC-001 quality thresholds.


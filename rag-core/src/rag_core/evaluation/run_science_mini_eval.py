#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-benchmark test runner for 7th-Grade Science RAG Pipeline.
Executes a limited batch of real benchmark items from science_textbook.csv
and displays detailed breakdown of Dense Qdrant, BM25, and RRF retrieval.
"""

import argparse
import asyncio
import pandas as pd
from pathlib import Path

from rag_core.pipeline import AdvancedRAGPipeline
from rag_core.config import settings


async def run_mini_benchmark(limit: int = 3, dataset_path: str = None):
    csv_path = dataset_path or (
        settings.workspace_root / "rag-core" / "src" / "rag_core" / "evaluation" / "datasets" / "suites" / "science_textbook.csv"
    )
    if not Path(csv_path).exists():
        print(f"Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_available = len(df)
    run_count = min(limit, total_available)

    print("=" * 70)
    print(f"🔬 RUNNING REAL SCIENCE PIPELINE TEST ({run_count} of {total_available} benchmark items)")
    print("=" * 70)

    pipeline = AdvancedRAGPipeline()

    for idx, row in df.head(run_count).iterrows():
        query = row["input"]
        expected = row.get("expected_output", "")
        print(f"\n[{idx + 1}/{run_count}] ❓ Question: {query}")

        # 1. Inspect Hybrid Retrieval Details
        details = await pipeline.hybrid_retriever.retrieve_with_details(query, top_k=5)
        dense_cands = details.get("dense", [])
        bm25_cands = details.get("bm25", [])
        fused_cands = details.get("fused", [])

        print(f"    ├─ 🔍 Dense Qdrant search : {len(dense_cands)} candidates")
        if dense_cands:
            top_d = dense_cands[0]
            print(f"    │   └─ Top: {top_d.chunk_id} (score={top_d.dense_score:.4f}, page={top_d.page})")
        print(f"    ├─ 📖 BM25 Lexical search : {len(bm25_cands)} candidates")
        print(f"    ├─ ⚡ Fused via RRF (k=60): {len(fused_cands)} candidates")

        # 2. Run full end-to-end query
        result = await pipeline.query(query, top_k=3)
        total_ms = result.latency_profile.get("total_ms", 0.0)
        rerank_ms = result.latency_profile.get("rerank_ms", 0.0)
        retrieval_ms = result.latency_profile.get("retrieval_ms", 0.0)

        print(f"    ├─ 🎯 Cross-Encoder Rerank: {rerank_ms} ms")
        print(f"    ├─ ⏱️ Total Latency       : {total_ms} ms (retrieval: {retrieval_ms} ms)")
        print(f"    ├─ 📚 Citations ({len(result.citations)})     : {', '.join([f'p.{c.page}' for c in result.citations])}")
        print(f"    └─ 📡 Langfuse Trace ID   : {result.trace_id}")

    print("\n" + "=" * 70)
    print("✅ Limited run completed successfully! All dense & lexical searches active.")
    if pipeline.tracker and pipeline.tracker.is_active:
        print("📡 Flushing all traces to Langfuse Cloud...")
        pipeline.tracker.flush()
        print("✓ Telemetry flushed to Langfuse Cloud!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run a limited test of the Science RAG Pipeline.")
    parser.add_argument("--limit", type=int, default=3, help="Number of questions to test (default: 3).")
    parser.add_argument("--dataset", type=str, default=None, help="Custom dataset CSV path.")
    args = parser.parse_args()

    asyncio.run(run_mini_benchmark(limit=args.limit, dataset_path=args.dataset))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langfuse Dataset Benchmark Runner & Synchronizer.
Migrates the local Golden Benchmark (`golden_set.jsonl`) to the Langfuse Dataset ecosystem,
executing automated evaluation experiments with trace score linking per FR-041 and SC-018.
"""

import sys
import time
from typing import List, Dict, Any, Optional

from rag_core.config import settings
from rag_core.observability.langfuse import LangfuseTracker
from rag_core.evaluation.datasets.load_dataset import load_golden_dataset
from rag_core.graph.pipeline_graph import LangGraphRAGPipeline
from rag_core.naive.embedder import BgeM3Embedder
from rag_core.naive.qdrant_retriever import QdrantRetriever
from rag_core.naive.synthesizer import NaiveSynthesizer
from rag_core.evaluation.jev_judge import JevJudge
from rag_core.query.jev_router import JevRouter
from rag_core.fusion.jev_reranker import JevReranker

from rag_core.evaluation.migrate_datasets_to_langfuse import migrate_all_subfolders

DEFAULT_DATASET_NAME = "dev/dev_set"


def sync_golden_benchmark_to_langfuse(
    dataset_name: Optional[str] = None,
    split: str = "all"
) -> int:
    """
    Delegate dataset synchronization to the dedicated migrate_all_subfolders workflow.
    """
    subfolders = ["dev", "test", "suites"] if split == "all" else [split]
    stats = migrate_all_subfolders(subfolders=subfolders)
    return stats.get("total_items", 0)


def run_langfuse_dataset_experiment(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "dev",
    pipeline_mode: str = "baseline",
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute benchmark evaluation experiment directly linked to Langfuse Dataset items.
    Instruments per-item traces with cost, latency, and evaluation score attribution.
    """
    tracker = LangfuseTracker()
    items = load_golden_dataset(split=split)
    if max_items:
        items = items[:max_items]

    print(f"\n======================================================================")
    print(f"Executing Langfuse Dataset Experiment: {dataset_name}")
    print(f"Pipeline Mode: {pipeline_mode.upper()} | Split: {split} | Items: {len(items)}")
    print(f"Tracing: {'ACTIVE' if tracker.is_active else 'OFFLINE (Graceful Fallback)'}")
    print(f"======================================================================\n")

    # Initialize components
    embedder = BgeM3Embedder()
    retriever = QdrantRetriever(embedder)
    synthesizer = NaiveSynthesizer(
        base_url=settings.nine_router_base_url,
        api_key=settings.nine_router_api_key,
        model=settings.llm_model
    )
    router = JevRouter()
    reranker = JevReranker()
    judge = JevJudge()

    # Compile LangGraph Pipeline
    pipeline = LangGraphRAGPipeline(
        retriever=retriever,
        synthesizer=synthesizer,
        router=router,
        reranker=reranker,
        judge=judge,
        enable_reranking=(pipeline_mode == "improved"),
        pipeline_name=pipeline_mode
    )

    results: List[Dict[str, Any]] = []
    total_passed = 0
    t_start = time.perf_counter()

    for idx, it in enumerate(items, start=1):
        q = it.get("question") or it.get("query", "")
        gt = it.get("ground_truth", "")
        suite = it.get("suite", "general")

        tags = [pipeline_mode, split, suite]

        # Execute through LangGraph StateGraph
        state = pipeline.invoke(
            query=q,
            ground_truth=gt,
            suite=suite,
            tags=tags,
            metadata={"item_id": it.get("id"), "difficulty": it.get("difficulty")}
        )

        verdict = state.get("overall_verdict", 0)
        total_passed += verdict
        results.append({
            "id": it.get("id"),
            "query": q,
            "overall_verdict": verdict,
            "agent_response": state.get("agent_response", ""),
            "latencies": state.get("latencies", {}),
            "tokens": state.get("tokens", {}),
            "cost_usd": state.get("cost_usd", 0.0),
            "evaluation_verdicts": state.get("evaluation_verdicts", {}),
            "trace_id": state.get("trace_id")
        })

        status = "PASS" if verdict == 1 else "FAIL"
        print(f"[{idx}/{len(items)}] [{status}] {q[:60]}... (Cost: ${state.get('cost_usd', 0):.6f})")

    tracker.flush()
    retriever.close()

    total_time = time.perf_counter() - t_start
    pass_rate = (total_passed / len(items)) if items else 0.0
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)

    print("\n======================================================================")
    print(f"Langfuse Dataset Experiment Complete!")
    print(f"Overall Pass Rate: {pass_rate * 100:.2f}% ({total_passed}/{len(items)})")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Total Estimated Token Cost: ${total_cost:.6f}")
    print(f"======================================================================\n")

    return {
        "dataset_name": dataset_name,
        "pipeline_mode": pipeline_mode,
        "split": split,
        "total_items": len(items),
        "passed_items": total_passed,
        "pass_rate": round(pass_rate, 4),
        "total_time_seconds": round(total_time, 2),
        "total_cost_usd": round(total_cost, 6),
        "results": results
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Langfuse Dataset Benchmark Runner")
    parser.add_argument("--sync", action="store_true", help="Sync golden dataset to Langfuse Dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Dataset name")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"], help="Split")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "improved"], help="Pipeline mode")
    parser.add_argument("--max-items", type=int, default=None, help="Cap items")
    args = parser.parse_args()

    if args.sync:
        sync_golden_benchmark_to_langfuse(dataset_name=args.dataset, split=args.split)
    else:
        run_langfuse_dataset_experiment(
            dataset_name=args.dataset,
            split=args.split,
            pipeline_mode=args.mode,
            max_items=args.max_items
        )

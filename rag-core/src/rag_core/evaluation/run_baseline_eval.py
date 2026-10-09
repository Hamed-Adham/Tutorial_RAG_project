#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run primary benchmark evaluation on Naive RAG Baseline.
Executes against golden_set.jsonl (dev split), evaluates using 5-part rubrics,
and produces the official "Ground Zero" baseline metrics report.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag_core.config import settings
from rag_core.evaluation.datasets.load_dataset import load_golden_dataset
from rag_core.evaluation.meta_eval import (
    ConfusionMatrix,
    compute_confusion_matrix,
    ReliabilityMetrics,
    evaluate_reliability_harness
)
from rag_core.naive.pipeline import NaiveRAGPipeline
from rag_core.evaluation.judge_evaluator import LLMJudge
from rag_core.evaluation.jev_judge import JevJudge
from rag_core.evaluation.code_gates import evaluate_code_gate

def run_baseline_evaluation(
    split: str = "dev",
    max_items: Optional[int] = None,
    output_report_path: Optional[Path] = None,
    retriever_type: str = "qdrant",
    force: bool = False,
    judge_type: str = "jev"
) -> Dict[str, Any]:
    """
    Execute baseline evaluation over the specified split.
    
    Args:
        split: 'dev' or 'test' (defaults to 'dev' for iteration baseline).
        max_items: Optional cap for testing/debugging.
        output_report_path: Path to save the resulting JSON report.
        retriever_type: 'qdrant' (default) or 'numpy'.
        force: If True, bypass cached checkpoints and evaluate every item live.
        judge_type: 'jev' (default, System One) or 'llm' (generative CoT).
    """
    default_filename = "qdrant_baseline_report.json" if retriever_type == "qdrant" else "naive_baseline_report.json"
    report_path = output_report_path or (
        settings.workspace_root / "rag-core" / "src" / "rag_core" / "evaluation" / "baselines" / default_filename
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_golden_dataset(split=split)
    if max_items and max_items < len(items):
        # Balanced sampling across all 4 suites
        by_suite = {}
        for it in items:
            by_suite.setdefault(it["suite"], []).append(it)
        sampled = []
        per_suite = max(1, max_items // len(by_suite))
        for s, s_items in by_suite.items():
            sampled.extend(s_items[:per_suite])
        # If remainder, fill until max_items
        for s, s_items in by_suite.items():
            for it in s_items[per_suite:]:
                if len(sampled) < max_items:
                    sampled.append(it)
        items = sampled

    print(f"\n=======================================================")
    print(f"Starting Baseline Evaluation ({retriever_type.upper()} retriever) on '{split}' split ({len(items)} items)")
    print(f"=======================================================\n")

    pipeline = NaiveRAGPipeline(retriever_type=retriever_type)
    pipeline.build_index()

    llm_judge = LLMJudge()
    jev_judge = JevJudge(llm_judge=llm_judge) if judge_type == "jev" else None

    results = []
    suite_verdicts: Dict[str, List[int]] = {}
    latencies = []
    run_session_id = int(time.time())

    # Check for existing checkpoint to resume
    completed_cache = {}
    if not force and report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                if "detailed_results" in prev_data and prev_data.get("split") == split and prev_data.get("retriever_type") == retriever_type:
                    for prev_r in prev_data["detailed_results"]:
                        completed_cache[prev_r["id"]] = prev_r
            if completed_cache:
                print(f"Resuming evaluation: found {len(completed_cache)} previously evaluated items in {report_path.name}.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    for idx, item in enumerate(items, 1):
        q = item["question"]
        suite = item["suite"]
        gt = item.get("ground_truth", "")
        item_id = item["id"]

        if item_id in completed_cache:
            prev_item = completed_cache[item_id]
            results.append(prev_item)
            overall_verdict = prev_item["overall_verdict"]
            suite_verdicts.setdefault(suite, []).append(overall_verdict)
            latencies.append(prev_item["latencies"]["total"])
            status_tag = "PASS" if overall_verdict == 1 else "FAIL"
            print(f"[{idx}/{len(items)}] Reusing cached {item_id} ({suite}): [{status_tag}] (latency: {prev_item['latencies']['total']}s)")
            continue

        print(f"[{idx}/{len(items)}] Evaluating {item_id} ({suite}): {q[:60]}...")

        # 1. Run Pipeline with granular Langfuse trace tags and session tracking
        eval_tags = ["baseline", f"split:{split}", f"suite:{suite}", f"judge:{judge_type}"]
        pipe_output = pipeline.query(
            q,
            top_k=3,
            tags=eval_tags,
            session_id=f"eval_{split}_{run_session_id}",
            metadata={"item_id": item_id, "suite": suite, "judge_type": judge_type}
        )
        answer = pipe_output["answer"]
        contexts = pipe_output["retrieved_contexts"]
        combined_context = "\n\n".join(contexts)
        latencies.append(pipe_output["total_latency"])

        # 2. Judge Evaluation based on Suite
        item_eval = {
            "id": item_id,
            "suite": suite,
            "question": q,
            "answer": answer,
            "retrieved_sources": pipe_output["sources"],
            "latencies": {
                "retrieval": pipe_output["retrieval_latency"],
                "generation": pipe_output["generation_latency"],
                "total": pipe_output["total_latency"]
            },
            "metrics": {}
        }

        # LAYER 1: Deterministic Code Gates (< 5ms, 0 token cost)
        gate_res = evaluate_code_gate(item, pipe_output)
        item_eval["hit_at_5"] = gate_res.hit_at_k

        if gate_res.gated:
            # Resolved deterministically in Layer 1!
            overall_verdict = gate_res.passed
            item_eval["metrics"][gate_res.metric] = {
                "verdict": gate_res.passed,
                "thought": gate_res.reason,
                "layer": "layer_1_deterministic_gate",
                "latency_seconds": gate_res.latency_seconds
            }
            if suite == "domain_routing":
                retrieved_sources = [s["source_type"] for s in pipe_output["sources"]]
                actual_route = max(set(retrieved_sources), key=retrieved_sources.count) if retrieved_sources else "unknown"
                item_eval["metrics"][gate_res.metric]["actual_route"] = actual_route
                item_eval["metrics"][gate_res.metric]["expected_route"] = item.get("expected_route", "")

        else:
            # LAYER 2 / 2.5: Semantic Evaluation (Jev System One or LLM-as-a-Judge)
            if judge_type == "jev" and jev_judge:
                # 1 Single Parallel Speculative Jev Call for ALL metrics (faithfulness, 3-level relevancy, context coverage)
                jev_verdicts = jev_judge.evaluate_all_speculative(
                    query=q,
                    response=answer,
                    context=combined_context,
                    ground_truth=gt,
                    suite=suite
                )
                v_faith = jev_verdicts.get("faithfulness")
                v_rel = jev_verdicts.get("answer_relevancy")
                v_cov = jev_verdicts.get("context_coverage")

                if v_faith:
                    item_eval["metrics"]["faithfulness"] = {
                        "verdict": v_faith.verdict,
                        "score": v_faith.score,
                        "thought": (
                            v_faith.details.get("thought", f"Jev calibrated probability: {v_faith.score:.3f}")
                            if v_faith.details else f"Score: {v_faith.score:.3f}"
                        ),
                        "layer": "layer_2_llm_judge_fallback" if v_faith.fallback_used else "layer_2.5_jev_system_one",
                        "confidence": v_faith.confidence,
                        "latency_seconds": v_faith.latency_seconds
                    }
                if v_rel:
                    item_eval["metrics"]["answer_relevancy"] = {
                        "verdict": v_rel.verdict,
                        "score": v_rel.score,
                        "thought": (
                            v_rel.details.get("thought", f"Jev 3-level score: {v_rel.score:.2f}/2.0")
                            if v_rel.details else f"Score: {v_rel.score:.2f}"
                        ),
                        "layer": "layer_2_llm_judge_fallback" if v_rel.fallback_used else "layer_2.5_jev_system_one",
                        "confidence": v_rel.confidence,
                        "latency_seconds": v_rel.latency_seconds
                    }
                if v_cov:
                    item_eval["metrics"]["context_coverage"] = {
                        "verdict": v_cov.verdict,
                        "score": v_cov.score,
                        "thought": f"Jev coverage probability: {v_cov.score:.3f}",
                        "layer": "layer_2_llm_judge_fallback" if v_cov.fallback_used else "layer_2.5_jev_system_one",
                        "confidence": v_cov.confidence,
                        "latency_seconds": v_cov.latency_seconds
                    }

                pass_faith = v_faith.verdict == 1 if v_faith else True
                pass_rel = v_rel.verdict == 1 if v_rel else True
                overall_verdict = 1 if (pass_faith and pass_rel) else 0
            else:
                v_faith = llm_judge.evaluate_metric(
                    rubric_name="faithfulness",
                    query=q,
                    response=answer,
                    context=combined_context
                )
                v_rel = llm_judge.evaluate_metric(
                    rubric_name="answer_relevancy",
                    query=q,
                    response=answer,
                    ground_truth=gt
                )
                item_eval["metrics"]["faithfulness"] = {
                    "verdict": v_faith.verdict,
                    "thought": v_faith.thought,
                    "layer": "layer_2_llm_judge",
                    "latency_seconds": v_faith.latency_seconds
                }
                item_eval["metrics"]["answer_relevancy"] = {
                    "verdict": v_rel.verdict,
                    "thought": v_rel.thought,
                    "layer": "layer_2_llm_judge",
                    "latency_seconds": v_rel.latency_seconds
                }
                overall_verdict = 1 if (v_faith.verdict == 1 and v_rel.verdict == 1) else 0

        item_eval["overall_verdict"] = overall_verdict
        results.append(item_eval)

        # Ingest evaluation verdicts into Langfuse trace if active
        trace_id = pipe_output.get("trace_id")
        if trace_id and pipeline.tracker.is_active:
            pipeline.tracker.add_trace_score(trace_id, "overall_verdict", float(overall_verdict))
            if gate_res.gated:
                pipeline.tracker.add_trace_score(trace_id, "gate_pass", float(gate_res.passed), comment=gate_res.reason)
            else:
                f_score = float(v_faith.verdict) if v_faith else 1.0
                r_score = float(v_rel.verdict) if v_rel else 1.0
                f_comment = v_faith.details.get("thought", "") if (v_faith and getattr(v_faith, "details", None)) else getattr(v_faith, "thought", "")
                r_comment = v_rel.details.get("thought", "") if (v_rel and getattr(v_rel, "details", None)) else getattr(v_rel, "thought", "")
                f_lat = getattr(v_faith, "latency_seconds", 0.0) if v_faith else 0.0
                r_lat = getattr(v_rel, "latency_seconds", 0.0) if v_rel else 0.0
                pipeline.tracker.add_trace_score(trace_id, "faithfulness", f_score, comment=str(f_comment)[:200])
                pipeline.tracker.add_trace_score(trace_id, "answer_relevancy", r_score, comment=str(r_comment)[:200])
                pipeline.tracker.add_trace_score(trace_id, "judge_latency", float(f_lat + r_lat))


        if suite not in suite_verdicts:
            suite_verdicts[suite] = []
        suite_verdicts[suite].append(overall_verdict)


        status_tag = "PASS" if overall_verdict == 1 else "FAIL"
        print(f"       -> Result: [{status_tag}] (latency: {pipe_output['total_latency']}s)")

        # Incremental checkpoint save every 5 items
        if idx % 5 == 0:
            interim_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": f"in_progress ({idx}/{len(items)})",
                "pipeline_name": f"Naive RAG Baseline ({retriever_type.upper()} Storage)",
                "retriever_type": retriever_type,
                "split": split,
                "total_items_processed": len(results),
                "interim_passed": sum(r["overall_verdict"] for r in results),
                "detailed_results": results
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(interim_report, f, ensure_ascii=False, indent=2)
            if hasattr(pipeline, "tracker") and pipeline.tracker.is_active:
                pipeline.tracker.flush()

    # Aggregate Statistics
    total_evaluated = len(results)
    total_passed = sum(r["overall_verdict"] for r in results)
    overall_pass_rate = (total_passed / total_evaluated) if total_evaluated > 0 else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    suite_summary = {}
    for s, verdicts in suite_verdicts.items():
        s_pass = sum(verdicts)
        s_total = len(verdicts)
        suite_summary[s] = {
            "passed": s_pass,
            "total": s_total,
            "pass_rate": round(s_pass / s_total, 4) if s_total > 0 else 0.0
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_name": f"Naive RAG Baseline ({retriever_type.upper()} Storage)",
        "retriever_type": retriever_type,
        "split": split,
        "total_items": total_evaluated,
        "overall_passed": total_passed,
        "overall_pass_rate": round(overall_pass_rate, 4),
        "mean_latency_seconds": round(avg_latency, 3),
        "suite_breakdown": suite_summary,
        "detailed_results": results
    }

    pipeline.close()

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=======================================================")
    print(f"Baseline Evaluation Complete! Saved report to: {report_path}")
    print(f"Overall Pass Rate: {overall_pass_rate * 100:.2f}% ({total_passed}/{total_evaluated})")
    print(f"Mean Latency: {avg_latency:.2f}s per query")
    print("Suite Breakdown:")
    for s, info in suite_summary.items():
        print(f"  - {s}: {info['pass_rate'] * 100:.2f}% ({info['passed']}/{info['total']})")
    print("=======================================================\n")

    if hasattr(pipeline, "tracker") and pipeline.tracker.is_active:
        pipeline.tracker.flush()

    return report


def run_gates_only_evaluation(
    split: str = "all",
    suite_filter: str = "all"
) -> bool:
    """
    Execute Tier 2: Layer 1 Deterministic Code Gates across benchmark items.
    Runs in < 2 seconds with zero LLM API tokens.
    Verifies Refusal Catch Rate >= 90% (SC-012) and Domain Routing Accuracy >= 95% (SC-011).
    Returns True if all constitutional thresholds are met, False otherwise.
    """
    print("\n======================================================================")
    print("Executing Tier 2: Layer 1 Deterministic Dataset Gates")
    print("Scope: Golden Benchmark (0 LLM Tokens, < 5ms per item)")
    print("======================================================================\n")

    # Load items across dev and test if split is "all"
    if split == "all":
        dev_items = load_golden_dataset(split="dev")
        test_items = load_golden_dataset(split="test")
        items = dev_items + test_items
    else:
        items = load_golden_dataset(split=split)

    if suite_filter != "all":
        items = [it for it in items if it.get("suite") == suite_filter]

    suite_results: Dict[str, Dict[str, int]] = {}
    latencies: List[float] = []
    total_passed = 0

    for it in items:
        suite = it.get("suite", "unknown")
        res = evaluate_code_gate(it, pipeline_output=None)
        latencies.append(res.latency_seconds)

        suite_results.setdefault(suite, {"passed": 0, "total": 0})
        suite_results[suite]["total"] += 1
        if res.passed:
            suite_results[suite]["passed"] += 1
            total_passed += 1

    total_items = len(items)
    overall_pass_rate = (total_passed / total_items) if total_items > 0 else 0.0
    mean_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0

    print(f"Total Benchmark Items Evaluated: {total_items}")
    print(f"Total Gates Passed: {total_passed}/{total_items} ({overall_pass_rate * 100:.1f}%)")
    print(f"Mean Execution Latency: {mean_latency_ms:.3f} ms per gate")
    print(f"Token Consumption: 0 tokens ($0.00)\n")
    print("Suite-by-Suite Breakdown:")

    all_passed = True
    for s, data in suite_results.items():
        s_pass = data["passed"]
        s_tot = data["total"]
        rate = (s_pass / s_tot) if s_tot > 0 else 0.0

        target_str = ""
        gate_status = "PASS"
        if s == "adversarial_refusal":
            target_str = " (Constitutional Target: >= 90.0%)"
            if rate < 0.90:
                gate_status = "FAIL"
                all_passed = False
        elif s == "domain_routing":
            target_str = " (Constitutional Target: >= 95.0%)"
            if rate < 0.95:
                gate_status = "FAIL"
                all_passed = False
        else:
            target_str = " (Content Schema & Ground Truth)"
            if rate < 1.0:
                gate_status = "FAIL"
                all_passed = False

        status_symbol = "✓" if gate_status == "PASS" else "✗"
        print(f"  {status_symbol} {s:<22}: {s_pass}/{s_tot} ({rate * 100:.1f}%){target_str} -> [{gate_status}]")

    print("\n======================================================================")
    if all_passed:
        print("✓ All Layer 1 Deterministic Code Gates Passed Constitutional Standards!")
    else:
        print("✗ One or more deterministic gates failed constitutional thresholds!")
    print("======================================================================\n")

    return all_passed


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run Baseline Evaluation")
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"], help="Dataset split to evaluate")
    parser.add_argument("--suite", default="all", choices=["all", "science_textbook", "administrative_faq", "domain_routing", "adversarial_refusal"], help="Filter by suite")
    parser.add_argument("--max-items", type=int, default=None, help="Cap number of items to evaluate")
    parser.add_argument("--retriever", default="qdrant", choices=["qdrant", "numpy"], help="Retriever backend")
    parser.add_argument("--gates-only", action="store_true", help="Run Tier 2 Layer 1 deterministic code gates only (0 tokens, < 2s)")
    parser.add_argument("--force", action="store_true", help="Force fresh live evaluation without reusing cached results")
    parser.add_argument("--judge", default="jev", choices=["jev", "llm"], help="Evaluation judge engine ('jev' default, 'llm' for generative CoT)")
    args = parser.parse_args()

    if args.gates_only:
        success = run_gates_only_evaluation(split=args.split, suite_filter=args.suite)
        sys.exit(0 if success else 1)
    else:
        eval_split = "dev" if args.split == "all" else args.split
        run_baseline_evaluation(
            split=eval_split,
            max_items=args.max_items,
            retriever_type=args.retriever,
            force=args.force,
            judge_type=args.judge
        )




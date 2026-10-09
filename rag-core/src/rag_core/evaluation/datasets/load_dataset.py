#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset loader and benchmark validator for Unified RAG Golden Evaluation Dataset.
Provides typed access, suite filtering, and integrity validation.
"""

import os
import json
from typing import List, Dict, Any, Optional

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")

def load_golden_dataset(
    suite: Optional[str] = None,
    split: Optional[str] = None,
    dataset_path: str = DATASET_PATH
) -> List[Dict[str, Any]]:
    """
    Load golden dataset items from JSONL.
    
    Args:
        suite: Optional suite filter ('science_textbook', 'administrative_faq', 
               'domain_routing', 'adversarial_refusal').
        split: Optional split filter ('dev' or 'test').
        dataset_path: Path to golden_set.jsonl file.
        
    Returns:
        List of benchmark test item dictionaries.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")

    items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if suite and item.get("suite") != suite:
                continue
            if split and item.get("split") != split:
                continue
            items.append(item)

    return items

def validate_dataset_integrity(dataset_path: str = DATASET_PATH) -> Dict[str, Any]:
    """
    Validate dataset completeness and report statistics.
    """
    items = load_golden_dataset(dataset_path=dataset_path)
    required_keys = {"id", "suite", "split", "category", "query_type", "question", "ground_truth", "expected_behavior"}
    
    stats = {
        "total_items": len(items),
        "suites": {},
        "splits": {},
        "query_types": {},
        "valid": True,
        "errors": []
    }

    for i, item in enumerate(items, 1):
        missing = required_keys - set(item.keys())
        if missing:
            stats["valid"] = False
            stats["errors"].append(f"Item #{i} (ID: {item.get('id', 'unknown')}) missing keys: {missing}")

        s = item.get("suite", "unknown")
        stats["suites"][s] = stats["suites"].get(s, 0) + 1

        sp = item.get("split", "unknown")
        stats["splits"][sp] = stats["splits"].get(sp, 0) + 1

        qt = item.get("query_type", "unknown")
        stats["query_types"][qt] = stats["query_types"].get(qt, 0) + 1

    return stats

CSV_FIELDS = ["input", "expected_output", "metadata"]
LANGFUSE_CSV_FIELDS = ["input", "expected_output", "metadata"]

def format_csv_row(item: Dict[str, Any]) -> Dict[str, str]:
    """Format a dataset item dictionary into standard 3-column CSV format with standardized metadata."""
    suite = item.get("suite", "")
    domain = suite if suite in ("science_textbook", "administrative_faq") else (
        "science_textbook" if "SCI" in str(item.get("id", ""))
        else ("administrative_faq" if "FAQ" in str(item.get("id", "")) else "out_of_scope")
    )
    meta = {
        "id": item.get("id"),
        "domain": domain,
        "query_type": item.get("query_type"),
        "ground_truth_chunks": item.get("ground_truth_chunks", []),
        "expected_behavior": item.get("expected_behavior", "answer"),
        "expected_route": item.get("expected_route", suite),
        "category": item.get("category", ""),
        "split": item.get("split", "dev")
    }
    return {
        "input": item.get("question", ""),
        "expected_output": item.get("ground_truth", ""),
        "metadata": json.dumps(meta, ensure_ascii=False),
    }

def format_langfuse_row(item: Dict[str, Any]) -> Dict[str, str]:
    """Format a dataset item dictionary into Langfuse 3-column format (input, expected_output, metadata)."""
    return format_csv_row(item)

def _write_csv(filepath: str, items: List[Dict[str, Any]]) -> None:
    """Helper to write benchmark items into a UTF-8 with BOM CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(format_csv_row(item))

def _write_langfuse_csv(filepath: str, items: List[Dict[str, Any]]) -> None:
    """Helper to write benchmark items into Langfuse auto-mapped 3-column CSV."""
    _write_csv(filepath, items)

def export_to_csv(
    dataset_path: str = DATASET_PATH,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Export golden_set.jsonl into suite-specific, split-specific, and full CSV files.
    
    Structure:
      - output_dir/
          - full_set.csv (all 100 benchmark items)
          - golden_set.csv (alias for full_set)
          - dev_set.csv (all 75 dev items)
          - test_set.csv (all 25 test items)
          - suites/
              - <suite_name>.csv (full per suite, e.g. science_textbook.csv)
          - dev/
              - dev_set.csv (full dev set)
              - <suite_name>_dev.csv (e.g. science_textbook_dev.csv)
          - test/
              - test_set.csv (full test set)
              - <suite_name>_test.csv (e.g. science_textbook_test.csv)
      
    Returns:
      Dictionary mapping export key to output filepath.
    """
    import csv
    if output_dir is None:
        output_dir = os.path.dirname(dataset_path)

    items = load_golden_dataset(dataset_path=dataset_path)
    exported = {}

    # Define directories
    suites_dir = os.path.join(output_dir, "suites")
    dev_dir = os.path.join(output_dir, "dev")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(suites_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Clean up legacy dev/test splits in suites/ and legacy root suite_*.csv files
    for fname in os.listdir(suites_dir):
        if fname.endswith("_dev.csv") or fname.endswith("_test.csv"):
            try:
                os.remove(os.path.join(suites_dir, fname))
            except OSError:
                pass

    for fname in os.listdir(output_dir):
        if fname.startswith("suite_") and fname.endswith(".csv"):
            try:
                os.remove(os.path.join(output_dir, fname))
            except OSError:
                pass

    # 1. Full datasets at root
    full_path = os.path.join(output_dir, "full_set.csv")
    _write_csv(full_path, items)
    exported["full_set"] = full_path

    golden_path = os.path.join(output_dir, "golden_set.csv")
    _write_csv(golden_path, items)
    exported["golden_set"] = golden_path

    # Langfuse-specific auto-mapped exports (input, expected_output, metadata)
    lf_dir = os.path.join(output_dir, "langfuse")
    lf_suites_dir = os.path.join(lf_dir, "suites")
    lf_dev_dir = os.path.join(lf_dir, "dev")
    lf_test_dir = os.path.join(lf_dir, "test")
    for d in [lf_dir, lf_suites_dir, lf_dev_dir, lf_test_dir]:
        os.makedirs(d, exist_ok=True)

    lf_full_path = os.path.join(lf_dir, "full_set.csv")
    _write_langfuse_csv(lf_full_path, items)
    exported["langfuse/full_set"] = lf_full_path

    # 2. Splits at root & in respective folders
    dev_items = [it for it in items if it.get("split") == "dev"]
    test_items = [it for it in items if it.get("split") == "test"]

    root_dev_path = os.path.join(output_dir, "dev_set.csv")
    _write_csv(root_dev_path, dev_items)
    exported["dev_set"] = root_dev_path

    folder_dev_path = os.path.join(dev_dir, "dev_set.csv")
    _write_csv(folder_dev_path, dev_items)
    exported["dev/dev_set"] = folder_dev_path

    lf_dev_path = os.path.join(lf_dev_dir, "dev_set.csv")
    _write_langfuse_csv(lf_dev_path, dev_items)
    exported["langfuse/dev/dev_set"] = lf_dev_path

    root_test_path = os.path.join(output_dir, "test_set.csv")
    _write_csv(root_test_path, test_items)
    exported["test_set"] = root_test_path

    folder_test_path = os.path.join(test_dir, "test_set.csv")
    _write_csv(folder_test_path, test_items)
    exported["test/test_set"] = folder_test_path

    lf_test_path = os.path.join(lf_test_dir, "test_set.csv")
    _write_langfuse_csv(lf_test_path, test_items)
    exported["langfuse/test/test_set"] = lf_test_path

    # 3. Per-suite breakdown
    suites = sorted({it.get("suite") for it in items if it.get("suite")})

    for suite_name in suites:
        suite_items = [it for it in items if it.get("suite") == suite_name]
        suite_dev = [it for it in suite_items if it.get("split") == "dev"]
        suite_test = [it for it in suite_items if it.get("split") == "test"]

        # 3a. Full suite only in suites/
        s_full = os.path.join(suites_dir, f"{suite_name}.csv")
        _write_csv(s_full, suite_items)
        exported[f"suites/{suite_name}"] = s_full

        lf_s_full = os.path.join(lf_suites_dir, f"{suite_name}.csv")
        _write_langfuse_csv(lf_s_full, suite_items)
        exported[f"langfuse/suites/{suite_name}"] = lf_s_full

        # 3b. Dev suite in dev/
        s_dev = os.path.join(dev_dir, f"{suite_name}_dev.csv")
        _write_csv(s_dev, suite_dev)
        exported[f"dev/{suite_name}_dev"] = s_dev

        lf_s_dev = os.path.join(lf_dev_dir, f"{suite_name}_dev.csv")
        _write_langfuse_csv(lf_s_dev, suite_dev)
        exported[f"langfuse/dev/{suite_name}_dev"] = lf_s_dev

        # 3c. Test suite in test/
        s_test = os.path.join(test_dir, f"{suite_name}_test.csv")
        _write_csv(s_test, suite_test)
        exported[f"test/{suite_name}_test"] = s_test

        lf_s_test = os.path.join(lf_test_dir, f"{suite_name}_test.csv")
        _write_langfuse_csv(lf_s_test, suite_test)
        exported[f"langfuse/test/{suite_name}_test"] = lf_s_test

    return exported

if __name__ == "__main__":
    import sys
    import csv

    # Check CLI options
    should_export = "--export-csv" in sys.argv or "--export" in sys.argv or "-e" in sys.argv

    if should_export:
        print("Exporting golden_set.jsonl to CSV formats...")
        exported_files = export_to_csv()
        for key, path in exported_files.items():
            print(f"  [Exported] {key} -> {path}")
        print()

    results = validate_dataset_integrity()
    print("=== Golden Dataset Integrity Report ===")
    print(f"Total Test Items: {results['total_items']}")
    print(f"Status: {'VALID' if results['valid'] else 'INVALID'}")
    print("\nSplit Distribution:")
    for sp, count in results["splits"].items():
        print(f"  - {sp}: {count} ({count/results['total_items']*100:.1f}%)")
    print("\nSuite Distribution:")
    for s, count in results["suites"].items():
        print(f"  - {s}: {count} ({count/results['total_items']*100:.1f}%)")
    print("\nQuery Type Distribution:")
    for qt, count in results["query_types"].items():
        print(f"  - {qt}: {count}")
    if results["errors"]:
        print(f"\nErrors found ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  - {err}")

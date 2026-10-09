#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract test for Master Golden Dataset Synchronization and Split Purity (US5).
Validates that golden_set.jsonl has >= 400 items, conforms to BenchmarkItemModel,
and maintains strict 75/25 split purity with zero overlap.
"""

import json
import unittest
from pathlib import Path
from rag_core.evaluation.benchmark_schema import BenchmarkItemModel


class TestGoldenSetSync(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = Path("rag-core/src/rag_core/evaluation/datasets")
        self.golden_jsonl = self.dataset_dir / "golden_set.jsonl"
        self.golden_csv = self.dataset_dir / "golden_set.csv"

    def test_golden_jsonl_existence_and_scale(self):
        if not self.golden_jsonl.exists():
            self.skipTest(f"{self.golden_jsonl} does not exist yet (run build_standardized_benchmarks.py first)")

        with open(self.golden_jsonl, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        self.assertGreaterEqual(len(lines), 400, f"Expected at least 400 items, found {len(lines)}")

    def test_golden_jsonl_schema_validation(self):
        if not self.golden_jsonl.exists():
            self.skipTest(f"{self.golden_jsonl} does not exist yet")

        with open(self.golden_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                data = json.loads(line)
                # Map old question/ground_truth format to input/expected_output if needed
                inp = data.get("input") or data.get("question", "")
                exp = data.get("expected_output") or data.get("ground_truth", "")
                meta = data.get("metadata", {})
                if isinstance(meta, str):
                    meta = json.loads(meta)

                # Ensure valid BenchmarkItemModel
                item = BenchmarkItemModel(
                    id=data.get("id", f"item-{idx}"),
                    input=inp,
                    expected_output=exp,
                    metadata=meta
                )
                self.assertIsNotNone(item)

    def test_suite_coverage_quotas(self):
        if not self.golden_jsonl.exists():
            self.skipTest(f"{self.golden_jsonl} does not exist yet")

        suite_counts = {}
        with open(self.golden_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                meta = data.get("metadata", {})
                if isinstance(meta, str):
                    meta = json.loads(meta)
                s = meta.get("suite") or data.get("suite", "unknown")
                suite_counts[s] = suite_counts.get(s, 0) + 1

        self.assertGreaterEqual(suite_counts.get("administrative_faq", 0), 200)
        self.assertGreaterEqual(suite_counts.get("science_textbook", 0), 180)
        self.assertGreaterEqual(suite_counts.get("domain_routing", 0), 20)
        self.assertGreaterEqual(suite_counts.get("adversarial_refusal", 0), 20)

    def test_split_purity_and_zero_leakage(self):
        dev_csv = self.dataset_dir / "dev" / "dev_set.csv"
        test_csv = self.dataset_dir / "test" / "test_set.csv"

        if not dev_csv.exists() or not test_csv.exists():
            self.skipTest("Dev/test CSVs do not exist yet")

        import csv
        def get_inputs(path: Path):
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                return set(r["input"].strip() for r in reader)

        dev_inputs = get_inputs(dev_csv)
        test_inputs = get_inputs(test_csv)

        # Zero overlap
        overlap = dev_inputs.intersection(test_inputs)
        self.assertEqual(len(overlap), 0, f"Found {len(overlap)} overlapping queries between dev and test!")

        total = len(dev_inputs) + len(test_inputs)
        dev_ratio = len(dev_inputs) / total
        self.assertAlmostEqual(dev_ratio, 0.75, delta=0.05)


if __name__ == "__main__":
    unittest.main()

# Quickstart & Validation Guide: Benchmark Dataset Scaling

This guide provides end-to-end validation commands and procedures to verify the 400+ item benchmark dataset, decoupled retrieval evaluation, and Langfuse synchronization.

---

## 1. Prerequisites

1. Ensure the Python environment is activated:
   ```bash
   source .venv/bin/activate
   ```
2. Ensure the Nine Router gateway is running at `http://localhost:20128/v1` with `gemini-cli` available.
3. Verify that the 15-chapter parsed textbook markdown files exist in `data/vlm_parsed/stitched/`.

---

## 2. Generate Expanded Benchmark Dataset

Run the automated benchmark generator to synthesize the FAQ and Science suites:

```bash
# 1. Run the standardized batch generation pipeline
python rag-core/src/rag_core/evaluation/build_standardized_benchmarks.py --scale-all

# 2. Verify dataset generation counts and schema validity
python -c "
import json
from pathlib import Path

p = Path('rag-core/src/rag_core/evaluation/datasets/golden_set.jsonl')
items = [json.loads(line) for line in p.read_text().strip().split('\n')]
print(f'Total Items: {len(items)}')
assert len(items) >= 400, f'Expected >= 400 items, got {len(items)}'

suites = {}
for it in items:
    s = it['metadata']['suite']
    suites[s] = suites.get(s, 0) + 1
print('Suite Counts:', suites)
assert suites.get('administrative_faq', 0) >= 200
assert suites.get('science_textbook', 0) >= 180
print('Benchmark Scale Verification: PASSED')
"
```

---

## 3. Validate Contract & Schema Integrity

Validate that all CSV and JSONL files conform to the contract defined in [benchmark_item_schema.json](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/002-benchmark-dataset-scaling/contracts/benchmark_item_schema.json):

```bash
# Verify schema against jsonschema validator
python -m pytest rag-core/tests/test_benchmark_schema.py -v
```

---

## 4. Run Decoupled Retrieval-Only Evaluation (Zero Token Cost)

Verify that the evaluation harness runs retrieval metrics independently in sub-second time without invoking the LLM judge:

```bash
# Run decoupled retrieval evaluation on the dev set
python rag-core/src/rag_core/evaluation/run_eval.py --retrieval-only --split dev
```

**Expected Output**:
```text
================ DECOUPLED RETRIEVAL BENCHMARK ================
Total Items Evaluated: 320
Hit@1:  78.4%
Hit@3:  89.1%
Hit@5:  94.2%
MRR:    0.841
Context Recall: 0.912
Execution Duration: 2.1s (0 LLM Judge Tokens consumed)
===============================================================
```

---

## 5. Synchronize to Langfuse Cloud

Push the newly generated datasets to Langfuse Cloud under virtual namespaces:

```bash
python rag-core/src/rag_core/evaluation/migrate_datasets_to_langfuse.py
```

**Verification**:
* Check Langfuse Cloud UI -> Datasets.
* Confirm `dev/*`, `test/*`, and `suites/*` namespaces contain the 400+ items with intact metadata attributes.

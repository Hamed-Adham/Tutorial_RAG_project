# Langfuse Datasets Specification & Implementation Standards

Comprehensive reference and technical standards for designing, creating, structuring, migrating, and experimenting with Langfuse Datasets per [Langfuse Official Documentation](https://langfuse.com/docs/evaluation/experiments/datasets).

---

## 1. Core Concepts

A Langfuse Dataset is a versioned collection of inputs, expected outputs, and metadata used to test and benchmark LLM applications across UI-based, SDK-based, and OpenTelemetry experiments.

### Key Purposes
- **Benchmark Evaluation**: Golden evaluation sets (dev, test, splits) for regression detection.
- **Production Curations**: Selecting production traces/observations and converting them into test cases with expert-annotated ground truths.
- **Single Source of Truth**: Collaborative dataset curation with full auditability and version timestamps.

---

## 2. Dataset Virtual Folders & Hierarchy

Langfuse organizes datasets into virtual folders using forward slashes (`/`) in the dataset name.
The Langfuse UI automatically renders every prefix before a `/` as an expandable folder.

### Naming Conventions
- Subfolder datasets: `<folder>/<dataset_name>` (e.g., `dev/dev_set`, `test/test_set`, `suites/science_textbook`).
- Avoid nesting under wrapper folders like `langfuse/` unless explicitly required; organize directly by domain, split, or suite.
- When querying datasets via API or TypeScript SDK with path parameters, URL-encode the name: `encodeURIComponent("dev/dev_set")` -> `dev%2Fdev_set`. In Python SDK, slashes can be passed directly as string names.

---

## 3. Dataset Creation & Schema Enforcement

Datasets are unique by name within a Langfuse project.

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Standard dataset creation
langfuse.create_dataset(
    name="dev/science_textbook_dev",
    description="7th Grade Science Textbook Dev Split Benchmark",
    metadata={
        "split": "dev",
        "suite": "science_textbook",
        "format_version": "v1.4.0",
        "author": "Unified RAG Team"
    }
)
```

### Schema Enforcement (Optional JSON Schema)
Datasets can enforce JSON schemas on `input` and `expected_output`:

```python
langfuse.create_dataset(
    name="suites/domain_routing",
    description="Domain routing classification test suite",
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "suite": {"type": "string"}
        },
        "required": ["question"]
    },
    expected_output_schema={
        "type": "object",
        "properties": {
            "expected_route": {"type": "string"}
        },
        "required": ["expected_route"]
    }
)
```

---

## 4. Dataset Item Creation & Idempotent Upserts

Dataset items represent individual test cases.

```python
langfuse.create_dataset_item(
    dataset_name="dev/dev_set",
    id="GT-SCI-001",  # Explicit ID ensures idempotent upserts (prevents duplicates)
    input="مراحل حل مسئله به روش علمی در علوم تجربی چیست؟",
    expected_output="مشاهده، طرح پرسش، فرضیه‌سازی، طراحی و انجام آزمایش، ثبت داده‌ها و نتیجه‌گیری نهایی.",
    metadata={
        "id": "GT-SCI-001",
        "suite": "science_textbook",
        "split": "dev",
        "category": "فصل 1: تفکر و تجربه",
        "query_type": "factoid",
        "reference_contexts": ["یادگیری در علوم با مشاهده، طرح پرسش..."],
        "source_document": "C706_olom_haftom-7-144-2-5.pdf",
        "expected_behavior": "answer"
    }
)
```

### Critical Field Standards:
- `id`: Always supply a deterministic persistent ID whenever available. **Important**: In Langfuse, item `id` is globally unique across the **entire project**, not just per dataset. When the same benchmark questions appear across multiple datasets (e.g. `dev/dev_set`, `suites/science_textbook`, `test/test_set`), scope the item ID with the dataset name (e.g. `f"{dataset_name}:{benchmark_id}"`). This guarantees idempotent upserts without triggering `LangfuseConflictError (409)`.
- `input`: Plain string query or structured JSON dictionary containing the user prompt/task.
- `expected_output`: Plain string or structured ground-truth response.
- `metadata`: Fully parsed JSON dictionary containing categories, reference contexts, difficulty, or source documents. Do NOT store stringified JSON as a raw string when uploading programmatically; parse it into native dicts so Langfuse UI displays structured key-value attributes.
- `status`: Status can be `"ACTIVE"` or `"ARCHIVED"`. Archived items are preserved for history but skipped during evaluation runs.

---

## 5. Converting Production Data to Dataset Items

To capture failure modes or high-value edge cases from live production traces:

```python
langfuse.create_dataset_item(
    dataset_name="production-evaluations/edge-cases",
    input={"question": "چرا آب در ارتفاعات در دمای کمتر می‌جوشد؟"},
    expected_output={"ground_truth": "به دلیل کاهش فشار هوای محیط در ارتفاعات بالاتر."},
    source_trace_id="trace_abc123",              # Links to live trace
    source_observation_id="observation_xyz789",  # Optional: Links to specific span/generation
    metadata={"reviewed_by": "domain_expert", "issue_type": "hallucination"}
)
```

In the UI:
1. Navigate to any trace or observation in the **Observations** table.
2. Select **Actions** -> **Add to dataset**.
3. Choose or create the dataset, configure field mappings, and save.

---

## 6. Migration Standards for Local Datasets (CSV / JSONL)

When migrating local evaluation datasets (such as CSV files or JSON Lines files) into Langfuse Cloud:

1. **Hierarchy Preservation**:
   - Inspect subfolders (e.g. `dev/`, `test/`, `suites/`).
   - Retain subfolder organization via slash naming: `<subfolder>/<filename_without_ext>`.
   - Strip redundant wrapper directories (e.g., migrate `langfuse/dev/dev_set.csv` as `dev/dev_set`).

2. **Idempotency**:
   - Use the item's benchmark ID as the item `id` so re-runs never duplicate test items.

3. **Metadata Hygiene**:
   - Automatically unpack JSON-encoded string fields in CSV rows into true Python dictionaries before sending to the Langfuse API.

4. **Batching & Rate Limiting**:
   - Add minimal sleep or concurrency controls if uploading thousands of items.
   - Always call `langfuse.flush()` at completion of each dataset to ensure all queued telemetry and API calls finish before script exit.

---

## 7. Running Evaluation Experiments on Versioned Datasets

Once datasets are migrated to Langfuse Cloud, run experiments linked directly to them:

```python
from langfuse import Langfuse

langfuse = Langfuse()
dataset = langfuse.get_dataset("dev/dev_set")

def evaluate_rag_pipeline(*, item, **kwargs):
    query = item.input if isinstance(item.input, str) else item.input.get("question")
    # Execute RAG pipeline
    response = rag_pipeline.run(query)
    return response

# Execute and register experiment in Langfuse UI
dataset.run_experiment(
    name="Baseline RAG Evaluation",
    description="Evaluation of baseline pipeline without reranking",
    task=evaluate_rag_pipeline
)
```

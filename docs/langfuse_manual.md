# Langfuse Observability & Benchmark Manual

A comprehensive operational guide and cheat sheet for tracing, latency profiling, cost attribution, and dataset experimentation across Unified RAG pipelines using Langfuse.

---

## 1. Quick Setup & Environment Credentials

Langfuse is enabled **by default** across all workflow executions and evaluation runs. Ensure the following environment variables are defined in `.env`:

```bash
# Langfuse Cloud or Self-Hosted Instance
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com        # Or https://us.cloud.langfuse.com / self-hosted
LANGFUSE_PUBLIC_KEY=pk-lf-...                   # Found in Langfuse Project Settings -> API Keys
LANGFUSE_SECRET_KEY=sk-lf-...                   # Found in Langfuse Project Settings -> API Keys
```

> **Note**: If Langfuse credentials are not provided or network access is restricted, the codebase automatically degrades to zero-overhead local logging without crashing or throwing exceptions.

---

## 2. Core Concepts: Latency, Cost & Dynamic Tags

Every query flowing through the LangGraph StateGraph records detailed telemetry:

### A. Granular Latency Profiling
Each pipeline stage records its execution duration in `state["latencies"]`:
- `guard_and_route`: TypeSafe Jev routing + prompt-injection check (< 300ms).
- `retrieval`: Dense vector search in Qdrant (< 30ms).
- `reranking`: Jev candidate chunk scoring and filtering (< 350ms).
- `synthesis`: Local LLM answer generation.
- `judge`: Single-call parallel evaluation verdict.

### B. Token & Cost Attribution
The synthesis node tracks token consumption and computes estimated cost in USD:
- Prompt Tokens (context + question)
- Completion Tokens (generated answer)
- Estimated Cost (e.g. $0.15/1M prompt tokens, $0.60/1M completion tokens)

### C. Structured Tagging Taxonomy
Every trace is automatically tagged to enable slicing in the Langfuse dashboard:

| Tag | Purpose | Example Query |
| :--- | :--- | :--- |
| `baseline` | Standard Naive RAG execution (BGE-M3 + Qdrant) | `make eval-baseline` |
| `improved` | Enhanced pipeline with reranking or query transforms | `make eval-improved` |
| `jev-reranker` | Traces that executed Jev candidate chunk reranking | Stage 6 Reranking |
| `dev` / `test` | Benchmark dataset split | Split filtering |
| `science` / `faq` / `refusal` | Knowledge domain or evaluation suite | Category analytics |

---

## 3. LangGraph Native Integration

Unified RAG uses LangGraph `StateGraph` as the execution standard. Native Langfuse tracing is connected via `CallbackHandler`:

```python
from rag_core.graph.pipeline_graph import LangGraphRAGPipeline
from rag_core.naive.embedder import BgeM3Embedder
from rag_core.naive.qdrant_retriever import QdrantRetriever
from rag_core.naive.synthesizer import NaiveSynthesizer

# 1. Initialize Pipeline
retriever = QdrantRetriever(BgeM3Embedder())
synthesizer = NaiveSynthesizer()
pipeline = LangGraphRAGPipeline(
    retriever=retriever,
    synthesizer=synthesizer,
    enable_reranking=True,        # Set True for improved pipeline mode
    pipeline_name="improved"
)

# 2. Execute Query with Automatic Langfuse Telemetry
result = pipeline.invoke(
    query="میتوکندری چیست؟",
    tags=["manual-query", "demo"]
)

print("Answer:", result["agent_response"])
print("Latencies:", result["latencies"])
print("Estimated Cost: $", result.get("cost_usd", 0.0))
print("Langfuse Trace ID:", result["trace_id"])
```

---

## 4. Migrating Benchmarks to Langfuse Datasets

Rather than evaluating solely against local files, all benchmark suites and splits are synchronized directly into Langfuse Cloud Datasets organized cleanly into virtual folders (`dev/`, `test/`, and `suites/`).

### Step 1: Synchronize Local Evaluation Datasets to Langfuse Cloud
Run this command to migrate all 14 datasets across `dev`, `test`, and `suites` subfolders:

```bash
# Migrate all subfolders (dev, test, suites) to Langfuse Cloud
PYTHONPATH=rag-core/src .venv/bin/python rag-core/src/rag_core/evaluation/migrate_datasets_to_langfuse.py
```

In the Langfuse UI:
1. Navigate to **Datasets** in the sidebar.
2. You will see 3 clean virtual folders: `dev/`, `test/`, and `suites/`.
3. Open any dataset (e.g., `dev/dev_set`, `suites/science_textbook`, `test/test_set`) to inspect the stored inputs, expected ground-truth outputs, and rich metadata.

---

### Step 2: Run an Evaluation Experiment Linked to Langfuse
Execute an evaluation run directly against the dataset:

```bash
# Run 10-item smoke test against baseline pipeline
PYTHONPATH=rag-core/src .venv/bin/python rag-core/src/rag_core/evaluation/langfuse_dataset.py --split dev --mode baseline --max-items 10

# Run full evaluation on improved pipeline with Jev reranker
PYTHONPATH=rag-core/src .venv/bin/python rag-core/src/rag_core/evaluation/langfuse_dataset.py --split dev --mode improved
```

Each run:
- Creates a dataset experiment run in Langfuse.
- Links individual traces to their corresponding dataset items.
- Logs evaluation scores (`faithfulness`, `answer_relevancy`, `context_coverage`, `overall_verdict`).
- Allows side-by-side comparison between `baseline` and `improved` in the Langfuse Experiment view!

---

## 5. Cheat Sheet: Common Commands

| Workflow Goal | Command |
| :--- | :--- |
| **Sync Benchmark to Langfuse** | `PYTHONPATH=rag-core/src .venv/bin/python rag-core/src/rag_core/evaluation/migrate_datasets_to_langfuse.py` |
| **Run Baseline Langfuse Experiment** | `PYTHONPATH=rag-core/src .venv/bin/python rag-core/src/rag_core/evaluation/langfuse_dataset.py --mode baseline` |
| **Run Improved Langfuse Experiment** | `PYTHONPATH=rag-core/src .venv/bin/python rag-core/src/rag_core/evaluation/langfuse_dataset.py --mode improved` |
| **Run Smoke Test (5 items)** | `make eval-smoke` |
| **Live Dev Benchmark (Jev Judge)** | `make eval-force` |
| **Local LLM Comparative Benchmark** | `make eval-llm` |
| **Run Unit Tests (including Graph)** | `PYTHONPATH=rag-core/src .venv/bin/python -m unittest discover rag-core/tests` |

---

## 6. How to Inspect Results in Langfuse UI

1. **Traces View (`/traces`)**:
   - Filter by tag: `baseline` to see baseline performance, or `improved` to see reranked queries.
   - Click any trace to view the **Latency Waterfall**: observe exact timing breakdown across `guard_and_route`, `retrieval`, `reranking`, and `synthesis`.
2. **Scores View (`/scores`)**:
   - Review histograms of `faithfulness`, `answer_relevancy`, and `context_coverage`.
   - Inspect comments on failing items to pinpoint hallucinations.
3. **Datasets & Experiments View (`/datasets`)**:
   - Click `unified-rag-golden-benchmark`.
   - Click **Compare Runs** to see a side-by-side metric comparison between `baseline` and `improved`.

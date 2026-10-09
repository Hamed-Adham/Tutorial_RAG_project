# Unified RAG Evaluation System: Architecture, Operations & Analysis Guide

This guide explains the end-to-end evaluation architecture built into the Unified RAG system. It serves as the definitive reference for running benchmarks, tracing pipeline latency and token usage, managing the Knowledge Graph in Neo4j, and systematically diagnosing retrieval versus generation bottlenecks when developing new RAG features.

---

## 1. System Architecture: The Tri-Layered "Swiss Cheese" Model

Rather than evaluating every query through expensive, slow LLM calls, our evaluation harness implements a **cascaded tri-layered defense hierarchy**:

```
[User Query + Retrieved Chunks + Generated Answer]
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ LAYER 1: DETERMINISTIC CODE GATES (< 5ms, 0 Tokens)    │
│ • Refusal Gate: String match & Persian regex patterns  │
│ • Domain Routing Gate: Majority vote vs Expected Route │
│ • Retrieval Gate: Hit@k against ground-truth source doc│
└───────────────────────┬────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
    [Gate Resolved]              [Gate Bypassed]
  • 0 Token Cost             • Passes to Layer 2
  • < 1ms Latency            • Evaluates Semantic Fidelity
  • Immediate Pass/Fail                        │
                                               ▼
┌────────────────────────────────────────────────────────┐
│ LAYER 2: STANDARDIZED METRICS & LLM-AS-A-JUDGE         │
│ • 5-Part Rubrics (Faithfulness, Relevancy) via CoT     │
│ • Formal Ragas Metrics (Local Nine Router Gateway)     │
│ • High-Resolution Execution Latency per Judge Call     │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ LAYER 3: OBSERVABILITY & META-EVALUATION               │
│ • Langfuse OpenTelemetry Span Tracing                  │
│ • Confusion Matrix: TP, FP, TN, FN                     │
│ • Hallucination Catch Rate: TN / (TN + FP)             │
└────────────────────────────────────────────────────────┘
```

---

### Component Breakdown

#### Layer 1: Deterministic Code Gates (`rag_core/evaluation/code_gates.py`)
- **Adversarial Refusal**: Evaluates whether unanswerable or adversarial prompts are rejected with the standard Persian refusal formula:
  `"اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد"`
  Matches exact text and normalized Persian regular expressions.
- **Domain Routing**: Checks whether the retrieved chunk domain matches the expected suite domain (`science_textbook` vs `administrative_faq`).
- **Retrieval Hit@k**: Deterministically calculates whether the ground-truth document was retrieved in the top `k` candidates.
- **Architectural Impact**: Eliminates over 40 redundant LLM judge calls per 75-item test run, saving compute time and guaranteeing 100% deterministic reproducibility.

#### Layer 2: Standardized Metrics & LLM-as-a-Judge
- **5-Part Persian Rubric Judge (`rag_core/naive/judge_evaluator.py`)**:
  - Uses role personas, XML-isolated contexts, discrete 0/1 pass-fail criteria, mandatory Persian Chain-of-Thought (CoT), and strict JSON extraction.
  - Automatically times execution duration (`latency_seconds`) on every single call.
- **Ragas Metric Adapter (`rag_core/evaluation/ragas_adapter.py`)**:
  - Connects formal Ragas metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) to the local OpenAI-compatible Nine Router gateway (`http://localhost:20128/v1`).
  - Leverages local BGE-M3 embeddings (`un/gpustack/bge-m3-GGUF`) and local Gemini LLM (`gemini-cli`).

#### Layer 3: Observability with Langfuse (`rag_core/naive/pipeline.py`)
- Instrumenting every pipeline execution with OpenTelemetry-compatible traces:
  - `rag_query`: Root trace containing user query, top-k, and final answer.
  - `retrieval`: Child span recording Qdrant latency, retrieved chunk count, and similarity scores.
  - `synthesis`: Child generation span recording LLM model name, prompt tokens, completion text, and latency.
- Flushes telemetry to Langfuse host (`http://localhost:3000` or Langfuse Cloud) when `LANGFUSE_ENABLED=true`.

#### Knowledge Graph with Neo4j (`rag_core/kg/neo4j_store.py`)
- Manages the structural knowledge graph for **Synthetic Testset Generation**:
  - Nodes: `(:Document)`, `(:Chunk)`, `(:Entity)`, `(:Keyphrase)`.
  - Edges: `(:Document)-[:HAS_CHUNK]->(:Chunk)`, `(:Chunk)-[:MENTIONS]->(:Entity)`, and `(:Chunk)-[:SIMILAR_TO]->(:Chunk)`.
  - Cypher Traversal (`find_multi_hop_candidates`): Identifies pairs of chunks from different chapters sharing entities to manufacture challenging multi-hop synthetic questions.

---

## 2. Step-by-Step Execution Guide

### Prerequisites & Configuration

Check that your root `.env` file contains the required settings:

```ini
# Nine Router Local Proxy
NINE_ROUTER_BASE_URL=http://localhost:20128/v1
NINE_ROUTER_API_KEY=your_key
LLM_MODEL=gemini-cli
EMBEDDING_MODEL_NAME=un/gpustack/bge-m3-GGUF

# Qdrant Vector Store
QDRANT_PATH=data_index/qdrant_db
QDRANT_COLLECTION=unified_rag_chunks

# Langfuse Observability
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_ENABLED=false

# Neo4j Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

---

### Step 1: Running the Baseline Evaluation Benchmark

Run the full evaluation over the 75-item `dev` split:

```bash
.venv/bin/python rag-core/src/rag_core/naive/run_baseline_eval.py
```

#### Running on a Small Subset for Rapid Testing:
To test during active code development without running all 75 items, specify `max_items`:

```python
from rag_core.naive.run_baseline_eval import run_baseline_evaluation

# Run on 5 items for fast verification
report = run_baseline_evaluation(split="dev", max_items=5)
```

#### Output Report:
The evaluation produces a detailed JSON report saved to:
`rag-core/src/rag_core/evaluation/baselines/qdrant_baseline_report.json`

---

### Step 2: Running Ragas Metric Evaluation

To compute formal Ragas scores over retrieved contexts and responses:

```python
from rag_core.evaluation.ragas_adapter import LocalRagasEvaluator

evaluator = LocalRagasEvaluator()

samples = [
    {
        "question": "یکای استاندارد جرم در سیستم بین‌المللی چیست؟",
        "contexts": ["یکای استاندارد طول متر و جرم کیلوگرم (kg) است."],
        "answer": "یکای استاندارد جرم کیلوگرم است.",
        "ground_truth": "کیلوگرم (kg)"
    }
]

report = evaluator.evaluate_samples(samples)
print("Ragas Scores:", report.metrics)
print(f"Elapsed: {report.latency_seconds}s")
```

---

### Step 3: Inspecting & Populating the Neo4j Knowledge Graph

To connect and inspect graph statistics:

```python
from rag_core.kg.neo4j_store import Neo4jKnowledgeGraph

kg = Neo4jKnowledgeGraph()
if kg.connect():
    kg.init_schema()
    stats = kg.get_graph_stats()
    print("Graph Stats:", stats)
    
    # Discover multi-hop test candidates
    candidates = kg.find_multi_hop_candidates(min_shared=2, limit=10)
    for c in candidates:
        print(f"Found pair: {c['chunk_1_source']} <-> {c['chunk_2_source']}")
        print(f"Shared Entities: {c['shared_entities']}")
    kg.close()
```

---

## 3. How to Analyze Results as the RAG Developer

As you develop new features (layout-aware chunking, hybrid search, cross-encoder re-ranking), analyze the benchmark report systematically:

```
┌────────────────────────────────────────────────────────┐
│             DEVELOPER DIAGNOSTIC WORKFLOW              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 1. Inspect Overall Pass Rate vs. Baseline (62.67%)     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Check Suite-Specific Breakdown                      │
│    • adversarial_refusal: Target 100%                  │
│    • domain_routing: Target 100%                       │
│    • administrative_faq: Baseline 56.52%               │
│    • science_textbook: Baseline 40.00%                 │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Isolate Failure Mode: Retrieval vs. Generation      │
│    • Low Hit@5 / Recall -> Chunking / Embedding issue  │
│    • High Hit@5 but Low Faithfulness -> LLM Synthesis  │
│    • Low Relevancy -> Prompt Instruction / Evasiveness │
└────────────────────────────────────────────────────────┘
```

---

### Diagnosing Specific Failure Modes

#### Failure Mode A: Low Hit@5 on Complex Science Layouts
- **Symptom**: `science_textbook` pass rate drops or remains at ~40%.
- **Root Cause**: Naive fixed-size 500-character chunking breaks multi-column PDF layouts, cutting formulas and sentences across chunk boundaries.
- **Remedy**:
  1. Upgrade from naive PDF loader to **PyMuPDF / Docling Layout Parsing**.
  2. Implement **Structure-Aware Semantic Chunking** preserving headings, tables, and paragraphs intact.

#### Failure Mode B: Lexical Vocabulary Mismatch in Administrative FAQs
- **Symptom**: Conversational or colloquial Persian queries fail to retrieve formal banking/circular documents (current pass rate 56.52%).
- **Root Cause**: Pure dense vector search (BGE-M3) alone struggles with exact regulatory numbers, circular dates, and exact loan acronyms.
- **Remedy**:
  1. Implement **Hybrid Search (Sparse BM25 + Dense Vector)**.
  2. Merge ranked lists using **Reciprocal Rank Fusion (RRF)**.

#### Failure Mode C: Low Faithfulness (Hallucination)
- **Symptom**: `faithfulness` verdict is 0; thought mentions the model introduced external facts not in the context.
- **Root Cause**: Generative LLM temperature too high, or prompt lacks explicit grounding constraints.
- **Remedy**:
  1. Ensure LLM temperature is `0.0`.
  2. Verify system prompt includes: *"پاسخ باید دقیقاً و صرفاً بر اساس مستندات ارائه شده تهیه شود. از هرگونه اطلاعات خارج از مستندات اکیداً خودداری کنید."*

#### Failure Mode D: Latency Bottlenecks
- **Symptom**: `total_latency` exceeds 3.5 seconds per query.
- **Diagnosis using Telemetry**:
  - If `retrieval_latency > 0.1s`: Check Qdrant index parameters (`ef_search`, HNSW vector cache).
  - If `generation_latency > 2.5s`: Reduce context token budget (top-k from 5 to 3), or enable streaming.
  - If `judge_latency > 3.0s`: Layer 1 deterministic gates are bypassing the judge for refusal/routing, but Layer 2 judge can be parallelized via `asyncio`.

---

## 4. Comparing Progress Against Official Baselines

Every future architecture milestone must be benchmarked against our verified **Ground Zero Baseline**:

| Benchmark Suite | Baseline Score (Naive RAG) | Target Goal (Advanced RAG) | Primary Technique Driving Gain |
|:---|:---:|:---:|:---|
| **`domain_routing`** | **100.00%** (11/11) | **100.00%** | Retain Qdrant metadata payload filtering |
| **`adversarial_refusal`**| **100.00%** (11/11) | **100.00%** | Layer 1 Deterministic Code Gate |
| **`administrative_faq`** | **56.52%** (13/23) | **>= 85.00%** | Hybrid Search (BM25 + Dense) via RRF |
| **`science_textbook`**   | **40.00%** (12/30) | **>= 80.00%** | Structure-Aware Layout Chunking + Cross-Encoder |
| **Overall Dev Score**    | **62.67%** (47/75) | **>= 85.00%** | Full Advanced RAG Pipeline |
| **Mean Latency**         | **8.75s**          | **< 2.5s**    | Layer 1 bypass + streaming synthesis |

---

## 5. Summary & Best Practices

1. **Gate before Judging**: Never invoke an LLM judge on responses that fail deterministic Layer 1 checks.
2. **Always Time Evaluations**: Every judge verdict records `latency_seconds` to catch evaluation drift and local gateway throttling.
3. **Trace Everything in Langfuse**: When investigating production bugs, open the corresponding Langfuse trace to inspect the exact prompt, retrieved chunks, and latency breakdown.
4. **Use Neo4j for Multi-Hop Discovery**: Traverse shared entities across documents in Neo4j to manufacture challenging synthetic test cases that stress-test your retrieval engine.

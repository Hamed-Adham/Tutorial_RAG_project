# Quickstart: Evaluation Harness & Verification Guide

This quickstart guide provides step-by-step instructions to validate the Tri-Layered "Swiss Cheese" Evaluation Engine, Neo4j Knowledge Graph Store, and Langfuse Observability for the Unified RAG system.

---

## 1. Prerequisites & Services Setup

### 1.1 Python Virtual Environment
Verify Python 3.11+ is active in the repository root:
```bash
source .venv/bin/activate
export PYTHONPATH=rag-core/src
```

### 1.2 Start Neo4j Graph Database
Run or verify the Neo4j Docker container:
```bash
docker run -d --name neo4j-rag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```
Verify container status:
```bash
docker ps --filter "name=neo4j-rag"
```
The web console is accessible at `http://localhost:7474`.

### 1.3 Local LLM Gateway (Nine Router)
Ensure the local Nine Router gateway is running at `http://localhost:20128/v1`:
```bash
curl http://localhost:20128/v1/models
```

### 1.4 Langfuse Observability Server
Ensure the Langfuse server is accessible at `http://localhost:3000` (or via Docker):
```bash
curl -I http://localhost:3000
```
Active credentials configured in `.env`:
- `LANGFUSE_BASE_URL=http://localhost:3000`
- `LANGFUSE_PUBLIC_KEY=pk-lf-ebbaea38-43ad-4265-947e-ef0eee663c8b`
- `LANGFUSE_SECRET_KEY=sk-lf-c5b79ad2-122e-490b-8eaa-9adf34f0d4e6`
- `LANGFUSE_ENABLED=true`

---


## 2. Component Verification Scenarios

### Scenario A: Layer 1 Deterministic Code Gates (< 5ms, 0 Tokens)
Test that Persian refusal patterns and out-of-scope queries are intercepted deterministically without invoking the LLM:

```bash
python -c "
from rag_core.evaluation.code_gates import evaluate_code_gate

# Test 1: Refusal query
res = evaluate_code_gate('لطفا به من بگویید پایتخت کجاست', 'اطلاعاتی در متن موجود نیست', 'science')
print(f'Refusal Gate: passed={res.passed}, latency={res.latency_seconds * 1000:.2f}ms, reason={res.reason}')

# Test 2: In-scope query (passes to Layer 2)
res2 = evaluate_code_gate('وظیفه میتوکندری چیست؟', 'میتوکندری وظیفه تولید ATP را دارد.', 'science')
print(f'Content Gate: passed={res2.passed}, latency={res2.latency_seconds * 1000:.2f}ms, reason={res2.reason}')
"
```
**Expected Outcome**: Execution latency is `< 5ms` with zero token consumption.

---

### Scenario B: Neo4j Knowledge Graph Multi-Hop Discovery
Test Neo4j connection and multi-hop candidate discovery query:

```bash
python -c "
from rag_core.kg.neo4j_store import Neo4jKnowledgeGraph

kg = Neo4jKnowledgeGraph()
candidates = kg.find_multi_hop_candidates(limit=5)
print(f'Neo4j connected: {kg.driver is not None}')
print(f'Multi-hop candidates discovered: {len(candidates)}')
kg.close()
"
```
**Expected Outcome**: Returns graph connection status and candidate pairs for synthetic question generation. If Neo4j is offline, falls back gracefully without unhandled exceptions.

---

### Scenario C: Benchmark Evaluation with Judge Latency Tracking
Run an evaluation subset across the 100-item Golden Benchmark:

```bash
python rag-core/src/rag_core/naive/run_baseline_eval.py --suite adversarial_refusal --limit 5
```
**Expected Outcome**:
- Layer 1 Code Gate bypasses negative samples deterministically.
- Layer 2 LLM Judge outputs `latency_seconds` per item.
- Final summary table reports accuracy, gate pass rate, and average judge latency.

---

### Scenario D: Local Ragas Adapter Execution
Validate the 4 core Ragas metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) via the local gateway adapter:

```bash
python -c "
from rag_core.evaluation.ragas_adapter import LocalRagasEvaluator

evaluator = LocalRagasEvaluator()
results = evaluator.evaluate(
    query='وظیفه میتوکندری چیست؟',
    response='میتوکندری اندامکی است که وظیفه تولید انرژی سلولی (ATP) را بر عهده دارد.',
    contexts=['میتوکندری به عنوان موتورخانه سلول شناخته شده و مسئول سنتز مولکول‌های ATP در تنفس سلولی است.'],
    ground_truth='میتوکندری مسئول تولید مولکول‌های ATP و تامین انرژی سلولی است.'
)
print('Ragas Metrics:', results)
"
```
**Expected Outcome**: All 4 metrics compute numerical scores bounded in `[0.0, 1.0]`.

---

### Scenario E: Developer Verification Hierarchy (`make test`, `make test-gates`, `make check`)
Validate the 3 tiers of testing using standardized `make` targets:

```bash
# 1. Tier 1: Unit & Component Tests (< 1s, 0 tokens)
make test

# 2. Tier 2: 100-Item Deterministic Code Gates (~1-2s, 0 tokens)
make test-gates

# 3. Pre-Commit / PR Unified Check (Tier 1 + Tier 2 in < 2s)
make check
```
**Expected Outcome**:
- `make test` executes pure Python unit tests in `rag-core/tests/` in `< 0.1s`.
- `make test-gates` evaluates all 100 items from `golden_set.jsonl` through Layer 1 Code Gates, verifying refusal and routing rates without calling any LLM.
- `make check` executes both targets sequentially with exit code 0.

---

### Scenario F: Langfuse Tracing & Trace Score Verification
Validate that Langfuse traces, nested spans, and evaluation scores are correctly structured:

```bash
python -c "
from rag_core.observability.langfuse import LangfuseTracker

tracker = LangfuseTracker()
with tracker.trace_query(query='تست ردیابی سیستم', metadata={'test': True, 'suite': 'system_check'}) as trace:
    with tracker.span('retrieval', input_data={'top_k': 3}) as span:
        pass
    with tracker.span('synthesis', input_data={'prompt': 'سنتز پاسخ'}) as span:
        pass
tracker.flush()
print('Langfuse tracing and span execution completed successfully!')
"
```
**Expected Outcome**:
- When `LANGFUSE_ENABLED=true` and server is reachable, traces and spans appear in the Langfuse UI at `http://localhost:3000`.
- When offline or unreachable, falls back cleanly with zero exceptions.

---

### Scenario G: Synchronous Cascade Hard-Deletion Verification
Test that document deletion in the manifest catalog triggers immediate cascade purging across secondary stores:

```bash
python -c "
from rag_core.manifest.catalog import PostgresDocumentCatalog

catalog = PostgresDocumentCatalog()
# Verify cascade delete interface
success = catalog.cascade_delete_document('doc-test-001')
print(f'Cascade purge execution: success={success}')
"
```
**Expected Outcome**: Returns `success=True` and verifies vectors are removed from Qdrant, tokens purged from BM25, and nodes deleted from Neo4j without orphaned entities.

---

### Scenario H: Persian Text Normalization & Lexical BM25 Tokenization
Validate that Persian compound tokens and ZWNJ characters are correctly normalized:

```bash
python -c "
import hazm
normalizer = hazm.Normalizer()
text = 'شرکت‌های دانش‌بنیان در حوزه‌های فناوری نوین'
normalized = normalizer.normalize(text)
print('Normalized Persian Text:', normalized)
"
```
**Expected Outcome**: ZWNJ characters are standardized and unified for BM25 inverted index tokenization.

---

## 3. Related Design Artifacts

- Specification: [spec.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/spec.md)
- Implementation Plan: [plan.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/plan.md)
- Data Models: [data-model.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/data-model.md)
- Backend OpenAPI Contract: [backend-api.yaml](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/contracts/backend-api.yaml)
- Core Protocols: [rag-core-interfaces.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/specs/001-enterprise-rag-system/contracts/rag-core-interfaces.md)


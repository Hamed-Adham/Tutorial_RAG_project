# Research & Architectural Decisions: Stage 10 Evaluation & Knowledge Graph

This document records the architectural research, design rationales, and rejected alternatives for Stage 10 (Evaluation, Observability & Guardrails) and Stage 4/10 (Knowledge Graph).

---

## 1. Tri-Layered "Swiss Cheese" Evaluation Strategy

- **Decision**: Adopt a 3-layer tiered evaluation architecture:
  1. **Layer 1: Deterministic Code Gates**: Heuristic regex pattern matching, refusal dictionary lookups, and lexical domain checks executing in `< 5ms` with zero token cost.
  2. **Layer 2: LLM-as-a-Judge & Ragas**: 5-part Chain-of-Thought rubrics and formal Ragas metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) recording execution latency (`latency_seconds`).
  3. **Layer 3: Human Meta-Evaluation**: Periodic validation calculating confusion matrices, Hallucination Catch Rate, and Judge-Human alignment.
- **Rationale**: Evaluating every single query with an LLM judge creates prohibitive token costs and slows down CI benchmarking to unacceptable levels. Placing deterministic code gates as an initial filter intercepts trivial refusals and out-of-scope queries in sub-millisecond time.
- **Alternatives Considered**:
  - *Pure LLM-as-a-Judge for all items*: Evaluated during initial benchmarking; took several minutes for small batches and incurred unnecessary API latency for simple negative samples.
  - *Pure Deterministic / Rule-based*: Insufficient for evaluating semantic nuance, faithfulness, or multi-sentence reasoning.

---

## 2. Knowledge Graph Technology: Neo4j Store

- **Decision**: Deploy Neo4j (Docker container `neo4j-rag`, Python driver `neo4j` 6.3.1) with a schema representing `Document`, `Chunk`, `Entity`, and `Keyphrase` nodes, linked by `HAS_CHUNK`, `MENTIONS`, and `SIMILAR_TO` edges.
- **Rationale**: Multi-hop synthetic question generation (Evol-Instruct) requires discovering entity bridges across chunks that originate from different sections or documents. Cypher graph traversal makes this query concise, indexed, and efficient.
- **Alternatives Considered**:
  - *In-memory NetworkX graph*: Fast for tiny experiments, but lacks persistence, transactional guarantees, and scalability to thousands of chunks.
  - *PostgreSQL Recursive Common Table Expressions (CTEs)*: Relational multi-hop queries become convoluted and inefficient compared to native graph graph traversal.

---

## 3. Tooling Scope Boundary

- **Decision**: Focus evaluation tooling strictly on **Ragas (0.4.3)** and **Langfuse (4.16.0)**, explicitly deferring **Patronus AI** and **DeepEval** for future milestone phases.
- **Rationale**:
  - Patronus AI requires proprietary enterprise cloud accounts and external API keys that break local private execution.
  - DeepEval introduces redundant abstraction layers over custom judge prompts while increasing dependency footprint.
  - Ragas + Langfuse + custom Code Gates provide 100% coverage of the constitution's quality requirements with zero licensing cost and full local execution over the Nine Router gateway.
- **Alternatives Considered**:
  - *DeepEval*: Deemed promising for unit test integration, but currently redundant with our custom Layer 1/2 harness.
  - *Patronus Lynx/HaluGate*: Postponed until production enterprise guardrails phase.

---

## 4. Execution Duration & Latency Profiling

- **Decision**: Mandate high-resolution execution duration logging (`latency_seconds`) on every LLM-as-a-Judge invocation and evaluation span.
- **Rationale**: Latency is a primary engineering KPI. Tracking judge latency allows performance regression detection, timeouts, and identification of bottlenecks in synthetic evaluation pipelines.

---

## 5. Langfuse AI Tracing Architecture & Credential Binding

- **Decision**: Standardize on Langfuse SDK v2.x with explicit dual environment variable binding (`LANGFUSE_BASE_URL` or `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`), `@observe()` span instrumentation, and trace score ingestion for Layer 1 Code Gate & Layer 2 LLM Judge verdicts.
- **Rationale**:
  - Centralizes observability across token consumption, cost calculation, and latency attribution in a self-hosted dashboard at `http://localhost:3000`.
  - Associating empirical evaluation scores (e.g., `faithfulness`, `gate_pass`, `answer_relevancy`) directly with trace IDs enables instant regression detection across prompt or pipeline iterations.
  - Graceful fallback logic ensures that offline development, continuous integration, or unconfigured environments run with zero crash risk.
- **Alternatives Considered**:
  - *Weights & Biases Weave / Arize Phoenix*: Required additional dependencies and lacked the native evaluation dataset and score management features natively supported by self-hosted Langfuse.
  - *Raw OpenTelemetry Console Exporter*: Produced voluminous unindexed logs without an interactive trace visualizer or token cost attribution.

---

## 6. Persian NLP Normalization & Lexical BM25 (Hazm / Parsivar)

- **Decision**: Integrate external Persian NLP tooling (`hazm` / `parsivar`) for text normalization, stemming, lemmatization, and zero-width non-joiner (ZWNJ) standardization during document parsing and BM25 tokenization.
- **Rationale**: The Golden Benchmark and target corpora feature extensive Persian text. Standard whitespace/punctuation tokenization fails on compound Persian tokens (e.g., `دانش‌بنیان` vs `دانش بنیان` and `ی/ي` character variations). Full lemmatization ensures accurate keyword recall in BM25.
- **Alternatives Considered**:
  - *Generic whitespace tokenization*: Fails completely on compound Persian words and variable Unicode codepoints.
  - *Custom in-tree regex normalizer*: Lightweight, but lacks grammatical stemmers and lemmatizers necessary for high lexical recall across varied morphological word forms.

---

## 7. Multi-Index Document Deletion Lifecycle

- **Decision**: Enforce immediate synchronous cascade hard-deletion across secondary indices (Qdrant vectors, BM25 inverted index, Neo4j knowledge graph nodes) whenever a document is removed from the PostgreSQL manifest catalog. If deletion fails on any downstream store, roll back the manifest transaction and raise a deletion error.
- **Rationale**: Guarantees zero index drift and prevents deleted or restricted documents from ever being surfaced in vector or BM25 retrieval results. Eliminates the complexity of maintaining query-time tombstone filters across heterogeneous stores.
- **Alternatives Considered**:
  - *Soft-delete with query-time payload filtering*: Postpones deletion, but adds query filtering overhead to every Qdrant and BM25 request, risking retrieval leakage if filters are omitted.
  - *Scheduled full re-indexing*: Causes long windows where stale or deleted documents remain searchable.

---

## 8. Scope Decision: Text-to-SQL (NL2SQL) Descoping

- **Decision**: Formally descope natural language to SQL generation (NL2SQL), database schema introspection, and SQL AST execution from this project. The multi-tool agent will focus strictly on Document RAG, GraphRAG, and Web Search.
- **Rationale**: Keeps project focus tightly aligned with enterprise unstructured and graph retrieval-augmented generation. Prevents split architectural focus between relational query synthesis and complex multi-hop document retrieval.
- **Alternatives Considered**:
  - *Include isolated SQLite/Postgres analytical DB*: Deemed out of scope for the current milestone per user directive.

---

## 9. Structure-Aware Chunking Boundaries

- **Decision**: Standardize atomic/child chunks at ~512 tokens (with up to 1024 tokens maximum for cohesive semantic sections). Parent synthesis chunks dynamically encapsulate the *N* constituent child chunks spanning the enclosing logical heading section.
- **Rationale**: 512 tokens provides an optimal balance for BGE-M3 dense semantic embedding and cross-encoder re-ranking precision. Modern LLM context windows readily process parent chunks consisting of multiple child chunks (~1500–3000 tokens) providing complete section context without truncation.
- **Alternatives Considered**:
  - *Fixed 256-token micro-chunks*: Dilutes paragraph coherence and requires higher retrieval overfetch multipliers.
  - *Sentence-level chunking*: Produces noisy retrieval candidates requiring excessive window expansion.

---

## 10. Adversarial Refusal Contract for Layer 1 Deterministic Code Gates

- **Decision**: Standardize on a deterministic Persian refusal message paired with structured metadata (`is_refusal: true`, `refusal_reason: ...`) for Layer 1 Code Gate rejections (< 5ms, 0 tokens), while Layer 2 LLM-as-a-Judge evaluates grounded fallback statements.
- **Rationale**: Layer 1 Code Gates must execute deterministically without invoking an LLM. Standardized templates guarantee zero token cost, instant response latency, and unambiguous automated scoring against Suite 4 of the Golden Benchmark.
- **Alternatives Considered**:
  - *Dispatching to LLM for customized refusal wording*: Violates the < 5ms / 0-token guarantee and incurs unnecessary LLM costs for obvious injection or out-of-scope prompts.


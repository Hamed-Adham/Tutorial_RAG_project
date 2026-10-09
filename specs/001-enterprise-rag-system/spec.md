# Feature Specification: Enterprise RAG + Text-to-SQL System

**Feature Branch**: `001-enterprise-rag-system`  
**Created**: 2026-10-01 | **Updated**: 2026-10-01 (Unified 10-Stage Architecture Integration)  
**Status**: Ready for Planning  
**Input**: Comprehensive technical blueprint synthesized from `PROJECT_DOCUMENTATION (1).pdf`, 100k-document production explainer (Ardya Dipta), and Advanced RAG Techniques (Weaviate eBook / DataPlusScience).

---

## Clarifications

### Session 2026-10-05
- Q: Should the document parsing and BM25 lexical retrieval pipelines incorporate dedicated Persian text normalization, such as zero-width non-joiner (ZWNJ) handling and character unification? (FR-015) → A: Integrate an external Persian NLP library (such as `hazm` or `parsivar`) with full stemming and lemmatization for Persian text parsing and BM25 tokenization.
- Q: How should document deletion or tombstoning in the PostgreSQL manifest catalog propagate to the multi-index views (Qdrant, BM25, and Neo4j)? (FR-003) → A: Immediate synchronous cascade hard-delete that removes vectors from Qdrant, entries from BM25, and nodes from Neo4j upon document deletion in the manifest catalog.
- Q: What target database and schema environment should the Text-to-SQL tool connect to for schema introspection and query execution? (FR-025) → A: Text-to-SQL (NL2SQL) tool and feature are explicitly descoped from this project; the system focuses on Document RAG, GraphRAG, and Web search.
- Q: What target token budget and overlap boundaries should be configured for structure-aware child chunks and parent synthesis chunks? (FR-007) → A: Child/atomic chunks target ~512 tokens (or up to 1024 tokens max for cohesive semantic sections); parent chunks dynamically encapsulate the N constituent child chunks spanning the enclosing logical section.
- Q: When an adversarial, out-of-scope, or unanswerable query triggers refusal via Layer 1 Deterministic Code Gates or Layer 2 LLM generation, what format should the refusal response take? (FR-034) → A: Standardized deterministic refusal messages with structured metadata (`is_refusal: true`, `refusal_reason: ...`) for Layer 1 Code Gates, and grounded fallback statements for Layer 2 synthesis.

---

## Unified 10-Stage Architecture Framework

| Stage | Name | Modular Options Available |
| :--- | :--- | :--- |
| **Stage 1** | **Ingestion & Manifest Catalog** | **Option 1A**: Postgres Manifest Catalog with SHA-256 exact hash collapse & MinHash near-duplicate clustering (idempotency, lineage, and immediate synchronous cascade hard-deletion across Qdrant, BM25, and Neo4j upon document removal)<br>**Option 1B**: LlamaHub / SimpleDirectoryReader connectors for cloud and local fileshares |
| **Stage 2** | **Tiered Parsing into Canonical IR** | **Option 2A**: Tiered routing: Tier 0 (Text-native + Docling/PyMuPDF layout), Tier 1 (OCR with confidence scores), Tier 2 (VLM captions for diagrams/tables) emitting unified block IR (`type`, `text`, `page`, `bbox`)<br>**Option 2B**: Multimodal direct embedding (ColPali/ColQwen visual patches)<br>**Option 2C**: LlamaParse / Unstructured for complex PDF table extraction |
| **Stage 3** | **Chunking Strategies** | **Option 3A**: Structure-aware splitting on heading hierarchy with breadcrumb prepending (`Doc > Section > Subsection (page)`), ~512-token atomic chunks (up to 1024 tokens max for cohesive semantic sections), and intact tables<br>**Option 3B**: Parent-Child chunking (~512-token child chunks indexed for retrieval matching, parent chunks dynamically aggregating N child chunks passed to LLM)<br>**Option 3C**: Semantic chunking based on cosine distance threshold shifts between sentences (max 1024 tokens)<br>**Option 3D**: Contextual Retrieval (Anthropic-style 1-2 sentence LLM prefix situating chunk context)<br>**Option 3E**: Recursive / Document-based chunking with hierarchical delimiters<br>**Option 3F**: Late chunking using long-context models |
| **Stage 4** | **Enrichment, Embedding & Multi-Index Views** | **Option 4A**: Multi-index derived views rebuildable from IR: Dense HNSW vector store, BM25 inverted index, and optional GraphRAG entity graph<br>**Option 4B**: Two-tier metadata enrichment (system metadata + LLM document-level summary/topic extraction)<br>**Option 4C**: GraphRAG entity & relationship extraction with Leiden community detection<br>**Option 4D**: Dense embedding models (BGE-M3 with int8 quantization) or domain-fine-tuned embeddings |
| **Stage 5** | **Pre-Retrieval & Query Understanding** | **Option 5A**: Query Rewriting (Rewrite-Retrieve-Read to resolve pronouns/context)<br>**Option 5B**: Query Expansion (multi-query generation & parallel retrieval)<br>**Option 5C**: Hypothetical Document Embeddings (HyDE)<br>**Option 5D**: Query Decomposition (sub-queries for multi-part questions)<br>**Option 5E**: Query Routing & Soft Filter Extraction (closed-schema metadata filter extraction + intent classification) |
| **Stage 6** | **Multi-Pronged Retrieval & Outlier Handling** | **Option 6A**: Session-derived mandatory security pre-filtering (Tenant & ACL filters enforced during HNSW traversal)<br>**Option 6B**: Lexical BM25 retrieval with identifier-preserving analyzer (error codes, part numbers), Hazm/Parsivar Persian NLP normalization and lemmatization, and field boosts (title^3, breadcrumb^2)<br>**Option 6C**: Dense semantic HNSW vector search with model query prefix and tuned `ef_search` beam width (overfetching 100-200 candidates)<br>**Option 6D**: Graph local entity expansion (1-2 hop neighborhood retrieval)<br>**Option 6E**: Outlier exclusion via fixed top-k, distance thresholding, or Autocut (clustering distance jumps) |
| **Stage 7** | **Fusion & Post-Retrieval Re-ranking** | **Option 7A**: Reciprocal Rank Fusion (RRF: `score = sum(1 / (60 + rank))`) merging BM25, Dense, and Graph lists<br>**Option 7B**: Weighted score fusion (`alpha * dense + (1 - alpha) * bm25`)<br>**Option 7C**: Cross-Encoder Re-ranking (BGE-Reranker v2 M3, Zerank 1 Small, Cohere Rerank) re-scoring top 50-100 down to 5-10<br>**Option 7D**: Document deduplication and chunk capping (maximum 2-3 chunks per source document) |
| **Stage 8** | **Context Post-Processing & Refinement** | **Option 8A**: Parent / Sentence Window Expansion (retrieve small chunk, expand to full parent or sentence window)<br>**Option 8B**: Contextual Compression (lexical or embedding-based compressor pruning extraneous tokens)<br>**Option 8C**: `LongContextReorder` (mitigating lost-in-the-middle degradation)<br>**Option 8D**: Bounding box (`bbox`) and page-level source citation assembly |
| **Stage 9** | **Generation, Prompt Engineering & Agent Orchestration** | **Option 9A**: Prompt engineering strategies (Grounded system prompt, Chain of Thought [CoT], Tree of Thoughts [ToT], ReAct)<br>**Option 9B**: Synthesis modes (Compact, Refine, Tree Summarize)<br>**Option 9C**: LangGraph StateGraph multi-tool agent (routing between Document RAG, GraphRAG, and Web search; Text-to-SQL descoped)<br>**Option 9D**: Model selection (GPT-4o for complex reasoning vs fine-tuned Llama-3.1 for on-premise privacy) |
| **Stage 10** | **Evaluation, Observability & Guardrails** | **Option 10A**: Tri-Layered "Swiss Cheese" Evaluation Harness: Layer 1 Deterministic Code Gates (<5ms, 0 tokens) for Persian refusal and domain routing; Layer 2 LLM-as-a-Judge with 5-part CoT rubrics, execution latency tracking, and Ragas metrics (Faithfulness >= 0.80, Relevancy >= 0.75, Precision >= 0.75, Recall >= 0.70); Layer 3 Human Meta-Evaluation (confusion matrix, Hallucination Catch Rate)<br>**Option 10B**: Langfuse OpenTelemetry tracing (spans for LLM synthesis, Qdrant retrieval, tokens, costs, latency profiling)<br>**Option 10C**: Neo4j Knowledge Graph for Synthetic Testset Generation (document-chunk-entity graph traversal for multi-hop candidate discovery)<br>**Option 10D**: Enterprise security guardrails (Microsoft Presidio PII redaction, structured audit logging)<br>**Option 10E**: Performance caching (Redis content-hash embedding cache, TTL LLM response cache) |

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Core Document Ingestion, Manifest Cataloging, Retrieval & Measurable Answer Synthesis (Priority: P1 - MVP)

As an enterprise knowledge worker or researcher,  
I want documents to be cataloged in an idempotent manifest, parsed into canonical IR blocks, chunked with hierarchical breadcrumbs, embedded with BGE-M3, and retrieved to produce grounded, cited answers backed by automated RAGAS evaluation,  
So that duplicate documents are eliminated, retrieval is precise, and quality is objectively measurable from Day 1.

**Why this priority**: Establishes the core foundation across Stages 1, 2, 3, 4, 6, 8, 9, and 10. Delivers a measurable end-to-end MVP.

**Independent Test**: Load documents into the manifest, verify SHA-256 deduplication, chunk with breadcrumbs, execute vector retrieval, generate an answer with page citations, and run RAGAS evaluation outputting the 4 core metrics.

**Acceptance Scenarios**:
1. **Given** duplicate files across folders, **When** the manifest cataloger runs (Stage 1), **Then** exact duplicates collapse to one canonical row and near-duplicates are flagged via MinHash.
2. **Given** documents in the catalog, **When** tiered parsing executes (Stage 2), **Then** content is parsed into canonical IR blocks (`type`, `text`, `page`, `bbox`).
3. **Given** IR blocks, **When** chunking runs (Stage 3), **Then** chunks maintain heading hierarchy, prepend breadcrumbs, and link child chunks to parent documents.
4. **Given** indexed chunks, **When** a user asks a question, **Then** vector retrieval returns top candidates, generates a grounded response with page-level citations, and traces all spans in Langfuse.
5. **Given** the 100-item multi-suite Golden Benchmark, **When** the evaluation harness runs, **Then** Layer 1 Deterministic Code Gates resolve refusal and domain routing items in < 5ms with 0 LLM tokens, and Layer 2 evaluates semantic faithfulness and relevancy with explicit execution latency tracking.
6. **Given** document chunks ingested into Neo4j, **When** multi-hop candidate discovery runs, **Then** Cypher graph traversal identifies pairs of chunks sharing entity relationships across chapters/documents for synthetic testset generation.
7. **Given** pipeline query execution, **When** Langfuse is enabled, **Then** `rag_query` traces, `retrieval` spans, and `synthesis` generation spans are recorded with latency and token attribution.

---

### User Story 2 - Hybrid Search (BM25 + Dense), RRF, Cross-Encoder Re-ranking & Query Transformation (Priority: P2)

As a knowledge user querying domain-specific terminology, acronyms, or error codes,  
I want the system to understand query intent, rewrite or decompose queries, run parallel BM25 and dense searches with Autocut outlier exclusion, and re-rank with a cross-encoder,  
So that exact part numbers and semantic context are both accurately retrieved without noise.

**Why this priority**: Implements Stages 5, 6, 7, and 8. Addresses lexical mismatch, out-of-vocabulary terms ("ERR-4092"), and ranking accuracy.

**Independent Test**: Execute complex queries containing technical identifiers and natural language phrasing; confirm parallel BM25 + dense execution, RRF fusion, Autocut filtering, and cross-encoder re-ranking to achieve higher Context Precision and Recall.

**Acceptance Scenarios**:
1. **Given** a query with specific codes, **When** retrieval runs (Stage 6), **Then** BM25 captures exact terms while dense search captures semantic intent, each overfetching candidates.
2. **Given** multiple candidate lists, **When** fusion executes (Stage 7), **Then** Reciprocal Rank Fusion (RRF) merges candidates, Autocut eliminates score drop-offs, and BGE-Reranker v2 M3 re-scores the top candidates down to 5-10.
3. **Given** an ambiguous or multi-part query, **When** query transformation triggers (Stage 5), **Then** query rewriting or sub-question decomposition produces optimized sub-queries.
4. **Given** retrieved child chunks, **When** context post-processing runs (Stage 8), **Then** child chunks expand to parent context or sentence windows, reordered via `LongContextReorder`.

---

### User Story 3 - Autonomous Multi-Tool Agent with Document RAG, GraphRAG & Web Search (Priority: P3)

As a researcher or decision maker,  
I want an intelligent agent to route queries between unstructured document RAG, multi-hop entity graphs (GraphRAG), and external web search,  
So that deep corpus answers, entity-spanning relationships, and live web knowledge are seamlessly synthesized in one conversation.

**Why this priority**: Unlocks multi-hop relational synthesis across Stages 4C, 6D, and 9C while preserving tool separation and clean agent orchestration.

**Independent Test**: Submit document questions (Document RAG), multi-hop entity questions (GraphRAG), and external/current information questions (Web search), confirming proper tool routing and coherent multi-source answers.

**Acceptance Scenarios**:
1. **Given** a multi-hop corpus-spanning question, **When** the LangGraph agent routes the request, **Then** it delegates to GraphRAG for entity extraction and community summaries.
2. **Given** a query requiring external real-time verification, **When** routed, **Then** it delegates to the Web search tool.
3. **Given** an unstructured document query, **When** routed, **Then** it delegates to the core Document RAG pipeline.

---

## Out of Scope

- **Text-to-SQL (NL2SQL) Tooling**: Natural language to SQL query generation, database schema introspection, and SQL AST execution are explicitly descoped from this project milestone and deferred to future releases.
- **Direct Database Mutation**: No agent tool is permitted to execute database write or DDL operations.

---

### User Story 4 - Enterprise Security, Performance Caching, FastAPI & Full-Stack UI (Priority: P4)

As an enterprise administrator and user,  
I want a responsive Next.js 14+ UI with CopilotKit chat, an evaluation dashboard, Redis caching, Presidio PII redaction, and Qdrant payload ACL filtering,  
So that data privacy is guaranteed, latency and cost are minimized, and operational visibility is accessible to the entire team.

**Why this priority**: Implements production-grade enterprise requirements across Stages 10C, 10D, backend API, and web interface.

**Independent Test**: Verify PII redaction during ingestion/querying, verify ACL payload filtering per user role, verify Redis cache hits on repeat queries, and inspect real-time metrics in the Next.js evaluation dashboard.

**Acceptance Scenarios**:
1. **Given** text containing PII, **When** ingested or queried, **Then** Presidio redacts sensitive entities before storage or LLM transmission.
2. **Given** an authenticated user session, **When** queries execute, **Then** ACL filters are pushed into the vector and BM25 retrievers, preventing unauthorized document leakage.
3. **Given** repeated queries, **When** cached, **Then** Redis embedding and LLM response caches return results with >= 80% latency reduction.
4. **Given** the Next.js frontend, **When** navigating to the evaluation dashboard, **Then** historical RAGAS metric trends, baseline comparisons, and quality gate pass/fail statuses are rendered.

---

## Edge Cases

- **Out-of-Vocabulary / Unrelated Query**: System must state lack of context rather than hallucinating.
- **Selective ACL Filter Starvation**: When an ACL filter matches only a handful of documents, HNSW graph traversal fallback to exact scan must prevent empty results.
- **Autocut Edge Cases**: If all returned chunks have identical distances or high distance drop-off at rank 1, fallback to minimum `k=3` chunks.
- **SQL Injection**: Any destructive SQL statement or chained query (`SELECT ...; DROP TABLE ...`) must be blocked by the AST parser before execution.
- **Malformed Documents & OCR Failures**: Parsing probe failure on Tier 0 must automatically fall back to Tier 1 OCR or isolate file errors in the manifest without breaking batch jobs.
- **Lost-in-the-Middle Phenomenon**: Long context windows must undergo `LongContextReorder` placing primary evidence at the boundaries.
- **Cascade Deletion Failure**: If synchronous deletion fails across secondary indices (Qdrant vectors, BM25 index, Neo4j nodes), the manifest catalog transaction must roll back and report a deletion failure to prevent index drift.

---

## Development Methodology: Component Maximization Precedes Component Accumulation

The system enforces a strict hill-climbing engineering discipline:
1. **Maximize Existing Components First**: Never introduce a new architectural component, model layer, or pipeline stage until existing components have been exhaustively tuned and evaluated against their performance ceiling.
2. **Empirical Gate for New Components**: A new component (e.g. cross-encoder reranking, query transformation, hybrid BM25, GraphRAG) is only permitted after error analysis on the Golden Benchmark proves that current components cannot resolve the failure modes, and the proposed component yields an empirical evaluation score delta that justifies its computational and latency cost.

---

## Functional Requirements (FR-001 to FR-030)

### Architecture & Foundation
- **FR-001**: System MUST enforce a 3-layer decoupled architecture (`rag-core` Python package, `backend` FastAPI server, `frontend` Next.js 14+ app).
- **FR-002**: System MUST structure the pipeline according to the 10-Stage Architecture Framework, supporting modular options per stage.
- **FR-002b**: System development MUST enforce Component Maximization Precedes Component Accumulation: developers and agents MUST exhaustively tune and benchmark existing components (chunk sizing, Qdrant HNSW `ef_search`, retrieval `top_k`, prompt engineering) before introducing new architectural components or model layers.

### Stages 1 to 4: Ingestion, Parsing, Chunking & Multi-Index
- **FR-003**: System MUST provide a document manifest catalog in PostgreSQL tracking stable `doc_id`, SHA-256 hash, tenant, ACL, and status, and MUST execute immediate synchronous cascade hard-deletion across Qdrant, BM25, and Neo4j whenever a document is removed (Stage 1).
- **FR-004**: System MUST collapse exact duplicate documents based on SHA-256 and identify near-duplicates via MinHash over normalized text (Stage 1).
- **FR-005**: System MUST implement tiered parsing with probe routing: Tier 0 text-native layout, Tier 1 OCR with confidence scores, and Tier 2 VLM captions (Stage 2).
- **FR-006**: System MUST output a standardized canonical IR representing blocks with `type`, `text`, `page`, and `bbox` (Stage 2).
- **FR-007**: System MUST support structure-aware chunking on headings with hierarchical breadcrumbs (`Doc > Section > Subsection (page)`), targeting ~512 tokens per atomic chunk (up to 1024 tokens max for cohesive semantic sections) and intact table preservation (Stage 3).
- **FR-008**: System MUST support parent-child chunking where child chunks (~512 tokens, max 1024) are indexed for retrieval and linked to parent chunks (dynamically aggregating their N constituent children across the enclosing heading section) for LLM synthesis (Stage 3).
- **FR-009**: System MUST support semantic chunking via cosine distance shifts and contextual retrieval prefixes (Stage 3).
- **FR-010**: System MUST derive rebuildable views from the canonical IR: Qdrant HNSW vector index, BM25 inverted index, and optional GraphRAG entity graph (Stage 4).
- **FR-011**: System MUST embed chunks using `BGE-M3` (`BAAI/bge-m3`) with int8 quantization support (Stage 4).

### Stages 5 to 8: Pre-Retrieval, Retrieval, Fusion & Post-Processing
- **FR-012**: System MUST provide query transformation capabilities: query rewriting, expansion, HyDE, and sub-question decomposition (Stage 5).
- **FR-013**: System MUST extract structured metadata filters from queries against a closed schema and perform query routing (Stage 5).
- **FR-014**: System MUST enforce session-derived tenant and ACL security filters during retrieval (Stage 6).
- **FR-015**: System MUST execute parallel lexical BM25 retrieval with identifier-preserving tokenization, Hazm/Parsivar-based Persian NLP normalization, stemming, and lemmatization, and dense HNSW retrieval with model-specific query prefixes (Stage 6).
- **FR-016**: System MUST support outlier exclusion techniques including fixed top-k, distance thresholds, and Autocut distance jump clustering (Stage 6).
- **FR-017**: System MUST fuse multi-retriever candidate lists using Reciprocal Rank Fusion (RRF: `score = sum(1 / (60 + rank))`) or weighted score fusion (Stage 7).
- **FR-018**: System MUST re-rank top candidates using a cross-encoder model (`BGE-Reranker v2 M3` or `Zerank 1 Small`) down to 5-10 final chunks (Stage 7).
- **FR-019**: System MUST enforce document diversity by capping maximum chunks per document (max 2-3 per doc) (Stage 7).
- **FR-020**: System MUST support parent chunk expansion and sentence window expansion post-retrieval (Stage 8).
- **FR-021**: System MUST apply `LongContextReorder` and context compression to minimize prompt token waste and lost-in-the-middle degradation (Stage 8).
- **FR-022**: System MUST attach page number, source document title, and bounding box coordinates for precise source citation (Stage 8).

### Stages 9 to 10: Generation, Agents, Evaluation & Security
- **FR-023**: System MUST support grounded generation with "I don't know" fallback using GPT-4o or local Llama-3.1-8B, implementing Compact, Refine, and Tree Summarize synthesis modes (Stage 9).
- **FR-024**: System MUST implement a LangGraph StateGraph agent coordinating document RAG, GraphRAG, and Web search tools (Stage 9).
- **FR-025**: [DESCOPED] Text-to-SQL tool and SQL AST execution are explicitly out of scope for the current system scope and deferred to future releases.
- **FR-026**: System MUST execute automated RAGAS evaluations computing Faithfulness, Answer Relevancy, Context Precision, and Context Recall against a versioned Golden Test Set (Stage 10).
- **FR-027**: System MUST enforce Quality Gate thresholds: Faithfulness >= 0.80, Answer Relevancy >= 0.75, Context Precision >= 0.75, Context Recall >= 0.70 (Stage 10).
- **FR-028**: System MUST trace all pipeline spans (prompts, completions, embeddings, tool arguments, latency, tokens, cost) via Langfuse conforming to OpenInference standards (Stage 10).
- **FR-029**: System MUST redact Personally Identifiable Information (PII) using Microsoft Presidio and generate structured audit logs (Stage 10).
- **FR-030**: System MUST provide Redis content-hash embedding caching and TTL LLM response caching (Stage 10).
- **FR-031**: System MUST maintain a 100-item multi-suite Golden Benchmark Dataset (`golden_set.jsonl`) encompassing 4 distinct evaluation suites: (1) Science Textbook Suite (40 items across 15 chapters), (2) Administrative FAQ Suite (30 items with exact and colloquial Evol-Instruct mutations across 9 categories), (3) Domain Routing & Disambiguation Suite (15 items), and (4) Adversarial & Refusal Suite (15 items) (Stage 10).
- **FR-032**: System MUST provide programmatic dataset loading and integrity validation via `load_dataset.py` supporting suite filtering and structure verification (Stage 10).
- **FR-033**: System MUST achieve a minimum Domain Routing Accuracy of >= 95% on the Domain Routing Suite without cross-corpus leakage (Stage 10).
- **FR-034**: System MUST achieve a Refusal Accuracy of >= 90% on the Adversarial & Refusal Suite, returning standardized deterministic refusal messages paired with structured metadata (`is_refusal: true`, `refusal_reason`) for Layer 1 Code Gate rejections and grounded fallback statements for Layer 2 synthesis, guaranteeing zero hallucination on out-of-scope or malicious queries (Stage 10).
- **FR-035**: System MUST implement a Tri-Layered "Swiss Cheese" Evaluation Engine cascading from Layer 1 Deterministic Code Gates (< 5ms, 0 tokens) to Layer 2 LLM-as-a-Judge with 5-part rubrics and Ragas adapters, through Layer 3 Human Meta-Evaluation (Stage 10).
- **FR-036**: System MUST record and output high-resolution execution duration (`latency_seconds`) whenever an LLM Judge or evaluation metric is invoked (Stage 10).
- **FR-037**: System MUST provide a Neo4j Knowledge Graph Store indexing Document-Chunk-Entity hierarchies and supporting Cypher-based multi-hop candidate discovery for synthetic testset generation (Stage 10).
- **FR-038**: System MUST instrument end-to-end pipeline spans (`rag_query`, `retrieval`, `synthesis`) with latency and token attribution via Langfuse when enabled (Stage 10).
- **FR-039**: System MUST support TypeSafe Jev (System One AI) as the primary, high-throughput decision engine across pipeline stages (Stage 1 & 2 Routing & Guardrails, Stage 6 Candidate Reranking, Layer 2.5 Automated Evaluation), minimizing external API calls by consolidating multiple decisions into single parallel speculative fan-out requests:
  1. **Consolidated Jev Pipeline Roles**:
     - **Pre-Retrieval Routing & Safety Guardrail**: In a **single parallel request**, Jev determines both domain routing (`Choice`: science_textbook, administrative_faq, out_of_scope) and adversarial prompt injection detection (`Noul`: is_adversarial).
     - **Post-Retrieval Chunk Reranking (`JevReranker`)**: In a **single parallel request**, Jev evaluates semantic relevance across all candidate chunks (`chunk_0`, `chunk_1`, ..., `chunk_k`), sorting by calibrated relevance and filtering sub-threshold chunks.
     - **Evaluation (Judge)**: In a **single parallel speculative fan-out request** per evaluated item, Jev simultaneously computes:

     - **Faithfulness (`Noul`)**: Calibrated probability (0.0 to 1.0) checking if all claims are strictly substantiated by retrieved context (Pass: `noul >= 0.80`).
     - **Answer Relevancy (`Score`)**: 3-level calibrated rubric distinguishing:
       - `Level 0 (Evasive/Off-Topic)`: Response completely avoids the question, discusses an irrelevant topic, wrongly refuses a legitimate question, or fails to address any core concept in the prompt.
       - `Level 1 (Partial/Incomplete)`: Response directly addresses the subject and provides accurate information, but omits key constraints, conditions, or secondary components of multi-part questions, or dilutes the answer with excessive tangential text.
       - `Level 2 (Complete & Direct)`: Response directly, accurately, and comprehensively satisfies all parts of the user inquiry without evasion, omission, or fluff (Pass: `score >= 1.50`).
     - **Context Coverage (`Noul`)**: Calibrated probability (0.0 to 1.0) whether retrieved context contains sufficient evidence to answer the query according to ground truth (Pass: `noul >= 0.70`).
     - **Adversarial Refusal (`Noul`)**: Calibrated probability verifying whether ungrounded, deceptive, or out-of-domain queries were safely refused (Pass: `noul >= 0.70`).
     - **Speculative Fan-out**: Independent evaluation questions over the same state MUST execute in a single parallel request.
  2. **Generative Boundary**: Free-form text generation—specifically Stage 5 Query Transformation (HyDE, query rewriting, sub-question decomposition) and Stage 9 Answer Synthesis—MUST remain on the generative LLM (`gemini-cli` via Nine Router), as System One models return typed decisions rather than generated text.
  3. **Automatic Fallback Protocol**: If the Jev API is unreachable, times out (> 5s), is blocked by network sandbox, or returns low confidence (< 0.65), the system MUST seamlessly fall back to local `LLMJudge` / router via Nine Router without throwing unhandled exceptions.
- **FR-040**: The system MUST implement LangChain and LangGraph as the mandatory architectural and execution standard across ALL RAG workflows—including the baseline naive pipeline, optimized hybrid retrieval, and multi-tool agent orchestration:
  1. **StateGraph Standard**: Every pipeline workflow MUST execute as a compiled LangGraph `StateGraph` with strongly-typed `RAGGraphState` containing query, routing flags, retrieved/reranked candidates, synthesized responses, and evaluation verdicts.
  2. **Composable Nodes**: Pipeline stages (Jev routing/guardrail, Qdrant retrieval, Jev candidate reranking, LLM synthesis, Jev evaluation) MUST be implemented as modular, reusable LangChain/LangGraph nodes and `Runnable` components.
  3. **Universal Compatibility**: The baseline RAG pipeline MUST be structured as a LangGraph `StateGraph`, establishing consistent state progression, conditional edges, and checkpointing from day one.
- **FR-041**: The system MUST provide comprehensive, non-optional Langfuse integration across all workflows, evaluations, and benchmarks:
  1. **Mandatory Tracing by Default**: Langfuse tracing MUST be active for EVERY workflow execution and evaluation run (not optional or off-by-default). If Langfuse environment keys are present, runs MUST automatically instrument end-to-end traces.
  2. **Granular Span Telemetry (Cost & Latency)**: Every node/span in the LangGraph execution MUST record high-resolution execution duration (`latency_seconds` / `latency_ms`) and token attribution (prompt tokens, completion tokens, total cost in USD) via native Langfuse `CallbackHandler` or `@observe` instrumentation.
  3. **Structured Pipeline Tagging**: Every trace MUST be tagged dynamically to enable cross-pipeline comparison in the Langfuse dashboard:
     - Baseline executions MUST carry the `baseline` tag.
     - Secondary/improved pipeline runs MUST carry appropriate enhancement tags (e.g. `improved`, `hybrid-bm25`, `jev-reranker`, `graphrag`).
     - Evaluation runs MUST carry split tags (`dev`, `test`, `smoke`) and suite tags (`science`, `faq`, `routing`, `refusal`).
  4. **Benchmark Migration to Langfuse Datasets**:
     - The Golden Benchmark dataset (`golden_set.jsonl`) MUST be programmatically synchronized into a persistent Langfuse Dataset (`unified-rag-golden-benchmark`).
     - Evaluation benchmarks MUST support running directly against Langfuse Datasets as Experiments/Runs, saving verdicts (`overall_verdict`, `faithfulness`, `answer_relevancy`, `context_coverage`) directly as trace scores attached to dataset items.

---

## Success Criteria *(mandatory)*

- **SC-001**: Faithfulness score on the Golden Test Set meets or exceeds **0.80** (zero hallucination tolerance).
- **SC-002**: Answer Relevancy score meets or exceeds **0.75**.
- **SC-003**: Context Precision meets or exceeds **0.75**.
- **SC-004**: Context Recall meets or exceeds **0.70**.
- **SC-005**: 100% of exact duplicate documents are collapsed into a single canonical row in the manifest catalog.
- **SC-006**: [DESCOPED] Blocked SQL mutation commands criterion deferred with Text-to-SQL descope.
- **SC-007**: 100% of PII detected by Presidio is anonymized prior to indexing or LLM submission.
- **SC-008**: Repeated identical queries achieve a latency reduction of >= 80% via Redis caching.
- **SC-009**: Autocut outlier exclusion and cross-encoder re-ranking demonstrate >= 15% improvement in Context Precision over basic vector top-k.
- **SC-010**: Automated CI workflow (`rag-eval.yml`) blocks PR merges whenever any evaluation metric regresses below baseline or required thresholds.
- **SC-011**: Domain Routing Accuracy across heterogeneous corpora meets or exceeds **95%** on Suite 3 test cases.
- **SC-012**: Adversarial Refusal Accuracy meets or exceeds **90%** on Suite 4 unanswerable test cases with zero hallucinations.
- **SC-013**: Layer 1 Deterministic Code Gates achieve < 5ms execution latency and 0 LLM token cost for all refusal and domain routing items.
- **SC-014**: Every LLM judge verdict outputs execution `latency_seconds` with automated console and JSON report persistence.
- **SC-015**: Neo4j Knowledge Graph establishes uniqueness constraints, links chunks by shared entity relations, and returns multi-hop traversal candidates across chapters.
- **SC-016**: Jev System One evaluation completes 75-item dev evaluation with mean per-item judgment latency < 500ms, achieving >= 90% verdict agreement with ground-truth labels while generating 0 completion tokens.
- **SC-017**: 100% of pipeline executions and evaluation queries produce valid Langfuse traces capturing per-node latency, token/cost attribution, and structured tags (`baseline`, `improved`, split/suite).
- **SC-018**: 100% of Golden Benchmark test items are synchronized to a persistent Langfuse Dataset, supporting experiment run tracking with automated score logging directly visible in Langfuse UI.



# Feature Specification: Enterprise Science RAG Pipeline

**Feature Branch**: `003-enterprise-science-pipeline`

**Created**: 2026-10-10

**Status**: Draft

**Input**: User description: "User Story 1: Hybrid Retrieval (BM25 + Qdrant Dense) with Reciprocal Rank Fusion on Science Book. User Story 2: Cross-Encoder Re-ranking (bge-reranker-v2-m3) + Dynamic Parent Chunk Expansion. User Story 3: Confidence-Gated Query Rewriting & Sub-query Decomposition. User Story 4: Latency-Budgeted AdvancedRAGPipeline (<2s p95) with Langfuse trace-level quality metrics."

---

## Clarifications

### Session 2026-10-10
- Q: How should the BGE-Reranker v2 M3 cross-encoder be executed to satisfy the 200–350ms latency budget? → A: Local in-process HuggingFace / PyTorch model loaded with CPU/MPS device selection, strictly bounded to a 350ms timeout after which it gracefully falls back to raw RRF ranking.
- Q: How should the BM25 lexical index for the 15 science chapters be stored and served? → A: In-memory rank_bm25 engine using Hazm tokenization and stemming, persisted as a local disk artifact (`data_index/science_bm25.pkl`) and loaded into RAM on startup for sub-5ms retrieval.
- Q: How should the confidence gate determine when to trigger the query rewriter? → A: Use a TypeSafe Jev / small classifier call on each query to score rewrite necessity confidence; if confidence indicates ambiguity or history dependency, trigger the rewriter, otherwise bypass to preserve the latency budget.
- Q: What maximum context token ceiling should be enforced during parent chunk expansion? → A: Dynamic deduplicated parent expansion capped at a maximum of 3 unique parent sections (~2,500 tokens max), ensuring generation TTFT remains under 1.0 second.
- Q: How should trace-level quality evaluators (Chunk Relevance, Context Adherence) be executed in Langfuse? → A: Dispatched as non-blocking background tasks (via `asyncio.create_task`) immediately post-response delivery, ensuring 100% trace coverage without consuming any user latency budget.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Hybrid Retrieval (BM25 + Qdrant Dense) with Reciprocal Rank Fusion (Priority: P1)

As an educational user or science student querying the 7th-grade science textbook,  
I want the retrieval system to search both exact scientific terminology/formulas (via BM25 lexical search) and conceptual semantic meaning (via dense vector search) simultaneously, and combine the best results using Reciprocal Rank Fusion,  
So that queries containing specific Persian terms, laboratory experiment names, or chapter numbers do not get missed even when semantic embeddings alone fail to prioritize them.

**Why this priority**: Solves Barnett Failure Point 2 (FP2: Missed top-ranked documents). Dense retrieval alone suffers from semantic drift on domain-specific scientific jargon and exact formula terms. Hybrid search is the foundational prerequisite for high retrieval recall.

**Independent Test**: Can be fully tested by querying the 180+ item `science_textbook.csv` suite containing exact chapter terms and experimental procedures, verifying that combined hybrid search with RRF achieves higher Hit@1 and Hit@5 than dense-only retrieval without requiring downstream generation.

**Acceptance Scenarios**:

1. **Given** a query with an exact laboratory term (e.g., *سانتریفیوژ*, *چگالی*, *پی‌اچ*), **When** hybrid retrieval executes, **Then** BM25 captures the exact term matches, dense retrieval captures semantic intent, and Reciprocal Rank Fusion ranks the relevant textbook passage in the top 5 results.
2. **Given** concurrent retrieval execution, **When** hybrid retrieval is triggered, **Then** BM25 and dense search execute in parallel without sequential blocking latency.
3. **Given** disparate score scales between dense cosine similarity and BM25 relevance, **When** ranking fusion runs, **Then** candidates are merged purely by reciprocal rank order (`score = sum(1 / (60 + rank))`) rather than raw unnormalized scores.

---

### User Story 2 - Cross-Encoder Re-ranking & Dynamic Parent Chunk Expansion (Priority: P2)

As a researcher studying scientific explanations,  
I want retrieved candidates to be re-scored by a high-precision cross-encoder and expanded from fine-grained child chunks to complete parent sections,  
So that the language model receives cohesive pedagogical context without broken sentences or dropped definitions (mitigating Barnett FP3 and FP4).

**Why this priority**: Solves Barnett Failure Point 3 (FP3: Chunks retrieved but lost in context window) and Failure Point 4 (FP4: Answer buried in context noise). Fine-grained child chunks maximize search precision, but parent window expansion provides the full conceptual context required for accurate reasoning.

**Independent Test**: Submit complex multi-sentence questions where answer evidence spans an entire subsection; verify that top 50 candidates are re-ranked to the top 5 by `bge-reranker-v2-m3` in under 300ms and that child chunks automatically expand to their parent sections with clean citation links.

**Acceptance Scenarios**:

1. **Given** 50 candidates retrieved across dense and BM25 lists, **When** cross-encoder re-ranking runs, **Then** the most semantically relevant 5 chunks are selected with significant margin over background distractors.
2. **Given** a high-scoring child chunk, **When** context assembly runs, **Then** the child chunk dynamically resolves to its parent section (aggregating adjacent child chunks) so that formulas and procedures are complete.
3. **Given** multiple retrieved child chunks belonging to the same parent section, **When** context assembly completes, **Then** parent chunks are deduplicated so the prompt does not repeat redundant sections.

---

### User Story 3 - Confidence-Gated Query Rewriting & Sub-query Decomposition (Priority: P3)

As a user engaged in a multi-turn conversation or asking complex multi-concept questions,  
I want the system to resolve conversational pronouns and decompose multi-step scientific questions into sub-queries, but only when necessary,  
So that complex questions get accurate answers while simple standalone queries are not slowed down by unnecessary language model rewrites.

**Why this priority**: Protects latency budgets while unlocking multi-hop scientific reasoning. As established in the Splunk enterprise architecture, rewriting every query consumes unnecessary latency and cost; confidence-gating rewrites captures accuracy gains while keeping latency low.

**Independent Test**: Execute standalone queries, pronoun-heavy follow-up questions (*"Where does this reaction happen?"*), and multi-hop queries (*"Compare the boiling point of pure water with salt water and explain how atmospheric pressure affects it"*); verify that standalone queries bypass the rewriter in <10ms, while multi-hop questions decompose into targeted sub-queries.

**Acceptance Scenarios**:

1. **Given** a clear standalone scientific question, **When** query analysis runs, **Then** the confidence gate approves the original query without invoking the query rewriter, saving latency.
2. **Given** a conversational follow-up query with ambiguous pronouns, **When** query analysis runs, **Then** the history-aware rewriter reconstructs a self-contained query with full entity context.
3. **Given** a multi-concept scientific inquiry spanning multiple curriculum domains, **When** query analysis runs, **Then** the inquiry is decomposed into 2-3 focused sub-queries whose retrieved candidates are merged.

---

### User Story 4 - Latency-Budgeted Pipeline with Langfuse Trace Quality Metrics (Priority: P4)

As an AI engineer and platform operator,  
I want the entire RAG pipeline to execute within strict latency budgets (<2.5s end-to-end, <2.0s p95) and record trace-level quality metrics (Chunk Relevance, Context Adherence, Completeness) in Langfuse asynchronously,  
So that live operational bottlenecks and hallucination risks are automatically detected without stealing latency from the user.

**Why this priority**: Solves the enterprise observability and production reliability requirement. Separating retrieval quality evaluation from generation quality evaluation allows developers to instantly pinpoint whether a production error originated in search or synthesis.

**Independent Test**: Execute 50 continuous queries through `AdvancedRAGPipeline`, measuring end-to-end wall-clock latency to confirm p95 < 2.0s, and inspect the Langfuse dashboard to verify that all nested spans and trace-level quality evaluators are properly recorded off the critical path.

**Acceptance Scenarios**:

1. **Given** a full end-to-end query execution, **When** the response is delivered, **Then** the total turnaround time meets the p95 latency target of under 2.0 seconds.
2. **Given** streaming generation, **When** the first token is emitted, **Then** Time-To-First-Token (TTFT) is under 1.0 second.
3. **Given** pipeline tracing in Langfuse, **When** a query completes, **Then** distinct nested spans for `query_understanding`, `retrieval`, `reranking`, and `generation` are recorded with token and latency attribution.
4. **Given** completed traces, **When** background evaluation runs, **Then** trace-level scores for Chunk Relevance (binary 0/1) and Context Adherence are recorded asynchronously without delaying the user response.

---

### Edge Cases

- **No Relevant Documents Found**: When both dense and BM25 retrievers return candidates below confidence thresholds, the system must emit a standardized grounded refusal (*"Based on the provided science curriculum, this information is not available"*) with `is_refusal: true` rather than fabricating facts.
- **Extreme Query Length**: Queries exceeding 512 tokens must be truncated or summarized by the pre-retrieval gate before vector embedding.
- **Conflicting Scientific Claims in Context**: When parent expansion incorporates competing or contradictory information across different curriculum revisions, the synthesizer must explicitly note the distinction rather than silently choosing one.
- **Provider Outage / Downstream Timeout**: If the cross-encoder or reranker times out (exceeding 400ms), the pipeline must gracefully fall back to the raw RRF candidate ranking without crashing the user request.
- **Network / Tracing Failure**: If the Langfuse server is unreachable, tracing operations must silently fail safe without throwing uncaught exceptions to the caller.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST index the 15 chapters of the 7th-grade science textbook into an in-memory `rank_bm25` lexical inverted index with `hazm` Persian text normalization and stemming, persisted as a disk artifact (`data_index/science_bm25.pkl`) with sub-5ms retrieval latency.
- **FR-002**: System MUST execute BM25 lexical search and Qdrant dense vector search in parallel (`asyncio.gather`) for incoming user queries.
- **FR-003**: System MUST combine dense and lexical retrieval result lists using Reciprocal Rank Fusion (RRF: `score = sum(1 / (60 + rank))`) over top candidates.
- **FR-004**: System MUST re-rank top candidate chunks using local in-process `BAAI/bge-reranker-v2-m3` (with CPU/MPS device detection) to select the top 5 most relevant chunks, gracefully falling back to raw RRF ranking if inference exceeds 350ms.
- **FR-005**: System MUST dynamically expand retrieved child chunks to their enclosing parent sections, deduplicating shared parents and capping total assembled context to a maximum of 3 unique parent sections (~2,500 tokens).
- **FR-006**: System MUST arrange selected context chunks using `LongContextReorder` (placing highest-ranked evidence at the beginning and end of the prompt window) prior to synthesis.
- **FR-007**: System MUST provide a confidence-gated query rewriting mechanism using a TypeSafe Jev / small classifier confidence check to route only ambiguous or history-dependent queries to the rewriter while bypassing standalone queries.
- **FR-008**: System MUST support sub-query decomposition for complex multi-part scientific queries.
- **FR-009**: System MUST enforce strict Markdown link page citations (`[Title](...#page=N)`) formatted according to validated curriculum page numbers.
- **FR-010**: System MUST assemble all stages into an orchestrating class `AdvancedRAGPipeline` located in `rag-core/src/rag_core/pipeline.py`.
- **FR-011**: System MUST instrument all pipeline stages with OpenTelemetry-compliant Langfuse traces and nested spans (`query_rewrite`, `hybrid_retrieval`, `cross_encoder_rerank`, `synthesis`).
- **FR-012**: System MUST record trace-level quality metrics in Langfuse (Chunk Relevance, Context Adherence, Completeness) dispatched as non-blocking background tasks (`asyncio.create_task`) off the critical user response path.
- **FR-013**: System MUST provide an offline evaluation command evaluating `AdvancedRAGPipeline` against `rag-core/src/rag_core/evaluation/datasets/suites/science_textbook.csv` (180+ items).
- **FR-014**: System MUST enforce strict timeout budgets per stage (guardrail <= 100ms, rewrite <= 300ms, retrieval <= 200ms, rerank <= 350ms, generation TTFT <= 1000ms).

### Key Entities

- **Document Chunk**: Atomic text unit (~512 tokens) with heading breadcrumbs, page attribution, and parent section reference ID.
- **Parent Section**: Logical chapter subsection encapsulating multiple constituent child chunks, preserving tables and experiment protocols intact.
- **Hybrid Search Result**: Unified candidate object containing chunk metadata, dense score, BM25 rank, combined RRF score, and cross-encoder score.
- **Pipeline Context**: Execution state containing original query, rewritten query, retrieved child chunks, expanded parent chunks, latency profiles, and citation references.
- **Trace Evaluation Record**: Observability artifact recording binary chunk relevance, context adherence, total token cost, and wall-clock latency per request.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: **Retrieval Recall**: Hybrid retrieval (Dense + BM25 + RRF + Cross-Encoder) achieves `Hit@5 >= 0.90` and `MRR >= 0.80` on the 180+ item `science_textbook.csv` benchmark suite.
- **SC-002**: **End-to-End Latency**: 95th-percentile (p95) end-to-end response latency is strictly under 2.0 seconds for standalone queries.
- **SC-003**: **Streaming TTFT**: Time-To-First-Token (TTFT) for streaming responses is under 1.0 second.
- **SC-004**: **Citation Fidelity**: 100% of factual assertions in generated answers include verified Markdown page citations (`[Title](...#page=N)`), achieving `Citation Precision >= 0.85`.
- **SC-005**: **Zero Hallucination Grounding**: Context Adherence / Faithfulness scores achieve `>= 0.85` on the science curriculum evaluation set.
- **SC-006**: **Parsimonious Rewriting**: Greater than 50% of direct standalone queries bypass the query rewriter stage without quality loss, maintaining zero unnecessary LLM token expenditure on unambiguous prompts.

---

## Assumptions

- The 15 chapters of the 7th-grade science textbook in `data/vlm_parsed/` provide high-fidelity markdown and layout attribution for all curriculum sections.
- The `bge-reranker-v2-m3` model will run locally (via CPU/MPS with int8/FP16 quantization) with execution latency bounded to <350ms for 50 candidates.
- The Qdrant vector database and Redis cache are accessible via local environment configurations.
- Langfuse is configured via environment variables and supports asynchronous, non-blocking trace and score submission.

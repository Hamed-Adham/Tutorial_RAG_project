# Feature Specification: Hierarchical SubQ Multi-Source Router & Compound Corpus Dispatcher

**Feature Branch**: `008-multisource-router-faq`
**Feature Directory**: `specs/008-multisource-router-faq`
**Created**: 2026-10-15
**Status**: Draft (In Clarification)
**Input**: "we put an subQ DC model as source router that analyze if this query relevant 1 by 1 to any of our sources . like the one we do here in the Decision model 1 when we have slected one source we straight go for Decision model 1 but with customized per source details but when we have choosen multiple sources , we need higher source router and some time it could happen that our source router routers our query to multiple sources not one when query require multi source to be ansered or it compound and each part need different source ."

## Clarifications

### Session 2026-10-15
- Q: How should the Level-0 SubQ Decision Source Router evaluate relevance across multiple corpora when multi-source mode is active? → A: Single multi-label classification prompt (a single model call evaluating all available corpora at once and returning the list of active corpora).
- Q: When a compound cross-corpus query is answered (combining Science and Administrative FAQ), how should the synthesized response be structured? → A: Partitioned response with domain headers (Option 1). Compound queries MUST first be transformed/decomposed into atomic sub-queries mapped to their respective corpora, and for complex multi-step inquiries, synthesis must address the steps systematically under the respective domain sections.
- Q: How should the lexical search (BM25) index for administrative_faq be structured and managed in the advanced pipeline? → A: Separate BM25 index per corpus (maintain data_index/faq_bm25.pkl alongside science_bm25.pkl, with a MultiCorpusBM25Retriever dispatching queries to the target index).
- Q: When a compound query is decomposed into atomic sub-queries for different corpora, how should each sub-query be processed before retrieval? → A: Keep the advanced pipeline architecture uniform and elegant: invoke Decision Model 1 once per active source in parallel (with customized domain state/prompting per source), execute parallel hybrid retrieval and transformations for each source, and generate a single unified response once at the end using the composite multi-domain synthesizer.

---

## 1. Executive Summary & Architectural Vision

The Unified RAG platform currently operates with an advanced enterprise pipeline that is tightly coupled to the 7th-grade science curriculum (`science_textbook`), while the 134-chunk administrative FAQ collection (`administrative_faq`) is isolated in the naive baseline. 

In production multi-corpus environments, user queries are not always confined to a single source:
1. **Explicit Single-Source Selection**: The user manually selects a single corpus (e.g., only Science or only FAQ). The query proceeds directly to that domain's specialized Decision Model 1.
2. **Autonomous Multi-Source / Cross-Corpus Routing**: The user selects multiple sources (or "All Sources"). A **Level-0 SubQ Decision Source Router** evaluates the query against all registered candidate corpora simultaneously using a single multi-label classification model call (<25ms).
3. **Parallel Per-Source DM1 & Hybrid Retrieval**: When multiple sources are active, the system executes **Decision Model 1 in parallel once per active source** (with customized domain states and domain-specific query expansion). Parallel hybrid retrievals search their respective collections (`science_textbook` and `administrative_faq`).
4. **Single Unified Response Generation**: All retrieved chunks are merged and reranked, and a **single response generation** call synthesizes the final answer with partitioned domain headers (`### ۱. مفاهیم علمی (کتاب علوم)` and `### ۲. مقررات و تسهیلات (صندوق نوآوری)`), answering multi-step complex inquiries systematically.

```
                              User Query
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       Single Source Selected          Multi-Source / All Selected
                  │                               │
                  │                               ▼
                  │                  Level-0 SubQ Source Router
                  │                 (Single Multi-Label Decision)
                  │                               │
                  │             ┌─────────────────┴─────────────────┐
                  │             ▼                                   ▼
                  │       Single Target                    Multiple Target Corpora
                  │       (Science or FAQ)                (Compound Cross-Corpus)
                  │             │                                   │
                  └────────► DM1 (Source)                  Parallel Per-Source DM1s
                                │                       (DM1-Science & DM1-FAQ in parallel)
                                │                                   │
                                ▼                                   ▼
                         Hybrid Retrieval                  Parallel Hybrid Retrievals
                         (Target Corpus)                   (Science Index & FAQ Index)
                                │                                   │
                                └─────────────────┬─────────────────┘
                                                  ▼
                                      Unified Cross-Encoder Rerank
                                                  ▼
                                  Single Final Response Generation
                                (Domain Partitioned: Science & FAQ)
```

---

## 2. User Scenarios & Testing (Mandatory)

### User Story 1 - Level-0 SubQ Source Routing for Multi-Source Queries (Priority: P1)

**User Journey**:
When a user submits a query with multiple corpora enabled (or in auto-routing mode), the system passes the query through a fast Level-0 SubQ Decision Source Router. The model evaluates relevance against each registered source (`science_textbook`, `administrative_faq`) using typed judgments and calibrated probabilities. If only one source is relevant, it smoothly delegates to that domain's Decision Model 1.

**Why this priority**:
Enables the advanced pipeline to ingest and serve administrative FAQ queries with zero friction, eliminating hardcoded science-only rejections.

**Independent Test**:
Submit an administrative FAQ query (`شرایط دریافت ضمانت‌نامه شرکت‌های دانش‌بنیان چیست؟`) with multi-source mode enabled. The Level-0 router must assign `administrative_faq` with probability >= 0.85 and route to the FAQ pipeline without manual source toggling.

**Acceptance Scenarios**:
1. **Given** multi-source mode is active, **When** user queries a textbook science topic, **Then** Level-0 router activates `science_textbook` and bypasses `administrative_faq`.
2. **Given** multi-source mode is active, **When** user queries an INIF loan/grant topic, **Then** Level-0 router activates `administrative_faq` and bypasses `science_textbook`.
3. **Given** multi-source mode is active, **When** user submits an out-of-scope or adversarial query, **Then** Level-0 router flags all sources as irrelevant and triggers early refusal in <25ms.

---

### User Story 2 - Compound Cross-Corpus Query & Multi-Source Dispatch (Priority: P2)

**User Journey**:
When a compound inquiry touches multiple distinct knowledge domains (e.g. *"دستگاه گوارش چیست و چگونه می‌توان برای اختراع آن از صندوق نوآوری تسهیلات گرفت؟"*), the Level-0 Source Router recognizes that multiple sources are required. The system invokes Decision Model 1 in parallel once per active source (with customized state and domain prompts for Science and FAQ), executes parallel hybrid retrievals and query transformations across both collections, and synthesizes a single final response with partitioned domain headers.

**Why this priority**:
Fulfills the core architectural goal of enterprise RAG: handling compound inquiries that span heterogeneous corporate and technical knowledge bases while keeping pipeline structure uniform.

**Independent Test**:
Submit a query containing both a science concept and an administrative FAQ requirement. Verify that Level-0 router selects both sources, parallel DM1 instances run for Science and FAQ, parallel retrievals query both indices, and the final response contains partitioned domain sections with corresponding citations.

**Acceptance Scenarios**:
1. **Given** a query requiring both Science and FAQ, **When** Level-0 router evaluates sources, **Then** both `science_textbook` and `administrative_faq` are flagged as active destinations.
2. **Given** multiple active destinations, **When** pipeline executes, **Then** DM1-Science and DM1-FAQ run concurrently in parallel with domain-customized states.
3. **Given** parallel multi-corpus retrieval, **When** synthesis runs once at the end, **Then** the answer partitions into domain headings (`### ۱. مفاهیم علمی (کتاب علوم)` and `### ۲. مقررات و تسهیلات (صندوق نوآوری)`), and citations display both textbook chapters and FAQ categories.

---

### User Story 3 - Full Administrative FAQ Advanced Pipeline Integration (Priority: P3)

**User Journey**:
When queries are routed to `administrative_faq` in the advanced pipeline, the system utilizes dedicated FAQ lexical search (BM25 on `faq_qa_pairs`), dense Qdrant search with category filtering, pass-through expansion (bypassing chapter VLM manifest), domain-appropriate synthesis prompting (INIF consultant persona), and category-aware citations.

**Why this priority**:
Brings the administrative FAQ collection up to full parity with the science textbook in the advanced enterprise pipeline.

**Independent Test**:
Execute an FAQ query with category pre-filter (`ضمانت‌نامه`). Verify that BM25 retrieves from `faq_bm25.pkl`, dense search filters by category in Qdrant, reranking scores candidates, and the response uses the INIF advisory persona.

**Acceptance Scenarios**:
1. **Given** an FAQ query, **When** BM25 executes, **Then** results are sourced from the dedicated FAQ lexical index.
2. **Given** category selection in the UI, **When** dense retrieval executes, **Then** Qdrant filters by `payload.category in selected_categories`.
3. **Given** FAQ chunk candidates, **When** expansion executes, **Then** chunks pass through directly without searching `manifest.json`.

---

### User Story 4 - Manual Corpus Override & UI Multi-Corpus Selection (Priority: P4)

**User Journey**:
In the Web UI and API, users can either check a specific source (single-corpus mode) or check multiple sources simultaneously (multi-corpus mode). When a single source is checked, the Level-0 router is bypassed and the system immediately invokes that domain's Decision Model 1. When multiple sources are checked, the Level-0 router is activated.

**Why this priority**:
Provides users and developers with explicit control while preserving autonomous multi-corpus intelligence.

**Independent Test**:
In the frontend header, select only "کتاب علوم" and verify Level-0 is bypassed. Select both "کتاب علوم" and "پرسش‌های متداول", verify multi-source dispatching is enabled.

---

## 3. Requirements (Mandatory)

### Functional Requirements

- **FR-001**: The system MUST implement a Level-0 Source Router using a single multi-label TypeSafe / SubQ Decision classification prompt that evaluates relevance against all registered corpora simultaneously in a single model call (<25ms) and returns the list of active corpora (e.g., `["science_textbook"]`, `["administrative_faq"]`, both, or empty for refusal).
- **FR-002**: When a single source is explicitly requested by the client, the Level-0 router MUST be bypassed, directing the query immediately to that source's Decision Model 1.
- **FR-003**: When multiple sources are active, the Level-0 router MUST be capable of selecting: (a) single source, (b) multiple sources, or (c) zero sources (refusal).
- **FR-004**: When multiple sources are selected by the Level-0 router, the system MUST invoke Decision Model 1 in parallel once per active source (with customized state and domain prompts for Science and FAQ), execute parallel hybrid retrievals and query transformations for each active source, and merge candidates for unified reranking and single generation at the end.
- **FR-005**: The system MUST provide a dedicated in-memory lexical index (`data_index/faq_bm25.pkl`) for `administrative_faq` with Persian normalization, managed by a `MultiCorpusBM25Retriever` that dynamically dispatches lexical queries to the corresponding domain index based on target corpus.
- **FR-006**: The hybrid retriever MUST support source-level pre-filtering (`filter_source_type`) and category-level pre-filtering (`filter_categories`) across Qdrant and BM25.
- **FR-007**: `ParentSectionExpander` MUST implement a pass-through mode for non-textbook chunks (such as FAQ QA pairs) without erroring on missing chapter manifests.
- **FR-008**: The synthesis layer MUST dynamically select the system prompt based on active source domains (`SYSTEM_PROMPT_PERSIAN_GROUNDED` for science, `SYSTEM_PROMPT_ADMIN_FAQ_GROUNDED` for FAQ, and `SYSTEM_PROMPT_COMPOSITE_GROUNDED` for cross-corpus). For cross-corpus answers, synthesis MUST partition the output using explicit Markdown domain headings (e.g., `### ۱. مفاهیم علمی (کتاب علوم)` and `### ۲. مقررات و تسهیلات (صندوق نوآوری)`) and answer complex multi-step inquiries systematically step-by-step under their corresponding domain sections.
- **FR-009**: Citations MUST support FAQ metadata (`[پرسش‌های متداول – دسته: {category} – ردیف: {row}]`).

---

## 4. Edge Cases & Boundary Conditions

1. **Vocabulary Collisions**: A query uses financial or administrative terminology that appears inside the science book (e.g., "هزینه انرژی و بازیافت" or "ثبت اختراع سلول خورشیدی"). The Level-0 router must evaluate semantic intent rather than simple keyword matches.
2. **Partial Multi-Source Availability**: One of the target corpora returns zero matching candidates while the other returns high-confidence candidates. Synthesis must gracefully answer using the available corpus while noting the lack of data for the second part under its section.
3. **Latency Budget**: Adding Level-0 source evaluation must not exceed 30ms to stay within the overall 180ms gate latency budget. Parallel per-source DM1 calls must run concurrently via `asyncio.gather`.
4. **Single Context Turn Injections**: When a user selects a specific historical turn from a different corpus than the current query, Decision Model 2 must evaluate relevance across corpora.

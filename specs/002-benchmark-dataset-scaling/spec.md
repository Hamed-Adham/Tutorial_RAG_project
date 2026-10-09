# Feature Specification: Benchmark Dataset Scaling & Comprehensive Quality Evaluation

**Feature Branch**: `002-benchmark-dataset-scaling`

**Created**: 2026-10-08

**Status**: Draft

**Input**: User description: "enhance our benchmark dataset we have to be bigger in sample size and better in quality . use md files that we extracted from the pdfs to generate query and ground truth from . for high quality query generation use the gemini-cli model . for faq synthesize questions with similar purpose but different phrasing and wording . scale to 400+ items with decoupled retrieval and generation evaluation ."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-Perspective FAQ Retrieval & Grounding Evaluation (Priority: P1)

As an evaluation engineer, I want the FAQ benchmark to test realistic user inquiries with diverse conversational phrasings, cross-category terminology clashes, and multi-service combinations, so that I can rigorously measure retrieval and response accuracy across the organization's entire knowledge catalog rather than relying on exact question matches.

**Why this priority**: The administrative and financial knowledge base represents core business policies. If queries only test literal FAQ titles, retrieval benchmarks will be artificially optimistic and will fail to detect real customer failures.

**Independent Test**: Can be fully tested by evaluating the retrieval engine and generator against a newly synthesized suite of 200+ colloquial, keyword-clashing, and multi-row administrative queries mapped to unique catalog reference entries, confirming both correct top-k retrieval and grounded response generation.

**Acceptance Scenarios**:

1. **Given** a Persian user query with conversational slang and non-standard phrasing regarding grant eligibility, **When** the evaluation harness executes retrieval and synthesis, **Then** the correct reference FAQ entry is identified in the top ranks, and the generated response matches the factual ground truth.
2. **Given** an ambiguous query sharing terms across distinct service categories (e.g., collateral requirements versus leasing guarantees), **When** retrieval executes, **Then** the system distinguishes the correct administrative department and retrieves the corresponding policy chunk without cross-department leakage.
3. **Given** a combined query inquiring about both financing limits and required collateral, **When** retrieval executes, **Then** all relevant reference chunks are retrieved, and the synthesizer produces a coherent composite answer without dropping conditions.
4. **Given** an unanswerable or out-of-scope administrative query (e.g., personal mortgage loan), **When** the system processes the request, **Then** it correctly triggers an unambiguous refusal instead of hallucinating terms.

---

### User Story 2 - Full-Curriculum Multimodal Science Benchmark Expansion (Priority: P2)

As a curriculum domain specialist and AI tester, I want the science evaluation benchmark to cover all 15 textbook chapters across all 412 curriculum sections, including scientific diagrams and pedagogical experiments, so that the RAG pipeline is thoroughly validated across the complete educational syllabus.

**Why this priority**: The current benchmark only covers Chapters 14 and 15, leaving ~85% of the textbook unmeasured. A comprehensive benchmark is required to verify system generalization across heterogeneous scientific disciplines (physics, chemistry, biology, geology).

**Independent Test**: Can be fully tested by running automated evaluation across 180+ curriculum items spanning Chapters 1 through 15, verifying that each chapter has dedicated factoid, multi-hop reasoning, diagram extraction, and experimental reasoning items.

**Acceptance Scenarios**:

1. **Given** a scientific question regarding a laboratory procedure described in a pedagogical experiment block, **When** the system retrieves context, **Then** it successfully retrieves the experiment section and answers the question with step-by-step procedural fidelity.
2. **Given** a query asking for structural details depicted in an anatomical or physical diagram, **When** the system searches the knowledge base, **Then** it retrieves the visual description and spatial attributes captured during document ingestion.
3. **Given** a multi-hop question requiring knowledge synthesis across two distinct curriculum chapters (e.g., cellular respiration linked to matter conservation), **When** retrieval executes, **Then** relevant chunks from both source chapters are retrieved in the top results.

---

### User Story 3 - Decoupled Retrieval vs. Generation Evaluation (Priority: P3)

As a RAG pipeline developer, I want evaluation runs to report retrieval metrics independently from generation metrics, so that I can immediately determine whether an error stems from search retrieval failure (missing chunks) or LLM generation failure (hallucination or claim omission).

**Why this priority**: When evaluation only reports an end-to-end pass/fail verdict, debugging requires manual trace inspection. Decoupled metrics pinpoint whether the retriever or the synthesizer requires optimization.

**Independent Test**: Can be tested by executing the evaluation suite and inspecting the summary report, which must output separate quantitative scoreboards for retrieval quality and response quality.

**Acceptance Scenarios**:

1. **Given** a benchmark run over 200 items, **When** the evaluation finishes, **Then** the report presents separate retrieval quality metrics (top-k hit rate, mean reciprocal rank, context recall) and generation quality metrics (groundedness/faithfulness, answer relevancy).
2. **Given** an item where the retriever successfully fetches the gold chunk but the generator produces an ungrounded claim, **When** scores are recorded, **Then** retrieval is scored as successful (hit=1) while generation is flagged as unfaithful (score=0).

---

### User Story 4 - Source Citation & Provenance Verification (Priority: P4)

As a compliance auditor, I want every benchmark response to have its cited source locations (document name, page number, and section) automatically audited against the reference ground truth, so that users never receive inaccurate citations or phantom references.

**Why this priority**: Citations build user trust. Hallucinated citations ("Ghost Citations") severely damage system credibility even if the accompanying text is superficially plausible.

**Independent Test**: Can be tested by comparing generated citation metadata against reference chunk identifiers and calculating attribution precision and recall.

**Acceptance Scenarios**:

1. **Given** a generated response containing source citations, **When** the attribution validator inspects the response, **Then** every citation is verified against the reference document and page specified in the benchmark metadata.
2. **Given** a response that cites a page containing irrelevant or contradictory information, **When** audited, **Then** the citation score is penalized and flagged as misattribution.

---

### User Story 5 - Centralized Golden Dataset Synchronization & Observability (Priority: P5)

As an operations engineer, I want the expanded benchmark to be cataloged in a standardized 3-column format and continuously synchronized with cloud observability datasets, so that regression test runs and historical score trends are tracked over time.

**Why this priority**: Centralized versioning ensures all developers and continuous integration pipelines benchmark against the exact same test items with reproducible split boundaries.

**Independent Test**: Can be tested by verifying that all benchmark subsets parse strictly into uniform schemas and synchronize idempotently to the remote observability platform under isolated dev and test namespaces.

**Acceptance Scenarios**:

1. **Given** the master benchmark file containing 400+ items, **When** partitioned into evaluation subsets, **Then** all files conform strictly to the standard schema with zero syntax or missing-key errors.
2. **Given** an updated benchmark release, **When** synchronization executes, **Then** all items are updated idempotently in the central dataset store without duplicate entries.

---

### Edge Cases

- **Colloquial Dialect Ambiguity**: A colloquial user query might use slang that could refer to two completely different loan products. The benchmark must specify acceptable multi-label expected routes.
- **Cross-Chapter Terminology Collisions**: Scientific terms (e.g., "energy", "force") appear in multiple chapters with different pedagogical focus. The benchmark must explicitly define the expected chapter grounding chunks.
- **Partial Context Retrieval**: What happens if a multi-hop query only retrieves 1 of the 2 required chunks? The decoupled evaluation must record partial context recall and verify whether the generator gracefully admits incomplete information.
- **Unanswerable Negative Boundaries**: Queries asking about real-world Iranian financial topics not governed by this specific fund (e.g., tax exemptions, standard bank mortgages). The benchmark must verify clean refusals with zero ungrounded policy extrapolation.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST catalog every reference FAQ entry in the administrative knowledge base with a permanent, unique reference identifier (e.g., `faq_000` to `faq_133`).
- **FR-002**: The system MUST generate multiple distinct user query variations for each FAQ entry, including colloquial Persian phrasing, keyword-mismatch traps, cross-policy combinations, and out-of-scope inquiries, scaling the FAQ benchmark suite to at least 200 curated items.
- **FR-003**: The system MUST expand the science textbook benchmark suite across all 15 curriculum chapters using extracted structured markdown representations, scaling the science suite to at least 180 curated items.
- **FR-004**: Each science benchmark item MUST be attributed to explicit source locations, including document identifier, page number, and section classification (main body, activity, or experiment).
- **FR-005**: The benchmark generation process MUST produce balanced query complexity types, including point factoids, multi-hop conceptual reasoning, diagram descriptions, and procedural experiment inquiries.
- **FR-006**: The evaluation execution engine MUST decouple intermediate evaluation stages, outputting independent retrieval metrics (Hit@1, Hit@3, Hit@5, MRR, Context Recall) and generation metrics (groundedness/faithfulness, answer relevancy, refusal accuracy). Out-of-scope and adversarial queries MUST be excluded from retrieval hit metrics and evaluated exclusively on refusal accuracy.
- **FR-007**: The evaluation engine MUST verify citation fidelity, checking whether generated citations adhere strictly to Markdown link syntax (`[Title](...#page=N)`) matching gold source locations, penalizing fabricated or misattributed citations.
- **FR-008**: All generated benchmark items MUST adhere to a standardized 3-column representation (`input`, `expected_output`, `metadata`) with structured metadata capturing suite, split, query type, ground truth chunks, expected route, and expected behavior.
- **FR-009**: The benchmark dataset MUST enforce a deterministic 75/25 development and sealed test split to prevent overfitting during prompt and retrieval tuning. Development routines MUST target `--split dev`; `--split test` MUST be sealed and evaluated only during pre-release gates and CI.
- **FR-010**: The system MUST synchronize all benchmark items and split variations to the central observability platform under scoped virtual namespaces (`dev`, `test`, and `suites`).

### Key Entities

- **BenchmarkItem**: A discrete evaluation record containing user prompt (`input`), target answer (`expected_output`), and structured attributes (`metadata`).
- **FAQReferenceDoc**: A canonical administrative knowledge record representing a single FAQ row with unique identifier, category, question, and authoritative answer.
- **TextbookSectionGrounding**: A curriculum knowledge unit with document identifier, page number, section title, and pedagogical classification (`main_text`, `experiment`, `activity`, `diagram`).
- **RetrievalMetrics**: Quantitative measures of search quality, including Hit@k (k=1, 3, 5), Mean Reciprocal Rank (MRR), and Context Recall.
- **GenerationMetrics**: Quantitative measures of synthesis quality, including Groundedness/Faithfulness, Answer Relevancy, and Refusal Accuracy.
- **CitationVerification**: An evaluation result assessing whether source citations in a generated answer match reference ground truth chunks without hallucination.

---

## Clarifications *(Ratified 2026-10-09)*

### Session 1 - Evaluation & Generation Rules (2026-10-09)

1. **Q-1: Out-of-Scope & Adversarial Retrieval Evaluation**
   - **Question**: How should decoupled retrieval evaluation score out-of-scope and adversarial queries that have no valid factual chunks in the knowledge base?
   - **Resolution**: Exclude out-of-scope queries from retrieval hit-rate metrics (Hit@k, MRR) and score them strictly on refusal accuracy at the synthesizer stage.
   - **Impact**: Eliminates retrieval penalty on unserviceable queries; separates retrieval failures from intentional refusals.

2. **Q-2: Multi-Hop Partial Context Retrieval Scoring**
   - **Question**: How should decoupled retrieval evaluation score multi-hop queries when only a subset of the required gold chunks is retrieved in top-k?
   - **Resolution**: Linear proportional credit (`Context Recall = retrieved_gold_chunks / total_required_gold_chunks`). `Hit@k` counts as positive if at least one required gold chunk appears in top-k.
   - **Impact**: Provides smooth, informative retrieval gradients for complex queries rather than an artificial all-or-nothing penalty.

3. **Q-3: Citation Syntax Validation Standard**
   - **Question**: What citation syntax should the automated citation auditing engine accept and validate when scoring source provenance?
   - **Resolution**: Strict Markdown link format only (`[Document Title](file:///path#page=N)` or `[FAQ Title](faq_qa_pairs.csv#row=N)`). Plain bracketed text references (`[منبع: ...]`) are treated as unverified.
   - **Impact**: Enforces clickable, reproducible citation UX across all user-facing answers.

4. **Q-4: Sealed Test Set Isolation & Workflow Gating**
   - **Question**: How should the evaluation harness enforce split boundaries between the 75% Development set and the 25% Sealed Test set to prevent benchmark overfitting?
   - **Resolution**: Strict workflow isolation. Routine development and retrieval parameter tuning run exclusively against `--split dev` (318 items). The sealed `--split test` (107 items) is reserved strictly for pre-release quality gates and CI merge verification.
   - **Impact**: Prevents data leakage and ensures the test split remains an untainted empirical benchmark.

5. **Q-5: Synthetic Query Generation Determinism & Caching**
   - **Question**: How should temperature diversity and execution caching be handled during synthetic benchmark query generation?
   - **Resolution**: Partitioned temperature with local disk caching. Temperature is set to 0.2 for factoid and grounding extraction, and 0.7 for colloquial and adversarial paraphrasing. All generation steps are backed by an idempotent JSON cache to eliminate duplicate API calls on re-runs.
   - **Impact**: Guarantees reproducibility, cost efficiency, and resilient resumption after API rate limits or network hiccups.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Total benchmark dataset volume scales from 110 items to **400+ validated items** (minimum 200 administrative FAQ items and 180+ science curriculum items).
- **SC-002**: Curriculum coverage reaches **100% of textbook chapters** (all 15 chapters represented with at least 10 items each, up from 2 chapters previously).
- **SC-003**: 100% of benchmark items possess verifiable chunk-level source attribution (`document#page` or `document#id`).
- **SC-004**: Decoupled retrieval evaluation executes with zero model-judge token consumption and produces automated Hit@1, Hit@3, Hit@5, and MRR metrics.
- **SC-005**: 100% of benchmark files pass strict schema validation (exact column headers, valid JSON metadata, and zero null fields).
- **SC-006**: Benchmark dataset synchronizes to the central observability platform with zero duplicate records across all suites and splits.

---

## Assumptions

- **Ingestion Precondition**: The entire 15-chapter science textbook has already been converted into high-fidelity structured Markdown documents (`data/vlm_parsed/`) with preserved pedagogical section tags.
- **Knowledge Catalog Stability**: The 134 administrative FAQ entries in `faq_qa_pairs.csv` represent the complete authoritative grounding source for fund policies.
- **Evaluation Splitting**: A 75% dev and 25% test stratified split provides an optimal balance between development iteration capacity and statistically sound test gating.
- **Linguistic Standard**: Synthetic query generation reflects modern Iranian Persian language patterns, incorporating conversational idioms and zero-width non-joiner (نیم‌فاصله) conventions.

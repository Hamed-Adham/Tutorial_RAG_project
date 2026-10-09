# Research & Architectural Decisions: Benchmark Dataset Scaling & Comprehensive Quality Evaluation

This document records the architectural research, design rationales, and rejected alternatives for Feature 002: Benchmark Dataset Scaling & Comprehensive Quality Evaluation.

---

## 1. Synthetic Query Generation Model & Gateway

- **Decision**: Standardize on `gemini-cli` via the local Nine Router gateway (`http://localhost:20128/v1`) using strict structured JSON completion prompts, with dynamic fallback to `ag/claude-sonnet-4-6`.
- **Rationale**: 
  - `gemini-cli` provides sub-second latency, zero external per-token cloud costs, and high fluency in contemporary Persian.
  - Feeding ground-truth markdown chunks into a temperature-controlled prompt (`temperature=0.7` for paraphrases, `0.2` for ground-truth answers) produces natural colloquial mutations while strictly bounding factual hallucinations.
- **Alternatives Considered**:
  - *Manual human authoring for all 400+ queries*: Cost-prohibitive and slow; subject to human annotator fatigue (~50% error rate per Evaluation Manual).
  - *Naive rule-based string perturbation*: Generates unnatural grammar and fails to capture conversational Persian idioms or complex multi-hop dependencies.

---

## 2. Source Grounding Ingestion Architecture

- **Decision**: 
  1. **Science Curriculum**: Directly consume the pre-parsed, context-bridged Markdown files in `data/vlm_parsed/stitched/*.md` and section metadata in `data/vlm_parsed/manifest.json` (412 total sections across all 15 chapters).
  2. **Administrative FAQ**: Ingest `data/faq_qa_pairs.csv` (134 records), indexing each row with a deterministic canonical identifier (`faq_000` through `faq_133`).
- **Rationale**:
  - Raw textbook PDFs suffer from embedded Persian font reversing. Using the Milestone 14 VLM-extracted Markdown guarantees that all extracted queries are derived from 100% verified, legible text with preserved pedagogical tags (`<activity>`, `<experiment>`, `<test_yourself>`).
  - Assigning stable canonical IDs (`faq_###`) turns the FAQ into a discrete retrieval catalog, enabling deterministic `Hit@k` and `MRR` scoring.
- **Alternatives Considered**:
  - *Re-extracting raw PDF pages with PyMuPDF*: Rejected because PyMuPDF produced reversed Persian text and missed tables/diagrams.

---

## 3. Query Taxonomy & Difficulty Stratification

- **Decision**: Structure the 400+ benchmark items into 4 rigorous difficulty and complexity dimensions:
  1. **Administrative FAQ (200 items)**:
     - 60% Colloquial & Semantic Paraphrases (120 items): Conversational Persian, informal grammar, missing half-spaces (نیم‌فاصله).
     - 15% Cross-Category Lexical Ambiguity / Hard Negatives (30 items): Queries sharing terms (`ضمانت`, `تسهیلات`, `سود`, `وثیقه`) across different departments (*فروش اقساطی*, *حقوقی*, *میز خدمت*).
     - 15% Multi-Row Composite Queries (30 items): Requiring concurrent facts from 2–3 rows (e.g. loan ceilings AND required collateral).
     - 10% Out-of-Domain Refusals (20 items): Queries on retail/mortgage banking or non-fund activities requiring strict refusal.
  2. **Science Curriculum (180 items across 15 chapters, ~12 per chapter)**:
     - 40% Core Factoids (72 items): Point factual recall.
     - 25% Multi-Hop Reasoning (45 items): Synthesizing concepts across sections or chapters.
     - 20% Diagram Extraction (36 items): Anatomical, physical, and geological diagrams and figures.
     - 15% Experimental Procedures (27 items): Laboratory steps from `<experiment>` and `<activity>` tags.
  3. **Domain Routing & Disambiguation (25 items)**: Hard crossover queries between biological science and commercial biotechnology grants.
  4. **Adversarial & Refusal Traps (25 items)**: False-premise scientific traps, prompt injections, and unindexed syllabi.
- **Rationale**: Matches the formal AI Evaluation Workshop standards and guarantees that the benchmark tests retrieval depth, ranking quality, and synthesizer discipline.

---

## 4. Decoupled Retrieval vs. Generation Evaluation Engine

- **Decision**: Refactor the evaluation harness ([run_eval.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/evaluation/run_eval.py)) into two decoupled evaluation stages:
  1. **Stage A: Retrieval Evaluation (Deterministic Code Gate, 0 Tokens, < 5ms per item)**:
     - Measures: `Hit@1`, `Hit@3`, `Hit@5`, `MRR` (Mean Reciprocal Rank), `Context Recall` against `ground_truth_chunks`.
  2. **Stage B: Generation Evaluation (LLM-as-a-Judge / 5-Part Rubric)**:
     - Measures: `Faithfulness` (claim entailment), `Answer Relevancy` (query alignment), and `Adversarial Refusal Rate`.
- **Rationale**: Decoupling allows developers to run fast, zero-token retrieval benchmarks during vector DB and chunking experiments without paying LLM judge latency. When generation fails, developers know instantly whether the retriever or the synthesizer failed.

---

## 5. Source Citation & Provenance Attribution Metric

- **Decision**: Introduce a dedicated **Citation Fidelity Metric** in Stage 8 postprocessing:
  - Generated citations (e.g., `[C706_olom_haftom-7-144-124-131.pdf, p. 4]`) are parsed and matched against benchmark `ground_truth_chunks`.
  - **Citation Precision**: `(Valid Matched Citations) / (Total Citations Generated)`.
  - **Citation Recall**: `(Gold Chunks Cited) / (Total Gold Chunks Required)`.
- **Rationale**: Citations are critical for enterprise compliance. Penalizes "Ghost Citations" (hallucinated sources) and misattributions.

---

## 6. Dataset Format & Centralized Synchronization

- **Decision**:
  - Canonical format: [golden_set.jsonl](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/evaluation/datasets/golden_set.jsonl) (430 items total).
  - Partitioned CSV exports: 3-column CSVs (`input`, `expected_output`, `metadata`) saved under `suites/`, `dev/`, and `test/`.
  - Deterministic 75/25 train/test split: 322 Dev items / 108 Test items.
  - Idempotent synchronization to Langfuse Cloud using scoped virtual dataset names (`dev/*`, `test/*`, `suites/*`).
- **Rationale**: Ensures complete backwards compatibility with existing pipelines while providing clean CI/CD regression testing.

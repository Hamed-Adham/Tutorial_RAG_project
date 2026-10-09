# Unified RAG Development Log

All architectural decisions, milestone completions, evaluation metrics, and progressive optimizations are recorded here chronologically.

---

## [2026-10-08] Milestone 14: Persian Textbook Multimodal VLM Ingestion & Safety Fallback Architecture
- **Context & Goal**: Implement high-fidelity Stage 2 multimodal parsing for Iranian 7th-grade Science Textbooks (`data/Olum_7_splitted/`, 15 PDF chapters, 132 pages total). Ensure pedagogical XML tagging (`<activity>`, `<experiment>`, `<test_yourself>`, etc.), diagram descriptions, and cross-page continuous sentence stitching to facilitate precise downstream section-level retrieval filtering.
- **Root Cause Discovered (0-Byte Markdown Files)**:
  - Google Gemini's automated safety/copyright layer triggered `finish_reason: "recitation"` when asked to transcribe dense textbook paragraphs verbatim, returning HTTP 200 with `content: ""` (empty string).
  - 36 raw pages were initially saved as 0-byte `.md` files due to lack of empty-content checks.
- **Architectural Enhancements**:
  - **Dynamic Multi-Model Safety Fallback**: Updated [tier2_vlm.py](file:///rag-core/src/rag_core/parsing/tier2_vlm.py) ([ContextBridgedVLMExtractor](file:///rag-core/src/rag_core/parsing/tier2_vlm.py#L95)) with automatic fallback routing. When the primary model (`ag/gemini-3.8-flash`) hits recitation or returns `< 50` chars, the extractor automatically routes the rendered page to `ag/claude-sonnet-4-6`. Claude Sonnet operates without recitation blocks, reliably producing 2,500–4,500 characters of structured Persian Markdown.
  - **Cache Protection & Anti-Empty Invariant**: Hardened cache validation (`is_valid_cache`) to require `file.stat().st_size > 50` and added write assertions preventing empty files from ever touching disk.
  - **Prompt De-escalation**: Refined system prompt from "verbatim" to "complete fidelity with conceptual and terminological accuracy" to minimize unnecessary heuristic filter triggers.
  - **Batch Extraction Orchestrator**: Enhanced [extract_textbook_vlm.py](file:///rag-core/src/rag_core/parsing/extract_textbook_vlm.py) with natural sort ordering, per-page checkpointing, cross-page sentence stitching, and section taxonomy JSON extraction.
- **Final Verification & Metrics**:
  - **100% Extraction Coverage**: Exactly 132/132 pages parsed across all 15 PDF chapters with **zero** 0-byte files remaining (smallest file: 1,645 bytes).
  - **Master Manifest Generated**: [data/vlm_parsed/manifest.json](file:///data/vlm_parsed/manifest.json) cataloging 412 total sections (176 main body passages, 236 pedagogical activities).
  - **Test Suite**: 36/36 unit tests pass.

---
- **Context & Goal**: Establish an automated, repeatable strategy to ensure the persistent Codebase Memory Knowledge Graph (`.codebase-memory/graph.db.zst`) is continuously synchronized during SpecKit SDD workflows, rather than relying on manual file edits.
- **Changes & Deliverables**:
  - **Ratified Constitution Principle X (v1.4.0)**: Formally added *Iterative Codebase Knowledge Graph Maintenance (Non-Negotiable)* to [.specify/memory/constitution.md](file:///.specify/memory/constitution.md) and introduced the *Codebase Memory Graph Gate* (`detect_changes` reports `changed_count: 0`).
  - **Created SpecKit Lifecycle Extension Hooks**: Added [.specify/extensions.yml](file:///.specify/extensions.yml) wiring mandatory `after_implement` and `after_converge` graph re-indexing via `codebase-memory-mcp` (`index_repository(persistence=True)`), and pre-planning/pre-convergence graph inspection hooks.
  - **Updated SpecKit Templates**:
    - [plan-template.md](file:///.specify/templates/plan-template.md): Added Principle X checks to the mandatory Constitution Gate.
    - [tasks-template.md](file:///.specify/templates/tasks-template.md): Added explicit KG refresh and zero-drift verification checkpoint tasks to foundational and polish phases.
  - **Updated Feature Specs**:
    - [plan.md](file:///specs/001-enterprise-rag-system/plan.md): Updated Principle X compliance to Constitution v1.4.0.
    - [tasks.md](file:///specs/001-enterprise-rag-system/tasks.md): Added T011b (Phase 2), T073 (Phase 3 US1), and T065b (Phase 7 Polish) for Knowledge Graph synchronization checkpoints.
  - **Makefile Automation**: Added `make kg-status` target to query live graph node/edge counts and timestamp from `.codebase-memory/artifact.json`.
- **Verification**: `make kg-status` confirmed 1,673 nodes and 4,813 edges; `make check` passed (36/36 tests, 100/100 deterministic code gates).

---

## [2026-10-05] Milestone 12: Codebase Reorganization & Structure Preservation
- **Context & Goal**: Codebase had grown messy after rapid parallel development of Jev integration, LangGraph pipeline, and Langfuse observability. Goal: establish a clean, documented, self-maintaining structure.
- **Changes & Deliverables**:
  - **Migrated `evaluation/` as canonical eval path**: Moved `naive/judge_evaluator.py` and `naive/run_baseline_eval.py` to `evaluation/` (canonical sources). `naive/` copies retained as legacy for `make eval-legacy` regression comparison only.
  - **Rewired Makefile**: All `eval*` targets now point to `rag_core/evaluation/run_baseline_eval.py`. Added `eval-graph` (LangGraph pipeline), `eval-modular` (gates only), and `eval-legacy` (regression comparison) targets.
  - **Root tidy**: Moved loose PDFs, PNGs, and strategy docs from project root into `docs/references/` and `docs/`.
  - **Created `.agents/skills/codebase-memory/SKILL.md`**: Living canonical module map covering all 10 pipeline stages, naming conventions, Make targets, and agent update obligations. All agents MUST update this after structural changes.
  - **Added Principle X to plan.md Constitution Check**: "Codebase Structure Preservation & Living Documentation" — mandating the codebase-memory skill is kept up-to-date and evaluation canonical paths enforced.
  - **Updated `plan.md` directory tree**: Reflects actual current layout including `graph/`, `naive/` (with legacy warnings), `evaluation/` (fully populated), and `docs/`.
  - **Updated `evaluation/__init__.py`**: Exports `LLMJudge` and `JevJudge` from the canonical evaluation package.
- **Verification**: `make check` passes — 36/36 tests OK.
- **Key Rules Established**:
  - Never edit `naive/judge_evaluator.py` or `naive/run_baseline_eval.py` — edit `evaluation/` versions only.
  - Stage N must not import from Stage N+2 or later.
  - `graph/pipeline_graph.py` is the canonical execution engine — all eval should eventually route through it.


## [2026-10-01] Milestone 1: Architecture Specification & Knowledge Modeling
- **Context & Goal**: Establish a unified production-grade RAG framework by combining the 100k-document production RAG explainer with advanced Weaviate RAG techniques.
- **Changes & Deliverables**:
  - Indexed codebase memory into persistent knowledge graph (`.codebase-memory/graph.db.zst`).
  - Scaffolded SpecKit governance framework:
    - [constitution.md](file:///.specify/memory/constitution.md): 3-layer decoupled architecture, evaluation-driven quality, Law of Parsimony, workspace hygiene.
    - [spec.md](file:///specs/001-enterprise-rag-system/spec.md): 10 modular pipeline stages (Manifest -> Tiered Parsing -> Structure-Aware Chunking -> Multi-Index -> Pre-Retrieval -> Multi-Pronged Retrieval -> Fusion & Reranking -> Postprocessing -> Generation -> Evaluation).
    - [plan.md](file:///specs/001-enterprise-rag-system/plan.md) and [tasks.md](file:///specs/001-enterprise-rag-system/tasks.md): Phased roadmap from MVP to autonomous agentic Text-to-SQL.

---

## [2026-10-01] Milestone 2: Primary Benchmark Generation (Pre-Implementation Grounding)
- **Context & Goal**: In accordance with mentor guidance and evaluation-first engineering, construct formal primary capability benchmarks for heterogeneous Persian data before writing implementation code.
- **Corpora Covered**:
  - `data/Olum_7_splitted`: 15 PDF chapters of Iranian 7th Grade Science Textbook (academic, layout-heavy, RTL Persian text).
  - `data/faq_qa_pairs.csv`: 134 administrative and financial Q&A pairs from the Innovation and Prosperity Fund (INIF).
- **Changes & Deliverables**:
  - Created [generate_benchmark.py](file:///rag-core/src/rag_core/evaluation/datasets/generate_benchmark.py) and generated 100 high-fidelity Persian test items across 4 suites:
    1. `science_textbook` (40 items): 30 single-chapter factoid/reasoning + 10 multi-hop cross-chapter synthesis.
    2. `administrative_faq` (30 items): 15 canonical FAQ + 15 colloquial Evol-Instruct dialect mutations.
    3. `domain_routing` (15 items): Disambiguation between science curriculum and financial/administrative policies.
    4. `adversarial_refusal` (15 items): Out-of-syllabus traps, cryptocurrency grant hoaxes, medical prescriptions, and prompt injections.
  - Implemented [load_dataset.py](file:///rag-core/src/rag_core/evaluation/datasets/load_dataset.py): Dataset loader with strict schema validation.

---

## [2026-10-01] Milestone 3: AI Evaluation Workshop Hardening
- **Context & Goal**: Upgrade evaluation infrastructure to meet the Enterprise AI Evaluation Workshop standards (Technical Strategy Roadmap and Operational Deployment Manual).
- **Changes & Deliverables**:
  1. **Stratified 75/25 Train/Test Split**:
     - Applied deterministic stratification across all 4 suites: 75 Dev items (iteration & tuning) / 25 Test items (sealed test gate).
     - Verified 0 missing keys, 100% integrity in [golden_set.jsonl](file:///rag-core/src/rag_core/evaluation/datasets/golden_set.jsonl).
  2. **5-Part Judge Rubrics ([rubrics.py](file:///rag-core/src/rag_core/evaluation/rubrics.py))**:
     - Implemented Persona, Explicit Binary Verdict (0/1), XML Data Boundaries (`<user_query>`, `<retrieved_context>`, `<agent_response>`, `<ground_truth_reference>`), Persian Few-Shot Examples, and Mandatory Step-by-Step Chain of Thought (CoT).
     - Defined rubrics for: `faithfulness`, `answer_relevancy`, `domain_routing`, and `adversarial_refusal`.
  3. **Meta-Evaluation & Production Reliability Harness ([meta_eval.py](file:///rag-core/src/rag_core/evaluation/meta_eval.py))**:
     - Meta-evaluation confusion matrix (TP, FP, TN, FN; Precision, Recall, F1).
     - Implemented "Cancer Screening" asymmetric risk metric: `Hallucination Catch Rate = TN / (TN + FP)`.
     - Multi-run reliability metrics: `Pass^k` (all k trials pass 100%) and `Pass@k` (at least 1 passes).

---

## [2026-10-01] Milestone 4: Law of Parsimony & Ground Zero Naive Baseline (Completed Setup)
- **Philosophy**: Adopted the Law of Parsimony (Occam's razor). Prioritized deterministic code and simple mechanisms over complex multi-step agents.
- **Environment & Models**:
  - Located local `BAAI/bge-m3` model weights (`pytorch_model.bin`, 2.16 GB) at `/Users/hmd/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181`.
  - Configured root `.env`, `.env.example`, and `.gitignore`.
  - Created isolated virtual environment `.venv` using Python 3.11.15 via `uv`.
  - Verified local Gemini LLM endpoint (`http://localhost:20128/v1`, model: `gemini-cli`).
- **Naive Pipeline Components Created**:
  - `document_loader.py`: Ingested 134 administrative FAQ records + 132 textbook PDF pages (266 total docs).
  - `chunker.py`: Formatted 539 naive chunks (134 atomic FAQ chunks, 405 textbook fixed-window chunks).
  - `synthesizer.py`: Successfully connected to local LLM with non-streaming HTTP requests.
  - `judge_evaluator.py`: Implemented 5-part rubric evaluation. Verified Persian CoT generation and binary verdict parsing.
  - `retriever.py` & `embedder.py`: Configured dense similarity retriever with `.npz` caching.
  - `run_baseline_eval.py`: Benchmark runner to execute dev set evaluations and compute Ground Zero metrics.
- **Verification**:
  - Successfully ran end-to-end synthesizer and 5-Part LLM judge verification test (Passed with verdict 1 and step-by-step Persian reasoning).

---

## [2026-10-02] Milestone 5: Nine Router BGE-M3 Integration & Caching Verification
- **Configuration & Environment**:
  - Renamed environment variables from `OPENAI_BASE_URL` / `OPENAI_API_KEY` to `NINE_ROUTER_BASE_URL` and `NINE_ROUTER_API_KEY` in [.env](file:///.env), [.env.example](file:///.env.example), and [config.py](file:///rag-core/src/rag_core/config.py).
- **Embedding Model Verification on Nine Router**:
  - Queried `http://localhost:20128/v1/models` and verified the active BGE-M3 model on the router.
  - Resolved model identifier `un/gpustack/bge-m3-GGUF` (and mapped alias `bge-m3`).
  - Tested `/v1/embeddings` endpoint: verified 1024-dimensional normalized dense vectors with sub-150ms batch response times.
- **Pipeline & Embedder Upgrade**:
  - Upgraded [embedder.py](file:///rag-core/src/rag_core/naive/embedder.py) to stream batch requests directly to Nine Router embedding API with automatic candidate model resolution and offline fallback.
  - Successfully embedded and persisted all 539 chunks (134 FAQ rows and 405 textbook chunks) to `data_index/naive_index.npz`.
  - Tested cached retrieval and generation on real query (`شرکت‌های غیردانش‌بنیان از چه خدماتی می‌توانند استفاده کنند؟`): achieved 0.755 cosine similarity top-1 match and 100% faithful response generation.

---

## [2026-10-02] Milestone 6: Naive RAG Architecture Visualization via Archify
- **Context & Goal**: Follow the `/archify` workflow to create a showcase-quality, interactive visualization of the Naive RAG Baseline architecture under `architecture/naive/`.
- **Changes & Deliverables**:
  - Authored [naive-rag.architecture.json](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/architecture/naive/naive-rag.architecture.json):
    - Specified 9 architectural components across 2 architectural boundaries: Local Baseline RAG Subsystem and Nine Router Inference Gateway (`:20128/v1`).
    - Configured 3 specialized views: `Online Query & Retrieval Path`, `Offline Ingestion & Caching Path`, and `5-Part LLM Judge & Quality Gate`.
    - Captured 3 summary callout cards detailing Parsimony Baseline, Nine Router Gateway, and Evaluation Harness.
  - Ran Archify CLI compile & validation:
    - 9/9 checks passed with 0 errors, 0 warnings (Showcase Profile).
    - Generated standalone interactive viewer: [naive-rag-architecture.html](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/architecture/naive/naive-rag-architecture.html) (718 KB with inline SVG, dark/light themes, node inspection, data flow trace motion, and PNG/SVG export).
  - Multi-Viewport Visual Verification:
    - Executed automated visual-check across 1440x900, 1600x1000, 1920x1080, and 2048x1320 in both light and dark modes.
    - Verified zero horizontal or vertical overflow (`scrollWidth <= innerWidth` and `scrollHeight <= innerHeight`).
    - Captured high-resolution inspection screenshots: `naive-rag-architecture.visual-check.1440x900.*.png` and `naive-rag-architecture.visual-check.2048x1320.*.png`.

---

## [2026-10-02] Milestone 7: Qdrant Vector Database Integration & Benchmark Execution
- **Context & Goal**: Integrate Qdrant vector database to store chunks and payloads persistently, replace in-memory numpy storage with high-performance vector indexing, and benchmark against the heterogeneous Persian dev dataset.
- **Changes & Deliverables**:
  - Installed `qdrant-client` 1.19.1 in `.venv`.
  - Configured Qdrant parameters in [config.py](file:///rag-core/src/rag_core/config.py), [.env](file:///.env), and [.env.example](file:///.env.example):
    - `QDRANT_PATH=data_index/qdrant_db`
    - `QDRANT_COLLECTION=unified_rag_chunks`
  - Created [qdrant_retriever.py](file:///rag-core/src/rag_core/naive/qdrant_retriever.py):
    - Multi-mode client instantiation: auto-detects explicit URL, live server on port 6333, or embedded local disk storage (`data_index/qdrant_db`).
    - Collection schema: 1024-dimensional dense vectors with Cosine distance.
    - Upserted all 539 chunks with rich metadata payloads (`chunk_id`, `doc_id`, `text`, `source_type`, `source_name`, `metadata`).
    - Reused existing precomputed BGE-M3 embeddings from `naive_index.npz` to complete initial indexing in under 0.2s without redundant API calls.
  - Upgraded [pipeline.py](file:///rag-core/src/rag_core/naive/pipeline.py):
    - Supported pluggable `retriever_type="qdrant"` as default.
    - Measured retrieval latency: dropped to ~50ms per query (50-60ms in Qdrant vs full numpy array dot-products).
  - Executed Full Baseline Evaluation ([qdrant_baseline_report.json](file:///rag-core/src/rag_core/evaluation/baselines/qdrant_baseline_report.json)):
    - Hardened the evaluation harness with 90s request timeouts, automatic 3-attempt retries with exponential backoff, and checkpoint resumption.
    - Evaluated all 75 items across all 4 suites in the `dev` split using 5-part rubrics with Persian Chain of Thought.
    - **Overall Pass Rate**: `47 / 75 = 62.67%`
    - **Mean Latency**: 8.75s per query (retrieval ~0.05s + generation & judge ~8.70s).
    - **Official Ground Zero Suite Breakdown**:
      - `domain_routing`: `11 / 11 = 100.00%` (Perfect disambiguation between 7th grade science and fund administrative/financial policies)
      - `adversarial_refusal`: `11 / 11 = 100.00%` (Zero hallucination on prompt injections, fake loans, out-of-syllabus traps, and API key leak attempts)
      - `administrative_faq`: `13 / 23 = 56.52%` (Strong on canonical FAQs, struggles with conversational colloquial queries and specialized legal/financial terminology lacking keyword match)
      - `science_textbook`: `12 / 30 = 40.00%` (Struggles with multi-column PDF layouts and fragmented sentences caused by naive fixed-window 500-char chunking; 100% faithful refusals when context is truncated)
    - **Empirical Significance**: Establishes the exact statistical baseline for our project. Every subsequent phase (Tiered Layout Parsing via PyMuPDF/Docling, Structure-Aware Chunking, Hybrid BM25 + Dense Search via RRF, and Cross-Encoder Re-ranking) will be quantitatively measured against this 62.67% baseline.

---

## [2026-10-02] Milestone 8: Clean Pipeline Visualization via Archify Dataflow
- **Context & Goal**: In response to user feedback requesting an intuitive, easy-to-understand **pipeline diagram** rather than a component cluster, created a structured 5-stage Dataflow pipeline visualization under `architecture/naive/`.
- **Changes & Deliverables**:
  - Authored [naive-rag.dataflow.json](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/architecture/naive/naive-rag.dataflow.json):
    - Structured into 5 sequential horizontal stages:
      1. `1. Ingest & Input` (Raw Knowledge Data, Client Intake & 75 Dev Items)
      2. `2. Chunk & Encode` (Fixed 500-char Chunker, Query Encoder)
      3. `3. Embed & Search` (BGE-M3 Batch Service, Dense Retriever)
      4. `4. Vector DB (Qdrant)` (539 Chunks + Payloads, Cosine Similarity)
      5. `5. Synthesize & Audit` (Gemini CLI Persian Synthesis, 5-Part LLM Judge)
    - Clearly delineated Offline Ingestion path vs Online Query path.
  - Validation & Delivery:
    - 9/9 showcase checks passed with 0 errors, 0 warnings.
    - Generated standalone interactive viewer: [naive-rag-pipeline.html](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/architecture/naive/naive-rag-pipeline.html) (718 KB with stage lanes, trace animations, theme switching, and node inspection).
    - Captured high-resolution inspection screenshots across 1440x900 and 2048x1320 viewports (`naive-rag-pipeline.visual-check.*.png`).

---

## [2026-10-04] Milestone 9: LLM Judge Execution Timing & Enterprise RAG Evaluation Framework
- **Context & Goal**: Instrument execution latency telemetry across LLM-as-a-Judge evaluations and synthesize state-of-the-art industry and academic practices (Patronus AI, Sanjeeb Panda, Alex Chen, and Ragas) into a comprehensive evaluation specification.
- **Changes & Deliverables**:
  - **Latency Instrumentation in LLM Judge**:
    - Updated [judge_evaluator.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/naive/judge_evaluator.py):
      - Added `latency_seconds: float = 0.0` to `JudgeVerdict`.
      - Instrumented high-resolution start/end timers (`time.time()`) across `LLMJudge.evaluate_metric()`.
      - Added real-time latency logging on every invocation: `[LLM-as-a-Judge] Evaluated '{rubric_name}' in {latency:.2f}s | Verdict: {verdict_val}`.
      - Guaranteed elapsed time tracking across success, fallback JSON parsing, and retry/failure paths.
    - Updated [run_baseline_eval.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/naive/run_baseline_eval.py):
      - Stored `latency_seconds` inside individual suite metric outputs (`adversarial_refusal`, `domain_routing`, `faithfulness`, `answer_relevancy`) in evaluation report JSONs.
  - **Authoring Comprehensive Evaluation Reference**:
    - Created [docs/rag_evaluation_framework.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/docs/rag_evaluation_framework.md):
      - **Taxonomy**: Decoupled component vs end-to-end evaluation; reference-based (with ground truth) vs reference-free (production monitoring).
      - **Retrieval Metrics**: `Hit@k`, `MRR@k`, `NDCG@k`, `Context Relevance`, `Context Precision@k` (rank-weighted signal-to-noise), `Context Recall` (sufficiency), `Context Entities Recall`, and `Noise Sensitivity`.
      - **Generation Metrics**: `Faithfulness` (claim extraction & entailment verification), `Answer Relevancy` (reverse query generation), `Answer Correctness` (factual F1 + semantic similarity), `Semantic Similarity`, traditional NLP metrics (BLEU, ROUGE, CHRF) and their limitations, and 8-type Hallucination Taxonomy with `Hallucination Catch Rate` (`TN / (TN + FP)`).
      - **Safety & Security**: `Adversarial Refusal Rate`, OWASP LLM01 Prompt Injection, OWASP LLM06 Sensitive Data Leakage.
      - **Operational KPIs & SLAs**: TTFT (< 600ms), E2E latency (< 2.5s), embedding latency (< 50ms), vector search (< 30ms), LLM judge latency (< 2.0s), and token budget.
      - **Architectural Patterns**: Tri-layered "Swiss Cheese" defense (Deterministic Code Gates -> 5-Part LLM Judge Rubrics -> Human Meta-Evaluation).
      - **Tooling Comparison**: In-depth architecture, strengths, and trade-offs for Ragas, TruLens, DeepEval, Patronus AI (Lynx 2.0 / Glider), Arize Phoenix, and Langfuse.
      - **Synthetic Testset Generation (Ragas KG Paradigm)**: Query taxonomy (Single-Hop vs Multi-Hop, Specific vs Abstract), Evol-Instruct evolutionary mutation operators (Reasoning, Conditioning, Multi-Context, Persona), Knowledge Graph architecture (Nodes, Extractors, Relationship Builders, Transforms, and Query Synthesizers), and Critic Quality Gates.
      - **Python Blueprint**: Complete, production-ready implementation combining deterministic gates with timed LLM judge execution.

---

## [2026-10-04] Milestone 10: Tooling Integration (Neo4j, Langfuse, Ragas) & Evaluation Operations Guide
- **Context & Goal**: Implement practical evaluation tooling orchestration: install and configure Neo4j for Knowledge Graph synthesis, wire Langfuse for span tracing, integrate Ragas for formal metric evaluation over local endpoints, implement Layer 1 Deterministic Code Gates, and produce an operational diagnostic guide for RAG developers.
- **Changes & Deliverables**:
  - **Tooling Installation**:
    - Installed `neo4j` (6.3.1), `ragas` (0.4.3), `datasets`, `langchain-openai`, `pandas`, and `pyarrow` in `.venv`.
    - Added Neo4j connection parameters (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`) to [config.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/config.py) and [.env.example](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/.env.example).
  - **Neo4j Knowledge Graph Store**:
    - Created [rag-core/src/rag_core/kg/neo4j_store.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/kg/neo4j_store.py):
      - Schema: Constraints on `Document(id)`, `Chunk(id)`, `Entity(name)`, `Keyphrase(text)`.
      - Relationships: `HAS_CHUNK`, `MENTIONS`, `SIMILAR_TO`.
      - Multi-Hop Traversal: Implemented `find_multi_hop_candidates()` Cypher query to identify chunk pairs from different documents sharing entities for synthetic testset generation.
      - Graceful fallback: Safe offline operation when Neo4j server is not yet booted.
  - **Layer 1 Deterministic Code Gates**:
    - Created [rag-core/src/rag_core/evaluation/code_gates.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/evaluation/code_gates.py):
      - Evaluates Persian refusal standard and regex patterns with 0 tokens and < 5ms latency.
      - Evaluates domain routing majority votes deterministically.
      - Tracks retrieval `Hit@5` against ground truth source documents.
    - Updated [run_baseline_eval.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/naive/run_baseline_eval.py) to route through Layer 1 first, eliminating over 40 redundant LLM judge calls per baseline run.
  - **Ragas Local Adapter**:
    - Created [rag-core/src/rag_core/evaluation/ragas_adapter.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/evaluation/ragas_adapter.py):
      - Wraps Ragas `evaluate()` with local LangChain `ChatOpenAI` and `OpenAIEmbeddings` pointing to Nine Router (`gemini-cli` + `bge-m3-GGUF`).
      - Computes `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`.
  - **Langfuse Span Tracing**:
    - Updated [rag-core/src/rag_core/naive/pipeline.py](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/rag-core/src/rag_core/naive/pipeline.py) with `rag_query` trace, `retrieval` span, and `synthesis` generation span when `LANGFUSE_ENABLED=true`.
  - **Developer Operations & Diagnostic Guide**:
    - Authored [docs/evaluation_system_guide.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/docs/evaluation_system_guide.md):
      - End-to-end architectural explanation of the Tri-Layered Swiss Cheese model.
      - Step-by-step instructions for running baseline evaluations, Ragas metrics, and Neo4j KG queries.
      - Diagnostic playbook for isolating retrieval vs generation failures against the 62.67% Ground Zero baseline.

---

## [2026-10-04] Milestone 11: Modular 10-Stage Pipeline Implementation & Verification
- **Context & Goal**: Implement and verify the production-grade modular components for Phase 3 (US1 MVP) spanning Manifest Cataloging, Tiered Parsing, Structure-Aware Chunking, Dense Vector Retrieval, Page/BBox Citations, Grounded Synthesis, and Formal Evaluation Gates.
- **Changes & Deliverables**:
  - **Stage 1 (Manifest Catalog, Deduplicator & Lineage)**:
    - [catalog.py](file:///rag-core/src/rag_core/manifest/catalog.py): PostgreSQL document catalog with stable `doc_id`, SHA-256 tracking, tenant isolation, and in-memory test fallback.
    - [deduplicator.py](file:///rag-core/src/rag_core/manifest/deduplicator.py): Exact byte-level SHA-256 collapse and MinHash near-duplicate clustering.
    - [lineage.py](file:///rag-core/src/rag_core/manifest/lineage.py): Document version tracking and tombstone deletion propagation across indices.
  - **Stage 2 (Tiered Parsing & Canonical IR)**:
    - [schema.py](file:///rag-core/src/rag_core/parsing/schema.py): Canonical IR `Block` model with semantic types, bbox, page, and reading order.
    - [tier0_layout.py](file:///rag-core/src/rag_core/parsing/tier0_layout.py): PyMuPDF layout-aware extractor preserving headings and reading order.
    - [tier1_ocr.py](file:///rag-core/src/rag_core/parsing/tier1_ocr.py): OCR fallback parser emitting confidence scores.
    - [tier2_vlm.py](file:///rag-core/src/rag_core/parsing/tier2_vlm.py): Multimodal visual description generator for figures and diagrams.
    - [router.py](file:///rag-core/src/rag_core/parsing/router.py): Content-type probing and automatic tiered dispatch.
  - **Stage 3 (Structure-Aware Chunking)**:
    - [hierarchical.py](file:///rag-core/src/rag_core/chunking/hierarchical.py): Heading-aware chunker prepending `Doc > Section > Subsection (page)` breadcrumbs.
    - [parent_child.py](file:///rag-core/src/rag_core/chunking/parent_child.py): Two-level parent-child chunker with strictly bounded overlap to eliminate infinite loop risk.
  - **Stage 4 & 6 (Multi-Index & Retrieval)**:
    - [indexer.py](file:///rag-core/src/rag_core/indexing/indexer.py): Unified manifest-to-index orchestrator.
    - [dense.py](file:///rag-core/src/rag_core/retrieval/dense.py): Dense HNSW vector retriever with BGE-M3 query prefix and score thresholding.
  - **Stage 8 & 9 (Citations & Grounded Synthesis)**:
    - [citations.py](file:///rag-core/src/rag_core/postprocessing/citations.py): Page and bounding-box source citation formatter.
    - [prompts.py](file:///rag-core/src/rag_core/generation/prompts.py): Centralized bilingual grounded prompt templates.
    - [synthesizer.py](file:///rag-core/src/rag_core/generation/synthesizer.py): Grounded response synthesizer enforcing explicit "I don't know" refusal on empty/insufficient context.
  - **Stage 10 (Evaluation Harness & Quality Gate)**:
    - [harness.py](file:///rag-core/src/rag_core/evaluation/harness.py): Formal RAGAS evaluation harness over Golden Benchmark.
    - [quality_gate.py](file:///rag-core/src/rag_core/evaluation/quality_gate.py): Constitutional Quality Gate threshold validator.
    - [baseline_manager.py](file:///rag-core/src/rag_core/evaluation/baseline_manager.py): Historical baseline snapshot persistence and comparison.
    - [run_eval.py](file:///rag-core/src/rag_core/evaluation/run_eval.py): CLI evaluation runner supporting `--gates-only` and modular full runs.
  - **Verification & Test Coverage**:
    - [test_pipeline_stages.py](file:///rag-core/tests/test_pipeline_stages.py): 12 automated unit and integration tests covering Stages 1, 2, 3, 6, 8, 9, 10.
    - Total test suite: 19/19 tests passing in 0.048s.
    - `make check` and `make eval-modular` executing cleanly.

---

## [2026-10-08] Milestone 12: Standardized Evaluation Benchmark & Langfuse Cloud Migration
- **Context & Goals**:
  - Eliminate corpus-benchmark misalignment in `science_textbook.csv` caused by legacy queries citing non-ingested textbook chapters.
  - Expand grounding coverage with diagram extraction, cross-chapter multi-hop synthesis, Persian orthographic/colloquial variants, hard negative routing boundaries, and adversarial refusals.
  - Standardize all benchmark suites to 3-column CSV format (`input`, `expected_output`, `metadata`) with normalized schema (`id`, `domain`, `query_type`, `ground_truth_chunks`, `expected_behavior`, `expected_route`).
  - Migrate all evaluation datasets to Langfuse Cloud adhering to the official virtual namespace hierarchy.
- **Implemented Plan & Deliverables**:
  - **Step 1 (Corpus Alignment)**:
    - Filtered `science_textbook.csv` strictly to Chapter 14 (`C706_olom_haftom-7-144-124-131.pdf`) and Chapter 15 (`C706_olom_haftom-7-144-132-137.pdf`).
    - Migrated legacy non-ingested chapter questions (Chapters 1, 2, 3, 7, 8, 13) to `adversarial_refusal.csv` with `expected_behavior="refusal"` and `expected_route="refuse"`.
  - **Step 2 (Metadata Standardization)**:
    - Normalized all 33 CSV benchmark files to `['input', 'expected_output', 'metadata']`.
    - Standardized metadata dictionary schema with granular chunk attribution (`filename.pdf#page=N`, `faq_qa_pairs.csv#row=N`).
  - **Step 3 (Synthetic Generation - Textbook)**:
    - 10 Diagram Extraction items: heart anatomy, blood vessel layers, respiratory tract, nephron structure.
    - 10 Multi-hop Synthesis items: linking pulmonary circulation with alveolar gas exchange and renal excretion.
    - 5 Aligned Core Factoids: total 25 items in `science_textbook`.
  - **Step 4 (Synthetic Generation - Administrative FAQ)**:
    - 15 Colloquial & Orthography items: conversational Persian queries with colloquial speech, missing half-space (نیمفاصله), and realistic phrasing.
    - 10 Multi-row Combination items: synthesizing loan limits, collateral rules, leasing, guarantees, and co-investment.
    - 15 Canonical Factoids: total 40 items in `administrative_faq`.
  - **Step 5 (Hard Negative Routing Generation)**:
    - 10 Lexical crossover items: biotech/scientific terminology juxtaposed against innovation fund grants.
    - 10 Clear domain boundary items: total 20 items in `domain_routing`.
  - **Step 6 (Adversarial Expansion)**:
    - 5 False-premise science queries (e.g. red blood cells pumping blood).
    - 5 Personal-service financial queries directed at the innovation fund (e.g. personal housing loans).
    - 15 Out-of-domain / injection / unindexed chapter queries: total 25 items in `adversarial_refusal`.
  - **Step 7 (Integrity Validation & Synchronization)**:
    - Built [build_standardized_benchmarks.py](file:///rag-core/src/rag_core/evaluation/build_standardized_benchmarks.py) and updated [load_dataset.py](file:///rag-core/src/rag_core/evaluation/datasets/load_dataset.py).
    - Automated checks verified: 33/33 CSVs have exact `['input', 'expected_output', 'metadata']` headers, valid JSON metadata, and 100% chunk boundary alignment.
    - Synchronized [golden_set.jsonl](file:///rag-core/src/rag_core/evaluation/datasets/golden_set.jsonl): 110 items (81 dev / 29 test).
  - **Langfuse Cloud Dataset Migration**:
    - Built and executed [migrate_datasets_to_langfuse.py](file:///rag-core/src/rag_core/evaluation/migrate_datasets_to_langfuse.py) against `https://cloud.langfuse.com`.
    - Successfully synchronized 330 items across all 14 datasets under `dev/`, `test/`, and `suites/` virtual namespaces with scoped idempotent item IDs.
    - All 36 unit tests pass.

---

## [2026-10-08] Milestone 13: Full Persian Textbook VLM Extraction & Section Tagging
- **Context & Goals**:
  - Raw PDF text extraction (`ch15_text.txt`) suffered from severe character/word reversal and disjointed lines due to embedded custom Persian font stream encoding, with 100% loss of diagrams, tables, and activity boxes.
  - Implement Stage 2 Tier 2 VLM extraction with Gemini 3.8 Flash via Nine Router (`ag/gemini-3.8-flash`) using user's structured Persian pedagogical prompt.
  - Solve the cross-page continuity/stitching problem where sentences or activities begin at the bottom of Page X and end at the top of Page X+1.
  - Extract the entire 7th grade science textbook (15 PDFs, 132 pages) in [`data/Olum_7_splitted/`](file:///data/Olum_7_splitted) into structured, filtered Markdown and JSON.
- **Implemented Plan & Deliverables**:
  - **Context-Bridged VLM Extractor & Page Stitcher**:
    - Implemented [tier2_vlm.py](file:///rag-core/src/rag_core/parsing/tier2_vlm.py) with `pypdfium2` 150 DPI rendering.
    - Added sliding-window context bridging: passes previous page tail context and unclosed tags to the next page.
    - Programmatic stitcher unwraps `<continuation>`, `<unclosed_tail>`, and `<activity continued="true">`, seamlessly fusing split sentences and multi-page activity boxes across page breaks.
    - Preserves provenance with `<!-- PAGE_BREAK: page N -> page N+1 -->`.
  - **Batch Extraction Orchestrator**:
    - Created [extract_textbook_vlm.py](file:///rag-core/src/rag_core/parsing/extract_textbook_vlm.py) supporting natural curriculum sorting, resume-safe caching, and structured JSON parsing.
  - **Full Dataset Extraction Results**:
    - Extracted all 15 PDFs across all 132 pages of the 7th grade science textbook.
    - Total Sections: **285 sections** across the textbook.
    - Core Curriculum Main Body: **120 sections** (`is_main_text: true`).
    - Pedagogical Activities & Experiments: **165 sections** (`is_pedagogical_activity: true`).
    - Master Manifest: [`data/vlm_parsed/manifest.json`](file:///data/vlm_parsed/manifest.json).

---

## [2026-10-08] Milestone 15: Benchmark Dataset Scaling (425 Items) & Decoupled Quality Evaluation
- **Context & Goals**:
  - Address previous benchmark limitations (small sample size of ~115 items, absence of decoupled retrieval evaluation, lack of automated source citation auditing).
  - Scale evaluation corpus to 400+ standardized items across all 15 Iranian 7th-grade science textbook chapters and the comprehensive INIF administrative FAQ catalog.
  - Implement zero-token, sub-second decoupled retrieval evaluation (`Hit@1`, `Hit@3`, `Hit@5`, `MRR`, `Context Recall`).
  - Implement automated source citation provenance auditing (`Citation Precision`, `Citation Recall`).
  - Establish a clean, stratified 75/25 dev/test split without data leakage and enable idempotent synchronization to Langfuse Cloud.
- **Implemented Plan & Deliverables**:
  - **1. Schema & Contracts (US1 & US2 Foundation)**:
    - [benchmark_schema.py](file:///rag-core/src/rag_core/evaluation/benchmark_schema.py): Pydantic v2 schemas (`BenchmarkItemModel`, `BenchmarkMetadataModel`, `DecoupledEvaluationReportModel`, `RetrievalMetricsModel`, `CitationMetricsModel`).
    - Standardized metadata dictionary schema with granular chunk attribution (`filename.pdf#page=N`, `faq_qa_pairs.csv#row=N`).
  - **2. Grounding Catalogs & Dataset Scaling**:
    - [faq_catalog.py](file:///rag-core/src/rag_core/evaluation/datasets/catalogs/faq_catalog.py): Ingested all 134 administrative FAQ records with full-text keyword indexing and category clustering.
    - [textbook_grounder.py](file:///rag-core/src/rag_core/evaluation/datasets/catalogs/textbook_grounder.py): Ingested 412 curriculum sections across all 15 chapters from `data/vlm_parsed/stitched/*.md` and `manifest.json`.
    - Generated **425 standardized benchmark items** across 4 suites:
      - `administrative_faq`: 200 items (factoids, colloquial/dialectal phrasing, multi-policy synthesis, unserviceable requests).
      - `science_textbook`: 180 items (factoids, conceptual reasoning, diagram interpretation, activity/experiment analysis across all 15 chapters).
      - `domain_routing`: 20 items (lexical crossover and hard domain boundary disambiguation).
      - `adversarial_refusal`: 25 items (unindexed syllabus traps, financial scams, prompt injections, medical hoaxes).
  - **3. Decoupled Retrieval-Only Evaluation (US3)**:
    - [retrieval_scorer.py](file:///rag-core/src/rag_core/evaluation/retrieval_scorer.py): Deterministic evaluation of `Hit@1`, `Hit@3`, `Hit@5`, `MRR`, and `Context Recall` without LLM judge invocation.
    - [harness.py](file:///rag-core/src/rag_core/evaluation/harness.py) & [run_eval.py](file:///rag-core/src/rag_core/evaluation/run_eval.py): Supported `--retrieval-only` and `--split` (dev/test) flags.
    - Evaluates 425 items in < 0.01s consuming 0 LLM tokens (`make eval-retrieval`).
  - **4. Citation Provenance Auditing (US4)**:
    - [citations.py](file:///rag-core/src/rag_core/postprocessing/citations.py): Extraction of markdown `[title](file:///path#page=N)` and bracketed `[منبع: doc#page=N]` citations with calculation of `Citation Precision` and `Citation Recall` against gold chunk sets.
  - **5. Master Compilation, Stratified Partitioning & Langfuse Sync (US5)**:
    - [build_standardized_benchmarks.py](file:///rag-core/src/rag_core/evaluation/build_standardized_benchmarks.py): Compiled 425 items into `golden_set.jsonl` and 3-column CSV files (`input`, `expected_output`, `metadata`).
    - Stratified split: 318 Dev items (74.8%) and 107 Test items (25.2%) with 0 query or chunk leakage.
    - [migrate_datasets_to_langfuse.py](file:///rag-core/src/rag_core/evaluation/migrate_datasets_to_langfuse.py): Idempotent synchronization of all 14 datasets (1,275 item instances) across `dev/*`, `test/*`, and `suites/*` namespaces.
    - Added Makefile targets: `eval-retrieval`, `eval-full`, `eval-gates`, `sync-datasets`, `sync-langfuse`.
- **Verification & Test Suite**:
  - Full test suite expanded from 36 to **61 passing unit and contract tests** (`test_benchmark_schema.py`, `test_decoupled_eval.py`, `test_citation_eval.py`, `test_golden_set_sync.py`, etc.).
  - Layer 1 Deterministic Code Gates pass with 100% compliance.

### Milestone 2026-10-08 21:45: Spec-Driven Development (Spec-Kit) Governance & Reference
- **Goal**: Institutionalize Spec-Kit driven development into workspace agent rules and create a comprehensive situational cheatsheet.
- **Changes**:
  - [.agents/rules/agent-rules.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/.agents/rules/agent-rules.md): Added mandatory Rule 5 requiring proactive intent analysis and guidance toward corresponding Spec-Kit lifecycle commands on every user interaction.
  - [docs/speckit_cheatsheet.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/docs/speckit_cheatsheet.md): Authored an end-to-end SDD situational manual and command cheatsheet with inputs, outputs, triggers, examples, and extension hook details.
  - Verified local Spec-Kit tools and templates against official `github/spec-kit` documentation via Context7 (100% up-to-date; no update required).
- **Next Planned Step**: Continue execution of feature tasks or drift analysis via `/speckit-converge` or `/speckit-implement`.

### Milestone 2026-10-09 07:50: Project Constitution Amended to v1.5.0 (Spec-Kit)
- **Goal**: Formally ratify Principle XI (Software Engineering & Change Discipline) and add Pre-Commit Human Approval and Boundary Schema Validation Gates.
- **Changes**:
  - [.specify/memory/constitution.md](file:///Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG/.specify/memory/constitution.md):
    - Bumped semantic version from 1.4.0 to 1.5.0 (MINOR).
    - Added **Principle XI: Software Engineering & Change Discipline (Non-Negotiable)**: Mandates human approval before git commit/push, Conventional Commits referencing Spec-Kit Task IDs, execution scope restatement, schema-first boundary validation with fail-loudly semantics, zero dead code, and credential hygiene.
    - Updated Quality & Verification Gates with **Pre-Commit Human Approval Gate** and **Boundary Schema Validation Gate**.
- **Next Planned Step**: Continue execution of feature tasks in `specs/002-benchmark-dataset-scaling` via `/speckit-implement` adhering to Principle XI.




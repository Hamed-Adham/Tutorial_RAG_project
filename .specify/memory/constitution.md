<!--
SYNC IMPACT REPORT
- Version Change: 1.5.0 → 1.6.0 (MINOR: Added Principle XII: Enterprise Rigor & The Tool Trinity (Git, Spec-Kit, Langfuse))
- Modified Principles:
  - Added XII. Enterprise Rigor & The Tool Trinity (Git, Spec-Kit, Langfuse) (Non-Negotiable): Forbids superficial MVPs or fragile workarounds, mandates proactive questioning upon ambiguity, and establishes Git, Spec-Kit, and Langfuse as the non-negotiable operational trinity.
- Added Sections: None
- Removed Sections: None
- Follow-up TODOs: None
-->

# Enterprise RAG System Constitution

## Core Principles

### I. Three-Layer Decoupled Architecture
The system enforces a strict 3-layer decoupled separation:
1. `rag-core`: Framework-independent, pure Python package containing all RAG, ingestion, indexing, retrieval, synthesis, agent, evaluation, and security logic. It MUST NOT depend on FastAPI or Next.js and MUST remain directly importable into CLI tools, notebooks, or external workflows.
2. `backend`: FastAPI API server providing typed REST endpoints (`/api/v1`), streaming chat, authentication, session state, background tasks, and database persistence (PostgreSQL via SQLAlchemy & Alembic). It depends on `rag-core`.
3. `frontend`: Next.js 14+ App Router application with CopilotKit chat interface, document management, and evaluation dashboards. It depends strictly on backend API contracts.

### II. Evaluation-Driven Quality (Non-Negotiable)
Vibe-based testing is deprecated. Evaluation is the central pillar of engineering reliability and must be operational from Day 1:
1. RAGAS standard metrics must be tracked continuously:
   - Faithfulness >= 0.80 (groundedness; prevents hallucinations)
   - Answer Relevancy >= 0.75 (pertinence to query)
   - Context Precision >= 0.75 (relevance of retrieved chunks)
   - Context Recall >= 0.70 (coverage of necessary information)
2. Every major change to prompts, retrieval, or models must pass through the automated Evaluation Harness against a version-controlled Golden Test Set (minimum 20-50 curated Q&A pairs).
3. Quality gates in CI/CD block pull requests that degrade quality below established baselines.

### III. Granular Observability & Open Inference Standards
Every execution path must be comprehensively observable from ingestion to evaluation:
1. OpenTelemetry and OpenInference standards govern all traces, spans, and metadata attributes.
2. Langfuse Observability is mandatory and operational across all layers:
   - Every user query initiates a root trace capturing user ID, session ID, release version, and operational tags.
   - Distinct nested spans MUST wrap each discrete processing stage: document parsing, chunking, embedding generation, dense/sparse/graph retrieval, reranking, and synthesis.
   - LLM generation steps MUST log exact model name, input prompts, system instructions, raw completions, structured outputs, latency in seconds, and token usage (prompt, completion, total, and cost).
   - Evaluation verdicts, code gate results, and LLM-as-a-Judge scores MUST be attached directly as scores to corresponding Langfuse traces to enable metric tracking and regression monitoring over time.
   - Configuration MUST bind to standard Langfuse environment credentials (`LANGFUSE_BASE_URL` / `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`).
3. Structured JSON logging (via `structlog`) is mandatory across all components to ensure auditability, correlation IDs, and unified logging.
4. Tracing clients MUST operate with graceful offline fallbacks so local development and unit testing never fail if the Langfuse server is unreachable.

### IV. Independent Multi-Tool Agent Design
Agent orchestration via LangGraph StateGraph must preserve separation of concerns:
1. Multi-tool routing dynamically selects tools (Document RAG, GraphRAG, Web Search) based on user intent.
2. Tools are isolated as independent, single-responsibility wrappers.
3. Natural Language to SQL (Text-to-SQL) is explicitly descoped for the current project milestone to maintain razor focus on unstructured and graph retrieval.

### V. Defense-in-Depth Security & Data Privacy
Security and access control must be enforced at the data layer, not at the prompt layer:
1. Access Control Lists (ACL) are enforced via vector database payload filters (Qdrant), guaranteeing users can never retrieve documents beyond their clearance.
2. Personally Identifiable Information (PII) must be detected and anonymized before indexing and LLM prompt generation using Microsoft Presidio.
3. All operations, queries, and data modifications produce tamper-evident structured audit logs.

### VI. Performance Caching & Cost Efficiency
Cost and latency must be actively managed:
1. Content-hashed embedding cache prevents duplicate vector embedding calls.
2. Query/context hash TTL cache (Redis) eliminates duplicate LLM generation for identical queries.

### VII. Law of Parsimony & Component Maximization (Non-Negotiable)
Simplicity, determinism, and existing component maximization precede architectural expansion:
1. Do not deploy autonomous agents or multi-step iterative loops where deterministic code, simple heuristics, or direct retrieval solves the problem.
2. Never start with the most complex or advanced architecture. Always build the simplest Minimum Viable Product (Naive RAG: simple text extraction, fixed-size chunking, naive top-k dense retrieval, straightforward synthesis).
3. The Naive Baseline MUST be benchmarked against our formal evaluation suites first. This establishes our quantitative "Ground Zero" so every subsequent optimization is justified by empirical delta, not intuition.
4. **Component Maximization Precedes Component Accumulation**:
   Developers and agents MUST squeeze maximum performance out of existing components before introducing any new architectural component, model layer, or pipeline stage (e.g., cross-encoder rerankers, LLM query transformations, BM25 hybrid search, or GraphRAG).
   - Before adding a new component, developers must exhaustively tune existing parameters (e.g., chunk size/overlap, Qdrant HNSW `ef_search`, retrieval top-k, system prompt instructions, few-shot Persian examples).
   - A new component is only permitted if rigorous error analysis on benchmark failure modes proves existing components have hit an intrinsic performance ceiling, and the empirical score delta justifies the latency, token cost, and complexity overhead.

### VIII. Continuous Development Logging
Progress transparency and decision tracking are mandatory:
1. A root `development_log.md` MUST be created and maintained continuously.
2. At every milestone, architectural decision point, or feature completion, a structured summary of changes, rationale, evaluation metrics, and next steps must be recorded in `development_log.md`.

### IX. Workspace Structure & Environment Hygiene
Strict repository cleanliness:
1. Python dependencies must be isolated inside `.venv`.
2. Secrets, API endpoints, and configuration parameters must live in `.env` (with `.env.example` committed).
3. Workspace files and folders must be strictly organized at all times (no dangling scripts, temp files, or orphaned outputs in the root).

### X. Iterative Codebase Knowledge Graph Maintenance (Non-Negotiable)
The codebase's structural map and dependency graph MUST be maintained as a live, queryable Knowledge Graph (`.codebase-memory/graph.db.zst`) powered by the codebase-memory indexing engine:
1. **Authoritative Source of Code Structure**: Agents and developers must treat the Knowledge Graph as the authoritative living source of code structure, symbol references, cross-module call boundaries, and file relationships.
2. **Mandatory Trigger Moments for Graph Refresh**:
   - **Pre-Planning / Pre-Convergence**: Before generating feature plans (`/speckit-plan`) or evaluating convergence gaps (`/speckit-converge`), the agent MUST inspect the graph status (`detect_changes` / `get_architecture`) to ground design decisions in actual code reality.
   - **Post-Implementation Phase Checkpoints**: At the conclusion of every implementation phase or major task in `tasks.md`, the agent MUST invoke `index_repository(persistence=True)` to rebuild and compress the Knowledge Graph into `.codebase-memory/graph.db.zst`.
   - **Structural Reorganization & Refactoring**: Any file rename, relocation, module migration, or deletion immediately triggers a graph re-indexing before claiming milestone completion.
   - **Zero-Drift Invariant**: Prior to closing any PR or marking a SpecKit implementation complete, `detect_changes` MUST report zero modified or unindexed files (`changed_count: 0`).
3. **SpecKit Hook Integration**: The graph refresh lifecycle is automated via `.specify/extensions.yml` post-implementation and pre-planning hooks, ensuring no manual steps are missed.

### XI. Software Engineering & Change Discipline (Non-Negotiable)
Strict development workflow, commit conventions, and quality control discipline govern all implementation activities:
1. **Mandatory Human Approval Before Commit / Push**:
   - Automated agents and scripts MUST NEVER execute `git commit` or `git push` autonomously.
   - Upon completing a logical task, the agent MUST present:
     - A concise summary of changes and the architectural rationale ("why", not just "what").
     - Concrete test execution results (pass/fail summary and command output).
     - Diff summary or `git status` / `git diff`.
   - The agent MUST explicitly pause and wait for the developer's confirmation (`"commit it"`, `"looks good"`, `"تایید"`, etc.) before executing any commit.
2. **Conventional Commits with Spec-Kit Task Traceability**:
   - Commits MUST follow Conventional Commits format: `type(scope): short description`.
   - Permitted types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `build`, `ci`.
   - Commits MUST remain atomic and scoped to a single logical change; multiple roadmap tasks MUST NOT be bundled into a single commit.
   - Commit message bodies MUST explain the underlying rationale and reference the active feature specification and Task ID (e.g., `Spec: 002 | Task: T014`).
3. **Scope Restatement & Zero Dead Code**:
   - Before modifying or creating code files, the agent MUST restate in 1–2 sentences what the task entails and explicitly list the files expected to be touched.
   - Commented-out legacy code implementations ("just-in-case" blocks) are strictly forbidden. Git history is the permanent safety net.
4. **Schema-First Input Validation at the System Boundary**:
   - All external inputs, API parameters, dataset records, and evaluation metrics MUST be parsed and validated through strict Pydantic v2 schemas before use. Handlers and evaluation modules MUST consume only validated schema fields (`model.field`), never raw untrusted dictionaries.
   - Fail Loudly: Systems MUST raise typed exceptions or return explicit validation rejections rather than silently masking missing data or returning misleading empty outputs.
5. **Zero-Credential Policy**:
   - Secrets, API credentials, private tokens, and environment values MUST NEVER be included in commits. `.env`, `.pem`, and credential files MUST be verified against `.gitignore` prior to staging.

### XII. Enterprise Rigor & The Tool Trinity (Git, Spec-Kit, Langfuse) (Non-Negotiable)
This project is engineered to enterprise standards; superficial "MVPs", fragile shortcuts, or unreviewed workarounds are strictly forbidden:
1. **No Shortcuts & Zero Unreviewed Hacks**: Every abstraction, schema, pipeline stage, and test suite must be built with production-grade robustness. Never deploy placeholder stubs, superficial mockups, or unvetted workarounds.
2. **Proactive Clarification Over Ungrounded Assumption**: Whenever ambiguity, conflicting constraints, or technical hurdles arise, agents and engineers MUST NOT improvise silent workarounds. They must stop and proactively ask targeted questions to verify intent before proceeding.
3. **The Mandatory Tool Trinity**:
   The primary developmental objective across all phases is to fully experience, master, and operate through the integrated tool trinity:
   - **Git Workflow**: Local branches, atomic Conventional Commits referencing Spec-Kit Task IDs, clean history, zero secret leaks, and mandatory human review before commits.
   - **Spec-Kit Workflow**: Rigorous execution of the complete Spec-Driven Development (SDD) lifecycle without bypassing governance: Constitution -> Specify -> Clarify -> Plan -> Checklist -> Tasks -> Implement -> Converge.
   - **Langfuse Workflow**: Exhaustive LLM observability, hierarchical trace/span instrumentation, dataset management with sealed splits, and quantitative scoring via automated harnesses.

## Phased Implementation Standard

Engineering deliverables must follow an incremental hill-climbing lifecycle:
- **Phase 0**: Project setup, environment isolation (.venv, .env), configuration schemas.
- **Phase 0.5 (Naive Baseline)**: Minimalist Naive RAG pipeline (simple PDF/CSV parsing, fixed chunking, dense retrieval, basic synthesis) evaluated on golden benchmark to set Ground Zero metrics.
- **Phase 1 (MVP)**: Ingestion (tiered parsing, structure-aware chunking), BGE-M3 embeddings, Qdrant vector retrieval, baseline generation, RAGAS evaluation, Langfuse tracing.
- **Phase 2 (Optimization via Component Maximization)**: First tune dense retrieval + chunking + prompt engineering to their ceiling; only add hybrid search (BM25 + Hazm Persian NLP) and cross-encoder reranking where empirical gaps remain.
- **Phase 3 (Agent & GraphRAG)**: LangGraph StateGraph orchestrator, LangChain tool wrappers, GraphRAG entity expansion, Web search tool (Text-to-SQL descoped).
- **Phase 4 (Enterprise)**: Presidio PII, Qdrant ACL, Redis caching, FastAPI service, Next.js CopilotKit UI, Docker & Kubernetes packaging, Alembic migrations.

## Quality & Verification Gates

1. **Pre-Commit Human Approval Gate**: Mandatory verification of test results, diff inspection, and explicit human confirmation before executing any `git commit` or `git push`.
2. **Boundary Schema Validation Gate**: All data entering or exiting core evaluation/inference pipelines must adhere to validated Pydantic schemas. Handlers accessing raw unvalidated payloads are rejected.
3. **Quality Gate**: RAGAS evaluation on PR for RAG-impacting code changes (`rag-eval.yml`). Merges are blocked if any core metric drops below threshold or baseline.
4. **Drift Gate**: Scheduled weekly evaluation (`scheduled-eval.yml`) against the golden dataset to detect model drift or data staleness.
5. **Codebase Memory Graph Gate**: Mandatory verification that `.codebase-memory/graph.db.zst` is refreshed (`index_repository(persistence=True)`) and `detect_changes` confirms 0 unindexed files (`changed_count: 0`) before marking any milestone or task complete.

## Governance

1. This Constitution is the authoritative architectural contract for the Enterprise RAG codebase.
2. Any architectural deviations, new dependencies, or metric threshold changes require explicit justification, documentation, and baseline updates.
3. Code reviews must verify compliance with the 3-layer decoupling, security boundaries, and evaluation coverage.

**Version**: 1.6.0 | **Ratified**: 2026-10-01 | **Last Amended**: 2026-10-09 | **Status**: Active

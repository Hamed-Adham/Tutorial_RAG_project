# Workspace Rules: Unified RAG Project

## 1. Development Log Maintenance (Mandatory)
- A root `development_log.md` MUST be updated at every logical milestone or significant progress point.
- Each log entry must concisely summarize:
  - Timestamp & milestone goal
  - Changes made (files created, modified, or removed)
  - Architectural decisions & rationale
  - Evaluation or verification metrics achieved
  - Next planned step

## 2. Law of Parsimony & Baseline-First Engineering
- Do not introduce autonomous multi-step agents or complex multi-hop architectures when a deterministic code solution or simple retrieval heuristic suffices.
- Avoid jumping to the most complex tools from the outset.
- Always implement the simplest baseline MVP first (Naive RAG: simple text extraction, fixed-size chunking, naive top-k dense retrieval, direct LLM generation) and benchmark it against our evaluation suite to establish the empirical "Ground Zero".
- Every subsequent advancement (hybrid search, RRF, reranking, query transforms, agent routing) must be validated by measurable metric improvements over this baseline.

## 3. Workspace Hygiene & File Organization
- Keep dependencies strictly isolated inside `.venv`.
- Configuration and API credentials must reside in `.env`.
- Keep the workspace root clean and tidy. No dangling scripts, temporary scratch files, or misplaced assets. Organise files into their respective directories: `rag-core/`, `data/`, `specs/`, etc.

## 4. Formula & Math Formatting
- NEVER output raw LaTeX syntax (`$ ... $`, `$$ ... $$`, `\frac`, etc.) in chat or markdown.
- ALWAYS use clean plain text, unicode, or markdown code blocks (e.g. `Precision = TP / (TP + FP)`).

## 5. Spec-Driven Development (Spec-Kit) Guidance (Mandatory)
- In every response, proactively analyze the user's current intent and guide them toward the appropriate Spec-Kit command/skill to build their Spec-Driven Development (SDD) habits.
- Clearly advise which command corresponds to their current stage in the lifecycle:
  - `/speckit-constitution`: Define or update fundamental project principles, governance, and architecture standards.
  - `/speckit-specify`: Create or update a feature specification (`spec.md`) from a natural language requirement or idea.
  - `/speckit-clarify`: Identify gaps or ambiguities in `spec.md` through targeted questions.
  - `/speckit-checklist`: Generate completeness, edge-case, and validation checklists.
  - `/speckit-plan`: Design implementation plans, data models, and architectural decisions (`plan.md`).
  - `/speckit-tasks`: Decompose the plan into dependency-ordered, actionable execution units (`tasks.md`).
  - `/speckit-analyze`: Verify consistency and quality across `spec.md`, `plan.md`, and `tasks.md`.
  - `/speckit-taskstoissues`: Convert task breakdown into tracked GitHub issues.
  - `/speckit-converge`: Audit codebase against spec/plan/tasks and append missing implementation tasks.
  - `/speckit-implement`: Execute code implementation following `tasks.md`.
- Conclude or frame every interaction with a clear Spec-Kit recommendation based on where the work currently stands.

## 6. Enterprise Professionalism, No Shortcuts & The Tool Trinity (Mandatory)
- **High Rigor & No Fragile Workarounds**: This project is super professional and production-grade. Do NOT build superficial "MVPs", cut corners, or deploy fragile hacks. Build robust, well-tested, and maintainable software.
- **Ask Questions When Stuck**: If you face ambiguity, conflicting requirements, or technical blockers, NEVER improvise silent workarounds or make ungrounded assumptions. Stop and proactively ask the user clear, targeted questions to resolve the issue.
- **The Core Development Trinity (Git + Spec-Kit + Langfuse)**:
  The explicit objective of the development phase is to fully experience and strictly execute the project using these three workflows:
  1. **Git Workflow**: Local feature branches, atomic Conventional Commits referencing Spec-Kit Task IDs, clean git log history, and mandatory human approval before any commit or push.
  2. **Spec-Kit Workflow (Spec-Driven Development)**: Full lifecycle discipline without skipping steps: Constitution -> Specify -> Clarify -> Plan -> Checklist -> Tasks -> Implement -> Converge.
  3. **Langfuse Workflow**: Rigorous LLM observability, trace/span hierarchy (latency, tokens, costs), dataset curation (dev/test stratified splits), and automated evaluation scoring.

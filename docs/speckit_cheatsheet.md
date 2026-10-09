# Spec-Kit Cheatsheet: Spec-Driven Development (SDD) Guide

This guide explains **how**, **when**, and **why** to use each Spec-Kit slash command. Spec-Driven Development (SDD) transforms freeform feature requests into structured, executable specifications, plans, and atomic implementation tasks before writing code.

---

## 1. The Core Lifecycle Pipeline

Spec-Kit follows a structured delivery pipeline:

```text
/speckit-constitution  (Project governance & architectural rules)
         │
         ▼
/speckit-specify       (Define requirements & user stories in spec.md)
         │
         ▼
/speckit-clarify       [Quality Gate] (Identify underspecified gaps)
         │
         ▼
/speckit-plan          (Technical design, data models, contracts in plan.md)
         │
         ▼
/speckit-checklist     [Quality Gate] (Acceptance & edge-case criteria)
         │
         ▼
/speckit-tasks         (Actionable, dependency-ordered tasks in tasks.md)
         │
         ▼
/speckit-analyze       [Quality Gate] (Cross-artifact consistency validation)
         │
         ▼
/speckit-implement     (Code generation & unit test execution)
         │
         ▼
/speckit-converge      (Codebase vs spec drift check; loop until converged)
```

---

## 2. Command Reference & Situational Guide

### 🏛️ `/speckit-constitution`
* **When to use**: 
  - Starting a new project or updating core engineering principles.
  - Adding non-negotiable architectural boundaries (e.g., "Always build baseline MVP before agentic routing", "No raw LaTeX", "Isolate dependencies").
* **Inputs**: Freeform rules or answers to prompt questions.
* **Output**: `.specify/memory/constitution.md`
* **Example**:
  ```text
  /speckit-constitution Prioritize baseline-first empirical evaluation and strict workspace hygiene.
  ```

---

### 📝 `/speckit-specify`
* **When to use**:
  - You have a new feature, capability, or user-facing requirement to build.
  - You need to create a new spec directory (`specs/XXX-feature-name/spec.md`).
  - You want to define *what* needs to be built and *why* (without implementation details).
* **Inputs**: Natural language feature description.
* **Output**: `specs/<feature>/spec.md` + `.specify/feature.json` updated to point to the active feature.
* **Example**:
  ```text
  /speckit-specify Add automated citation verification metric to rag-core evaluation suite.
  ```

---

### 🔍 `/speckit-clarify` *(Quality Gate)*
* **When to use**:
  - Right after `/speckit-specify` or before `/speckit-plan`.
  - The feature has ambiguous edge cases, undefined constraints, or unknown business rules.
  - You want the agent to ask you up to 5 targeted clarifying questions and encode the answers directly back into `spec.md`.
* **Inputs**: Existing `spec.md` (optional focus area argument).
* **Output**: Refined `spec.md` with resolved ambiguities.
* **Example**:
  ```text
  /speckit-clarify Focus on failure handling when the LLM hallucinating citation indices.
  ```

---

### 📐 `/speckit-plan`
* **When to use**:
  - The feature spec is clear and you need to design *how* to build it.
  - Establishing architectural patterns, research findings (`research.md`), data contracts (`data-model.md`), and technology choices.
* **Inputs**: `spec.md` + active codebase state.
* **Output**: `specs/<feature>/plan.md`, `research.md`, `data-model.md`, `contracts/`.
* **Example**:
  ```text
  /speckit-plan Use Pydantic models for metric outputs and integrate with Langfuse dataset runs.
  ```

---

### ✅ `/speckit-checklist` *(Quality Gate)*
* **When to use**:
  - You want to generate rigorous validation checklists for requirements, security, performance, or edge cases before starting tasks.
* **Inputs**: `spec.md`, `plan.md`.
* **Output**: `specs/<feature>/checklists/requirements.md`
* **Example**:
  ```text
  /speckit-checklist Create acceptance and edge-case checklists for dataset scaling.
  ```

---

### 📋 `/speckit-tasks`
* **When to use**:
  - Architecture plan is finalized (`plan.md` exists) and you need actionable tasks.
  - Breaks work down into Phase 1 (Setup), Phase 2 (Foundational Prerequisites), Phase 3+ (User Stories ordered by priority), and Phase N (Polish/Finalize).
  - Marks tasks that can be parallelized (`[P]`).
* **Inputs**: `spec.md`, `plan.md`, `data-model.md`.
* **Output**: `specs/<feature>/tasks.md`
* **Example**:
  ```text
  /speckit-tasks
  ```

---

### 🔬 `/speckit-analyze` *(Quality Gate)*
* **When to use**:
  - Before starting code implementation.
  - Non-destructive sanity check: ensures `tasks.md` actually implements everything promised in `spec.md` and designed in `plan.md` without gaps or hallucinations.
* **Inputs**: `spec.md`, `plan.md`, `tasks.md`.
* **Output**: Consistency analysis report with warnings and suggested task additions.
* **Example**:
  ```text
  /speckit-analyze
  ```

---

### 🎫 `/speckit-taskstoissues` *(Optional)*
* **When to use**:
  - You want to sync your local `tasks.md` into GitHub Issues for team tracking or project boards.
* **Inputs**: `tasks.md` + GitHub origin remote.
* **Output**: Created GitHub Issues with labels and task IDs.
* **Example**:
  ```text
  /speckit-taskstoissues
  ```

---

### 🛠️ `/speckit-implement`
* **When to use**:
  - You are ready to write code, create tests, and fulfill the tasks in `tasks.md`.
  - Executes phase-by-phase or task-by-task, executing tests as each task finishes.
* **Inputs**: `tasks.md`, `plan.md`, `spec.md`.
* **Output**: Working code, passing unit tests, checked-off boxes `[X]` in `tasks.md`.
* **Example**:
  ```text
  /speckit-implement
  ```

---

### 🔁 `/speckit-converge`
* **When to use**:
  - Run immediately after `/speckit-implement` finishes a phase or the entire feature.
  - Evaluates actual code vs `spec.md`/`tasks.md` to confirm nothing was missed.
  - **Append-only**: Never modifies or deletes code. If gaps are found, appends them to a `Convergence` section in `tasks.md`.
  - **The Convergence Loop**: Repeat `implement -> converge -> implement -> converge` until it reports `Converged`.
* **Inputs**: Codebase state, `spec.md`, `plan.md`, `tasks.md`.
* **Output**: Clean status (`✅ Converged`) or new appended tasks in `tasks.md`.
* **Example**:
  ```text
  /speckit-converge
  ```

---

## 3. Quick Decision Matrix: "What Should I Run Right Now?"

| Your Current Situation / Intent | Run This Command |
| :--- | :--- |
| "I have a new idea / need a new feature from scratch" | `/speckit-specify <describe idea>` |
| "The spec is vague / I don't know all edge cases yet" | `/speckit-clarify` |
| "The spec is done, now I need to choose libraries and architecture" | `/speckit-plan <technical constraints>` |
| "I want to be sure our tests and requirements are fully covered" | `/speckit-checklist` |
| "The technical plan is ready, now turn it into task check-boxes" | `/speckit-tasks` |
| "Are my tasks, plan, and spec consistent with each other?" | `/speckit-analyze` |
| "Time to write the code and pass the tests" | `/speckit-implement` |
| "Did we actually build everything promised in the spec?" | `/speckit-converge` |
| "Convergence found missing items in tasks.md" | `/speckit-implement` (to execute the new tasks) |
| "We need to set baseline project rules and architecture standards" | `/speckit-constitution` |

---

## 4. Local Repository Integrations (`.specify/extensions.yml`)

In this repository, Spec-Kit is augmented with lifecycle hooks for the **Codebase Memory Knowledge Graph** (`codebase-memory-mcp`):
1. **Before Plan**: Checks codebase graph status and index drift.
2. **After Implement**: Automatically triggers re-indexing to keep `.codebase-memory/graph.db.zst` synchronized.
3. **Before Converge**: Queries the knowledge graph architecture to check existing vs missing modules.
4. **After Converge**: Re-indexes the updated codebase.

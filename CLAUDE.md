# Unified RAG — Claude Code Entry Point

Project-wide rules live in a single source of truth shared with Antigravity:

@.agents/rules/agent-rules.md

Imported above rather than duplicated so edits in one tool apply to both.

## Spec-Kit

Spec-Kit v1.0.8.dev0 is initialized in this project with both `agy` and `claude`
integrations. Workflow skills are available in both tools:

- `.agents/skills/speckit-*` — Antigravity
- `.claude/skills/speckit-*` — Claude Code

Active feature is tracked in `.specify/feature.json`
(current: `specs/003-enterprise-science-pipeline`).

## Skills

Custom skills live in `.agents/skills/` (git-tracked) and are bridged into
`.claude/skills/` so both tools resolve the same files:

- `langfuse` — tracing, evals, prompt management (see `docker-compose.langfuse.yml`)
- `codebase-memory` — canonical module map, single source of truth for layout
- `typesafe-ai` — typed LLM decision primitives

## Conventions

Follow `.agents/rules/agent-rules.md`: maintain `development_log.md`, keep a
deterministic baseline before adding complexity, isolate deps in `.venv`,
credentials in `.env`, and never emit raw LaTeX.
# Implementation Plan: QueryGate & Conversational Memory Routing (006)

**Branch**: `006-query-gate-conversational-memory` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from [specs/006-query-gate-conversational-memory/spec.md](spec.md)

---

## Summary

Deliver a production-grade, token-efficient query understanding and conversational memory routing system for the Iranian 7th-grade science textbook RAG assistant:
1. **QueryGate Renaming & Multi-Tier Classification**: Rename `gated_rewriter` -> `query_gate` across the entire codebase. Implement 3-tier curriculum intent gating (`completely_irrelevant`, `partially_relevant`, `relevant`) with upfront query purification for mixed queries, preventing vector retrieval poisoning and adversarial prompt injection.
2. **Context Need Dispatching**: Classify queries into 4 execution tiers (`needs_history_and_retrieve`, `needs_history_only`, `needs_retrieve_only`, `needs_neither`). For `needs_history_only` (meta-questions, rephrasings, summaries), bypass dense/sparse retrieval and reranking entirely, cutting latency by 70%+ (<600ms) and attaching a dedicated conversational context badge.
3. **SubQ Decision Model 2 with 10-10 Parallel Batching**: Integrate SubQ Decisions API (`POST https://api.subq.ai/v1/systemone`, model `subq-snap-preview`) to evaluate candidate turns from the last 20 messages. Split candidate turns into two concurrent requests of up to 10 questions each via `asyncio.gather` to respect SubQ's 16-question limit.
4. **Fidelity-Scaled Context Selection**: For turns classified as needed, inject **full Q&A text** for turns within the 5 most recent turns, and **concise summaries** for older turns (turns 6 through 20) with no artificial numerical limit.
5. **Deterministic Backfill & Incremental Summaries**: Deterministically inspect session turns for missing summaries and invoke a single batch LLM call to backfill them with a 2.0s fallback guard. In response synthesis, incrementally generate turn summaries (<= 512 tokens / ~6 lines) and cache them in session storage.
6. **Qwen-Style UI Reply & Context Selection**: On each chat message card, provide a "Reply / Select as Context" button with a dismissible floating preview chip above the input box, allowing students to manually attach at most one past turn when history is toggled OFF.

---

## Technical Context

**Language/Version**: Python 3.11 (`rag-core`, `backend`), TypeScript 5.6 / Node.js 20+ (`frontend`)  
**Primary Dependencies**:
* `rag-core`: `langgraph>=0.2.0`, `langchain-core>=0.3.0`, `httpx>=0.27.0`, `pydantic>=2.9.0`, `qdrant-client>=1.11.0`, `rank-bm25>=0.2.2`, `hazm>=0.10.0`
* `backend`: `fastapi>=0.115.0`, `uvicorn>=0.30.0`, `redis>=5.0.0`, `sse-starlette>=2.0.0`
* `external APIs`: SubQ Decisions API (`https://api.subq.ai/v1/systemone`, `subq-snap-preview`, 384K context), Localhost Nine Router gateway / Jev System One  
* `frontend`: `next@14.2.15`, `react@18.3.1`, `tailwindcss@3.4.1`, `lucide-react@0.446.0`  
**Storage**: Redis 7.0+ / `InMemorySessionStore` with metadata dictionary supporting `q_and_a_summary` and turn indexing.  
**Testing**: `pytest` (`rag-core/tests/`, `backend/tests/`), Next.js `npm run build`.  
**Target Platform**: Linux / macOS ASGI server + Next.js web application.  
**Project Type**: Decoupled 3-layer architecture (`rag-core` -> `backend` -> `frontend`).  
**Performance Goals**:
* Gate 1 (`QueryGate`) evaluation latency < 25ms.
* Decision Model 2 evaluation latency < 350ms across candidate turns.
* `needs_history_only` end-to-end response latency < 600ms (70%+ drop from ~1.8s).
* 100% of out-of-scope adversarial/chitchat inquiries intercepted with 0 retrieval and 0 synthesis tokens.  
**Constraints**:
* No raw LaTeX formatting in responses or markdown output.
* SubQ 16-question limit per request strictly respected via 10-10 chunking.
* 100% Persian RTL compatibility with Peyda font typography.  
**Scale/Scope**: 15 chapters (412 sections) of Iranian 7th-grade science textbook; 20-turn conversational history window; 50 concurrent sessions.

---

## Constitution Check

*GATE: All checks verified prior to implementation.*

* [x] **Principle I: Three-Layer Decoupled Architecture**: `rag-core` (pure Python library) houses `QueryGate` and SubQ decision client; `backend` manages session storage and endpoints; `frontend` handles Qwen-style UI quote chip.
* [x] **Principle II: Evaluation-Driven Quality**: Unit and contract tests defined for all 6 user stories in `quickstart.md` and `contracts/query_gate_contracts.py`.
* [x] **Principle III: Granular Observability**: Langfuse traces instrumented for Gate 1 decision, SubQ Decision Model 2 evaluation, and synthesis turn summary.
* [x] **Principle IV: Independent Multi-Tool Agent Design**: `EnterpriseScienceGraph` orchestrates conditional branching (`query_gate` -> `refusal` | `hybrid_retrieve` | `synthesis`).
* [x] **Principle V: Defense-in-Depth Security**: Completely irrelevant and adversarial inquiries intercepted at Gate 1 before any database or LLM invocations.
* [x] **Principle VI: Performance Caching & Cost Efficiency**: Retrieval bypass for Tier 2 saves dense vector calls and reranker compute; incremental summary caching eliminates repeated backfill calls.
* [x] **Principle VII: Law of Parsimony & Component Maximization**: Maximizes existing Jev and SubQ System One decision primitives; avoids heavy agentic loops.
* [x] **Principle VIII: Continuous Development Logging**: Milestone recorded in `development_log.md`.
* [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance**: `codebase-memory-mcp` checked before planning; graph refresh checkpoint scheduled.
* [x] **Principle XI: Software Engineering & Change Discipline**: Human confirmation required before git commits; typed Pydantic v2 schemas at all boundaries.
* [x] **Principle XII: Enterprise Rigor & The Tool Trinity**: Full integration across Git, Spec-Kit, and Langfuse.

---

## Project Structure

### Documentation (this feature)

```text
specs/006-query-gate-conversational-memory/
├── spec.md              # Feature specification (100% validated)
├── plan.md              # This implementation plan
├── research.md          # Technical decisions and rationale
├── data-model.md        # Entities, Enums, and Pydantic contracts
├── quickstart.md        # Step-by-step verification guide
├── contracts/
│   ├── query_gate_contracts.py  # Python protocols and models
│   ├── subq_decision_schema.json # JSON schema for SubQ Decisions API
│   └── api_contracts.md         # REST & SSE streaming contracts
└── checklists/
    └── requirements.md  # Quality checklist (16/16 passed)
```

### Source Code Layout

```text
rag-core/
├── src/rag_core/
│   ├── query/
│   │   ├── query_gate.py         # Renamed & upgraded QueryGate (3-tier intent + 4-tier dispatch)
│   │   ├── subq_client.py        # SubQ Decisions API client (model: subq-snap-preview, 10-10 chunking)
│   │   ├── summarizer.py         # Structured batch summary backfill & incremental turn summarizer
│   │   └── __init__.py           # Exports QueryGate, SubQClient, Summarizer
│   ├── graph/
│   │   ├── state.py              # Extended RAGGraphState (query_gate_decision, selected_history_items, etc.)
│   │   ├── nodes.py              # create_enterprise_query_gate_node, updated refusal and synthesis nodes
│   │   └── pipeline_graph.py     # Graph wiring with conditional bypass edge to synthesis
│   ├── generation/
│   │   ├── synthesizer.py        # Incremental turn summary generation under q_and_a_summary
│   │   └── prompts.py            # Prompts for partial query purification and batch summarization
│   └── indexing/
│       └── science_schema.py     # Data contracts for QueryGateDecision and SelectedHistoryItem
└── tests/
    └── test_query_gate.py        # Comprehensive test suite for QueryGate, SubQ, and backfill

backend/
├── app/
│   ├── schemas/
│   │   └── chat.py               # Updated ChatRequest (selected_context_turn_id) and ChatResponse
│   └── services/
│       ├── session_store.py      # Session store persistence for q_and_a_summary in turn metadata
│       └── rag_service.py        # Bridge handling manual context injection and history-only badging
└── tests/
    └── test_chat_endpoints.py    # Updated API endpoint tests

frontend/
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx     # Floating dismissible quote chip above textarea
│   │   ├── MessageItem.tsx       # "Reply / Select as context" action button on message cards
│   │   └── Badge.tsx             # Conversational context badge ("پاسخ بر اساس گفتگوی پیشین")
│   └── types/
│       └── chat.ts               # TypeScript interfaces for SelectedContext and QueryGateDecision
```

---

## Complexity Tracking

| Aspect | Architectural Rationale | Simpler Alternative Rejected Because |
|---|---|---|
| **SubQ 10-10 Parallel Batching** | Candidate window is 20 turns, while SubQ enforces a 16-question limit per request. Splitting into 10-10 executed via `asyncio.gather` complies with API limits without sequential latency. | A single 20-question call fails SubQ API validation with a `400` error. Sequential calls take >3.0s. |
| **Upfront Query Purification** | Mixed queries (e.g. *"امروز بارونیه ولی چگالی چوب چقدره؟"*) must be cleaned before Decision Model 2 and Retrieval. | Passing raw mixed queries contaminates dense vector search and confuses SubQ historical relevance scoring. |
| **Qwen-Style Context Attachment** | Allows users to manually attach one prior Q&A when automatic history is toggled OFF. | Without manual selection, users in private/stateless mode cannot resolve even simple isolated follow-up questions. |

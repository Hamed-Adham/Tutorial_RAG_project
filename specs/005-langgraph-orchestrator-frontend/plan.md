# Implementation Plan: LangGraph Orchestrator & Modern Next.js Chatbot UI

**Branch**: `005-langgraph-orchestrator-frontend` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from [specs/005-langgraph-orchestrator-frontend/spec.md](spec.md)

---

## Summary

Deliver a production-grade educational chatbot user interface and declarative graph orchestrator for the Iranian 7th-grade science textbook RAG system:
1. **Frontend**: Next.js 14 App Router conversational UI inspired by McKay Wrigley's `chatbot-ui`, featuring native Persian RTL alignment, self-hosted **Peyda (پیدا)** typography, collapsible session sidebar with LocalStorage caching, chapter scope selection (Mode 1 Scoped Practice vs. Mode 2 Global Search), evidence drawer with textbook citations, latency waterfall badge, and live Prometheus telemetry dashboard.
2. **Orchestrator**: LangGraph `StateGraph` (`EnterpriseScienceGraph`) standardizing on LangChain Document and BaseRetriever protocols, featuring conditional refusal branching (`should_enterprise_refuse`), streaming tokens via Server-Sent Events (SSE), and native Langfuse `CallbackHandler` telemetry.
3. **Backend Bridge**: High-concurrency FastAPI gateway (`ConcurrencyGateway`) with `asyncio.Semaphore(50)` rate-guard and Redis session memory.

---

## Technical Context

**Language/Version**: Python 3.11 (`rag-core`, `backend`), TypeScript 5.6 / Node.js 20+ (`frontend`)  
**Primary Dependencies**:
* `rag-core`: `langgraph>=0.2.0`, `langchain-core>=0.3.0`, `langfuse>=2.50.0`, `qdrant-client>=1.11.0`, `rank-bm25>=0.2.2`, `hazm>=0.10.0`
* `backend`: `fastapi>=0.115.0`, `uvicorn>=0.30.0`, `redis>=5.0.0`, `sse-starlette>=2.0.0`, `pydantic>=2.9.0`
* `frontend`: `next@14.2.15`, `react@18.3.1`, `tailwindcss@3.4.1`, `lucide-react@0.446.0`  
**Storage**: Redis 7.0+ for multi-turn session memory (4-turn sliding window, 2h TTL); browser `LocalStorage` for client session caching; Qdrant for dense HNSW embeddings.  
**Testing**: `pytest` (`backend/tests/`, `rag-core/tests/`), Next.js production build (`next build`).  
**Target Platform**: Node.js web server / edge-compatible UI + ASGI Python 3.11 service.  
**Project Type**: Decoupled fullstack application (3-tier architecture).  
**Performance Goals**: Time-To-First-Token (`TTFT`) < 1.0s; p95 end-to-end latency < 2.5s; 50 concurrent sessions supported without connection exhaustion.  
**Constraints**: 100% Persian RTL compatibility with Peyda font typography; zero-token early refusal on out-of-scope inquiries.  
**Scale/Scope**: 15 chapters (412 sections) of Iranian 7th-grade science textbook; 50 concurrent users.

---

## Constitution Check

*GATE: All checks verified prior to implementation.*

* [x] **Principle I: Three-Layer Decoupled Architecture**: `rag-core` (pure Python library) → `backend` (FastAPI REST service) → `frontend` (Next.js App Router). Cleanly isolated.
* [x] **Principle II: Evaluation-Driven Quality**: 115 passing tests (97 `rag-core` + 18 `backend`).
* [x] **Principle III: Granular Observability**: Langfuse OpenInference `CallbackHandler` traces graph node executions and attaches metrics.
* [x] **Principle IV: Independent Multi-Tool Agent Design**: `EnterpriseScienceGraph` built as a modular LangGraph `StateGraph`.
* [x] **Principle V: Defense-in-Depth Security**: Deterministic refusal branching halts out-of-scope inquiries before search execution.
* [x] **Principle VI: Performance Caching & Cost Efficiency**: Redis session memory + early refusal bypass saves LLM and vector tokens.
* [x] **Principle VII: Law of Parsimony & Component Maximization**: Reuses existing BM25, Qdrant, and CrossEncoder components without unnecessary complexity.
* [x] **Principle VIII: Continuous Development Logging**: Milestone 23 and Milestone 24 recorded in `development_log.md`.
* [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance**: `detect_changes` verified before planning.
* [x] **Principle XI: Software Engineering & Change Discipline**: Human confirmation required before git commits.
* [x] **Principle XII: Enterprise Rigor & The Tool Trinity**: Full integration across Git, Spec-Kit, and Langfuse.

---

## Project Structure

### Documentation (this feature)

```text
specs/005-langgraph-orchestrator-frontend/
├── spec.md              # Feature specification (approved)
├── plan.md              # This implementation plan
├── research.md          # Technical decisions and rationale
├── data-model.md        # Entities, TypeScript interfaces, and graph state
├── quickstart.md        # Verification and validation guide
├── contracts/
│   ├── api_contracts.md # OpenAPI and SSE streaming contracts
│   └── graph_contracts.py # Python protocols for LangGraph and LangChain Document
└── checklists/
    └── requirements.md  # Quality checklist (100% PASS)
```

### Source Code Layout

```text
rag-core/
├── src/rag_core/
│   ├── graph/
│   │   ├── state.py           # RAGGraphState TypedDict
│   │   ├── nodes.py           # Modular node factories (gated rewrite, refusal, hybrid retrieve, rerank, expansion, synthesis)
│   │   └── pipeline_graph.py  # EnterpriseScienceGraph StateGraph
│   ├── indexing/
│   │   └── science_schema.py  # RetrievalCandidate.to_langchain_document()
│   ├── retrieval/
│   │   └── hybrid.py          # HybridRetriever.retrieve_documents()
│   └── pipeline.py            # AdvancedRAGPipeline.as_langgraph_pipeline() bridge
└── tests/
    └── test_enterprise_graph.py # Unit tests for StateGraph and LangChain protocols

backend/
├── app/
│   ├── core/
│   │   ├── gateway.py         # ConcurrencyGateway (asyncio.Semaphore(50))
│   │   └── telemetry.py       # OperationalMetricsTracker
│   ├── services/
│   │   ├── session_store.py   # RedisSessionStore + InMemorySessionStore
│   │   └── rag_service.py     # RAGService bridge supporting streaming and batch
│   └── api/v1/
│       └── endpoints/
│           ├── chat.py        # POST /api/v1/chat and POST /api/v1/chat/stream
│           ├── metrics.py     # GET /api/v1/metrics
│           └── health.py      # GET /api/v1/health
└── tests/
    ├── test_chat_endpoints.py
    ├── test_gateway_concurrency.py
    └── test_session_store.py

frontend/
├── public/
│   └── fonts/                 # Self-hosted Peyda WOFF2 web fonts (Regular, Medium, Bold)
├── src/
│   ├── app/
│   │   ├── globals.css        # Tailwind base + @font-face rules for Peyda
│   │   ├── layout.tsx         # Root layout with Persian RTL dir="rtl"
│   │   └── page.tsx           # Chat page with SSE streaming, session state, modals
│   ├── components/
│   │   ├── Sidebar.tsx        # Session drawer with New Chat, search, rename, delete
│   │   ├── ChatHeader.tsx     # Navigation header with Chapter Scope and Telemetry triggers
│   │   ├── ChatMessage.tsx    # Message bubble, citations drawer, latency waterfall pill
│   │   ├── ChatInput.tsx      # Auto-resizing textarea with curriculum suggestion prompts
│   │   ├── ChapterModal.tsx   # 15-chapter scope practice selector modal
│   │   └── MetricsModal.tsx   # Live operational telemetry dashboard
│   ├── lib/
│   │   └── constants.ts       # Chapter list, quick prompts, Iranian curriculum metadata
│   └── types/
│       └── index.ts           # TypeScript interfaces (Session, Message, Citation, Metrics)
├── next.config.js             # API rewrites proxying /api/v1/:path* to backend
└── tailwind.config.js         # Typography and styling tokens with Peyda font
```

---

## Complexity Tracking

> *No constitutional violations. Design adheres strictly to parsimony and component maximization.*

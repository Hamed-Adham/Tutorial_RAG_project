# Implementation Plan: Concurrency Gateway, Session Memory & Production Telemetry

**Branch**: `004-concurrency-gateway-sessions` | **Date**: 2026-10-13 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/004-concurrency-gateway-sessions/spec.md` defining high-concurrency 50-user load resilience, Redis multi-turn session memory, Server-Sent Events (SSE) streaming token delivery, and live operational health telemetry (Locust load tests, queue wait tracking, circuit breaker trip rate).

---

## Summary

Build and validate the **Enterprise FastAPI Concurrency Gateway & Session Management Layer** (`backend/app/`), wrapping our verified `AdvancedRAGPipeline` core into an asynchronous, production-grade service capable of serving 50 simultaneous users. The system provides an asynchronous event loop with an `asyncio.Semaphore(50)` rate-guard, Redis-backed multi-turn dialogue memory with a 4-turn sliding window and 2-hour TTL expiration, SSE streaming token delivery with Time-To-First-Token (`TTFT`) under 1.0 second, live operational telemetry (`/api/v1/metrics`, `/api/v1/health`), and an automated Locust concurrency benchmark simulating 50 virtual users.

---

## Technical Context

**Language/Version**: Python 3.11 / 3.12 (isolated in `.venv`)  
**Primary Dependencies**: `fastapi>=0.111.0`, `uvicorn[standard]>=0.29.0`, `redis>=5.0.0`, `pydantic>=2.7.0`, `httpx>=0.27.0`, `locust>=2.20.0`, `structlog`  
**Storage**:
- Session Memory: Redis key-value store with in-memory LRU fallback (`InMemorySessionStore`)
- Dense Vectors: Qdrant collection `science_textbook_dense`
- Sparse Index: In-memory `science_bm25.pkl` (434 chunks)  
**Testing**: `pytest`, `pytest-asyncio`, `httpx.AsyncClient` (integration & contract tests), `locust` (concurrency load tests)  
**Target Platform**: macOS (Apple Silicon MPS / Darwin) / Linux x86_64 server  
**Project Type**: Asynchronous Web Service (`backend/app/`) consuming decoupled core package (`rag-core`)  
**Performance Goals**:
- Concurrency Target: 50 concurrent active users with 0% dropped transactions (error rate < 0.1%)
- Semaphore Queue Wait: Average < 200ms
- Response Responsiveness: p95 latency < 2.5s under full concurrent load
- Streaming Responsiveness: TTFT < 1.0s for 95% of queries
- Session Lookup Latency: < 15ms  
**Constraints**: Pure 3-layer decoupled architecture (Principle I), zero unindexed drift (Principle X), human approval before commits (Principle XI).

---

## Constitution Check

*GATE: Evaluated during Phase 0 research and verified post-Phase 1 design.*

- [x] **Principle I: Three-Layer Decoupled Architecture**: `backend` imports `rag-core` as a clean dependency. No FastAPI or web dependencies leak backward into `rag-core`.
- [x] **Principle II: Evaluation-Driven Quality**: Includes automated Locust load testing and dual-scorer validation preserving Hit@5, MRR, and Faithfulness.
- [x] **Principle III: Granular Observability**: OpenTelemetry Langfuse trace IDs correlated across FastAPI request headers (`X-Trace-ID`) and SSE done events.
- [x] **Principle V: Defense-in-Depth Security**: Strict session isolation preventing cross-session memory leakage; input query length bounded to 1,000 characters.
- [x] **Principle VI: Performance Caching & Cost Efficiency**: Sliding-window session memory bounded to 4 turns; cost per 1,000 queries and cache hit rate tracked in `/api/v1/metrics`.
- [x] **Principle VII: Law of Parsimony & Component Maximization**: Reuses existing `AdvancedRAGPipeline` and `GatedQueryRewriter` without re-architecting the core; native `asyncio.Semaphore` used instead of external heavy message brokers.
- [x] **Principle VIII: Continuous Development Logging**: Milestone recorded in `development_log.md`.
- [x] **Principle X: Iterative Codebase Knowledge Graph Maintenance**: Pre-hook verified; post-implementation hooks will re-index `.codebase-memory/graph.db.zst`.
- [x] **Principle XI & XII: Enterprise Rigor & The Tool Trinity**: Built with full unit, integration, and load test suites without fragile hacks or ungrounded assumptions.

---

## Project Structure

### Documentation (this feature)

```text
specs/004-concurrency-gateway-sessions/
├── spec.md              # Feature specification
├── plan.md              # This file (implementation plan)
├── research.md          # Technical decisions and architectural rationales
├── data-model.md        # Pydantic models, dialogue turn, and session schemas
├── quickstart.md        # Step-by-step runnable validation and load test guide
├── checklists/
│   └── requirements.md  # Spec quality checklist (16/16 passed)
└── contracts/
    ├── gateway_contracts.py  # Abstract interfaces (ISessionStore, IConcurrencyGateway)
    └── openapi_endpoints.md  # REST & SSE endpoint contracts
```

### Source Code (repository root)

```text
backend/
├── pyproject.toml                     # Adds redis and locust dependencies
├── app/
│   ├── main.py                        # FastAPI application factory, lifespan, CORS
│   ├── core/
│   │   ├── config.py                  # Gateway settings (concurrency limit, redis url)
│   │   ├── gateway.py                 # ConcurrencyGateway (asyncio.Semaphore=50 rate-guard)
│   │   └── telemetry.py               # OperationalMetricsTracker (CBR, queue delay, percentiles)
│   ├── schemas/
│   │   ├── chat.py                    # ChatRequest, ChatResponse, StreamEvent models
│   │   └── session.py                 # SessionState, DialogueTurn models
│   ├── services/
│   │   ├── session_store.py           # ISessionStore (RedisSessionStore + InMemory fallback)
│   │   └── rag_service.py             # RAGService bridge to AdvancedRAGPipeline
│   └── api/
│       └── v1/
│           ├── router.py              # APIRouter mounting all v1 endpoints
│           └── endpoints/
│               ├── chat.py            # POST /chat and POST /chat/stream (SSE)
│               ├── sessions.py        # GET & DELETE /sessions/{session_id}
│               ├── metrics.py         # GET /metrics (Prometheus-ready KPIs)
│               └── health.py          # GET /health (Readiness & liveness probes)
└── tests/
    ├── conftest.py                    # Async test fixtures, test client, mock sessions
    ├── test_gateway_concurrency.py    # Unit & integration tests for semaphore and rate-guard
    ├── test_session_store.py          # Tests for Redis and InMemory sliding-window TTL
    ├── test_chat_endpoints.py         # REST & SSE streaming endpoint integration tests
    └── load/
        ├── locustfile.py              # Locust 50-user load test simulation
        └── run_load_test.py           # Automated headless load test runner and report generator
```

---

## Phased Implementation Sequence

### Phase 1: Setup & Dependencies
- Add `redis>=5.0.0` and `locust>=2.20.0` to `backend/pyproject.toml`.
- Configure gateway settings in `backend/app/core/config.py` (`concurrency_limit=50`, `session_ttl_seconds=7200`, `redis_url`).

### Phase 2: Session Store & Multi-Turn Memory
- Implement `ISessionStore` with `RedisSessionStore` and `InMemorySessionStore` in `backend/app/services/session_store.py`.
- Enforce 4-turn sliding window retention and 2-hour TTL expiration.
- Write unit tests in `backend/tests/test_session_store.py`.

### Phase 3: Concurrency Rate-Guard & Telemetry Tracker
- Implement `OperationalMetricsTracker` in `backend/app/core/telemetry.py` measuring queue latency, active requests, circuit breaker trip rate, and latency percentiles.
- Implement `ConcurrencyGateway` in `backend/app/core/gateway.py` with `asyncio.Semaphore(50)` rate-guard.
- Write unit tests in `backend/tests/test_gateway_concurrency.py`.

### Phase 4: API Endpoints (Chat, Streaming, Health, Metrics)
- Implement `RAGService` bridging FastAPI to `AdvancedRAGPipeline` (with history resolution).
- Build `POST /api/v1/chat` and `POST /api/v1/chat/stream` (SSE) in `backend/app/api/v1/endpoints/chat.py`.
- Build `GET /api/v1/metrics` and `GET /api/v1/health` in `endpoints/metrics.py` and `health.py`.
- Write API integration tests in `backend/tests/test_chat_endpoints.py`.

### Phase 5: Locust 50-User Concurrency Load Test Suite
- Implement `backend/tests/load/locustfile.py` and `run_load_test.py`.
- Execute 50-virtual-user concurrency load test validating error rate < 0.1% and p95 latency < 2.5s.

### Phase 6: Polish & Governance
- Add Makefile targets (`make serve`, `make load-test`).
- Refresh Codebase Knowledge Graph via codebase-memory-reindex.
- Record Milestone 22 in `development_log.md`.

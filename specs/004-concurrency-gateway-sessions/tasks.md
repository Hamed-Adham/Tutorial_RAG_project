# Tasks: Concurrency Gateway, Session Memory & Production Telemetry

**Feature Branch**: `004-concurrency-gateway-sessions`  
**Specification**: [spec.md](spec.md)  
**Plan**: [plan.md](plan.md)  
**Data Model**: [data-model.md](data-model.md)  
**Contracts**: [contracts/gateway_contracts.py](contracts/gateway_contracts.py)  

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Configure backend dependencies, gateway settings, and shared Pydantic data schemas.

- [X] T001 Configure backend dependencies (`redis>=5.0.0`, `locust>=2.20.0`, `sse-starlette>=2.0.0`) in `backend/pyproject.toml`
- [X] T002 [P] Define gateway settings (`concurrency_limit=50`, `session_ttl_seconds=7200`, `redis_url="redis://localhost:6379/0"`) in `backend/app/core/config.py`
- [X] T003 [P] Create Pydantic request and response schemas (`ChatRequest`, `ChatResponse`, `DialogueTurnModel`, `SessionStateModel`) in `backend/app/schemas/chat.py` per data-model.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core session storage infrastructure and abstract gateway protocols required by all user stories.

**⚠️ CRITICAL**: No user story can begin until this phase is complete.

- [X] T004 Define session state and turn models in `backend/app/schemas/session.py` matching `ISessionStore` contract
- [X] T005 [P] Implement `ISessionStore` abstract protocol in `backend/app/services/session_store.py`
- [X] T006 [P] Implement `InMemorySessionStore` fallback with 4-turn sliding-window retention and TTL eviction in `backend/app/services/session_store.py`
- [X] T007 [P] Implement `RedisSessionStore` using `redis.asyncio` with 2-hour TTL and automated in-memory fallback in `backend/app/services/session_store.py`
- [X] T008 Refresh Codebase Knowledge Graph via codebase-memory-mcp `index_repository(persistence=True)` to record foundational baseline

**Checkpoint**: Session storage operational with Redis and in-memory fallback.

---

## Phase 3: User Story 1 - High-Throughput Concurrency & Load Protection (Priority: P1) 🎯 MVP

**Goal**: Deliver an asynchronous FastAPI gateway with an `asyncio.Semaphore(50)` rate-guard serving 50 simultaneous student inquiries without connection exhaustion.

**Independent Test**: Run `pytest backend/tests/test_gateway_concurrency.py` asserting 50 parallel requests acquire semaphore slots without connection starvation.

### Tests for User Story 1
- [X] T009 [P] [US1] Write unit and concurrency tests for `ConcurrencyGateway` semaphore rate-guard in `backend/tests/test_gateway_concurrency.py`

### Implementation for User Story 1
- [X] T010 [P] [US1] Implement `ConcurrencyGateway` with `asyncio.Semaphore(50)` and queue delay tracking in `backend/app/core/gateway.py`
- [X] T011 [US1] Implement `RAGService` bridging FastAPI to `AdvancedRAGPipeline` with client connection pooling in `backend/app/services/rag_service.py`
- [X] T012 [US1] Implement batch chat endpoint `POST /api/v1/chat` in `backend/app/api/v1/endpoints/chat.py`
- [X] T013 [US1] Mount API v1 router and initialize FastAPI application with lifespan warmup in `backend/app/main.py`
- [X] T014 [US1] Execute concurrency integration test verifying 50 parallel requests succeed without error in `backend/tests/test_gateway_concurrency.py`

**Checkpoint**: User Story 1 functional and independently testable (MVP reached).

---

## Phase 4: User Story 2 - Multi-Turn Conversational Session Memory (Priority: P2)

**Goal**: Persist dialogue history across session turns in Redis with sliding-window context retention (last 4 turns) and 2-hour TTL expiration for pronoun resolution.

**Independent Test**: Run `pytest backend/tests/test_session_store.py` verifying pronoun antecedent resolution and 2-hour TTL expiration.

### Tests for User Story 2
- [X] T015 [P] [US2] Write unit tests for multi-turn session persistence, sliding window (last 4 turns), and TTL expiration in `backend/tests/test_session_store.py`

### Implementation for User Story 2
- [X] T016 [US2] Integrate session store history retrieval and turn appending into `RAGService` in `backend/app/services/rag_service.py`
- [X] T017 [US2] Wire conversation history into `AdvancedRAGPipeline.query` for pronoun resolution via `GatedQueryRewriter` in `backend/app/api/v1/endpoints/chat.py`
- [X] T018 [US2] Implement session management endpoint `DELETE /api/v1/sessions/{session_id}` in `backend/app/api/v1/endpoints/sessions.py`
- [X] T019 [US2] Execute multi-turn conversation test verifying pronoun resolution across 4 turns in `backend/tests/test_session_store.py`

**Checkpoint**: Multi-turn session memory operational with sliding-window retention and pronoun resolution.

---

## Phase 5: User Story 3 - Real-Time Streaming Token Delivery (Priority: P3)

**Goal**: Deliver progressive token streaming via Server-Sent Events (SSE) with Time-To-First-Token (`TTFT`) under 1.0 second and verified citations at stream completion.

**Independent Test**: Run `pytest backend/tests/test_chat_endpoints.py` verifying SSE token streaming and client cancellation handling.

### Tests for User Story 3
- [X] T020 [P] [US3] Write integration tests for SSE streaming endpoint and client cancellation in `backend/tests/test_chat_endpoints.py`

### Implementation for User Story 3
- [X] T021 [US3] Implement SSE streaming endpoint `POST /api/v1/chat/stream` consuming `AdvancedRAGPipeline.query_stream` in `backend/app/api/v1/endpoints/chat.py`
- [X] T022 [US3] Implement formatted citation event emission and TTFT latency measurement in `backend/app/api/v1/endpoints/chat.py`
- [X] T023 [US3] Execute streaming verification test proving TTFT < 1000ms and clean client disconnect cancellation in `backend/tests/test_chat_endpoints.py`

**Checkpoint**: Streaming token delivery operational with TTFT < 1.0s and citation injection.

---

## Phase 6: User Story 4 - Operational Resilience & Production Telemetry (Priority: P4)

**Goal**: Instrument live operational KPIs (queue wait times, circuit breaker trip rates, cost per 1k requests) and execute 50-user Locust load tests.

**Independent Test**: Execute Locust load test script simulating 50 concurrent virtual users; verify zero HTTP error responses and p95 latency under 2.5s.

### Tests for User Story 4
- [X] T024 [P] [US4] Write unit tests for `OperationalMetricsTracker` in `backend/tests/test_telemetry.py`

### Implementation for User Story 4
- [X] T025 [US4] Implement `OperationalMetricsTracker` tracking active workers, queue wait ms, circuit breaker trip rate, and p95/p99 latency in `backend/app/core/telemetry.py`
- [X] T026 [US4] Implement telemetry endpoint `GET /api/v1/metrics` and health probe `GET /api/v1/health` in `backend/app/api/v1/endpoints/metrics.py` and `health.py`
- [X] T027 [US4] Implement Locust multi-user load test simulation script in `backend/tests/load/locustfile.py`
- [X] T028 [US4] Implement headless load test runner and report generator in `backend/tests/load/run_load_test.py`
- [X] T029 [US4] Execute 50-user Locust load test verifying 0% dropped transactions and p95 latency < 2.5s against local gateway

**Checkpoint**: Production telemetry live and 50-user concurrency benchmark verified.

---

## Phase 7: Polish & Governance

**Purpose**: Automation, regression testing, and constitution governance compliance.

- [X] T030 [P] Add `serve` and `load-test` targets in root `Makefile`
- [X] T031 Execute full test suite (`pytest backend/tests/`) and validate `specs/004-concurrency-gateway-sessions/quickstart.md`
- [X] T032 Record Milestone 22 in `development_log.md`
- [X] T033 Refresh Codebase Knowledge Graph via codebase-memory-reindex
- [X] T034 Verify Zero Drift (Principle X Gate: `detect_changes` reports `changed_count == 0`)

---

## Dependencies & Execution Order

```text
Phase 1: Setup (T001..T003)
       │
       ▼
Phase 2: Foundational (T004..T008)
       │
       ▼
Phase 3: User Story 1 - Concurrency Gateway (T009..T014) 🎯 MVP
       │
       ▼
Phase 4: User Story 2 - Multi-Turn Session Memory (T015..T019)
       │
       ▼
Phase 5: User Story 3 - Streaming Token Delivery (T020..T023)
       │
       ▼
Phase 6: User Story 4 - Operational Telemetry & Locust Load Tests (T024..T029)
       │
       ▼
Phase 7: Polish & Governance (T030..T034)
```

---

## Parallel Execution Opportunities

- **Phase 1**: T002 (Config) and T003 (Schemas) can be authored in parallel.
- **Phase 2**: T005 (Interface), T006 (InMemory), and T007 (Redis) can be developed in parallel.
- **Phase 3 (US1)**: T009 (Tests) and T010 (Gateway implementation) can be authored in parallel.
- **Phase 4 (US2)**: T015 (Session tests) can run in parallel with T018 (Sessions endpoint).
- **Phase 5 (US3)**: T020 (Streaming tests) can run in parallel with T021 (Endpoint setup).
- **Phase 6 (US4)**: T024 (Metrics tests) and T027 (Locustfile) can be developed in parallel.
- **Phase 7**: T030 (Makefile) can run in parallel with documentation tasks.

---

## Implementation Strategy (MVP First)

1. **Step 1**: Complete Phase 1 (Setup) and Phase 2 (Foundational session storage).
2. **Step 2**: Execute **Phase 3 (User Story 1 - Concurrency Gateway & Semaphore Rate-Guard)**. This delivers the core high-concurrency MVP endpoint (`POST /api/v1/chat`).
3. **Step 3**: Execute Phase 4 (Multi-Turn Session Memory) to enable natural conversational pronouns.
4. **Step 4**: Execute Phase 5 (Streaming Token Delivery) to unlock sub-1.0s TTFT.
5. **Step 5**: Execute Phase 6 (Operational Telemetry & 50-User Locust Load Test).
6. **Step 6**: Execute Phase 7 (Verification, Knowledge Graph synchronization, and milestone logging).

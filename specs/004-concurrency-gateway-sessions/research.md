# Technical Research: Concurrency Gateway, Session Memory & Production Telemetry

**Feature**: `004-concurrency-gateway-sessions`  
**Date**: 2026-10-13  
**Spec Reference**: [spec.md](spec.md)  

---

## 1. Concurrency Architecture & Semaphore Rate Guard

### Decision
Use FastAPI's native asynchronous event loop (`asyncio`) coupled with an explicit, global `asyncio.Semaphore(value=50)` rate-guard protecting the inference and retrieval pipelines. Pair this with a dedicated connection pool for HTTP/client requests (`httpx.AsyncClient(limits=httpx.Limits(max_connections=100, max_keepalive_connections=50))`) and Qdrant connection pooling (`qdrant_client.QdrantClient(limits=50)`).

### Rationale
- **Zero Process Blocking**: All I/O operations (Qdrant vector queries, BM25 lookup, Redis session reads, and Nine Router streaming LLM calls) are non-blocking coroutines.
- **Backpressure & Queue Protection**: Without a semaphore, 50 simultaneous requests hitting CPU cross-encoders (`bge-reranker-v2-m3`) would cause severe thread preemption, memory thrashing, and process termination (`SIGABRT`). The semaphore ensures at most 50 requests execute concurrently, while additional bursts wait in the async queue with their wait time tracked.
- **Fail-Safe Throttling**: If the queue exceeds a configured timeout (e.g. 10 seconds), requests receive a graceful HTTP 429 response with a `Retry-After: 5` header rather than hanging indefinitely.

### Alternatives Considered
- *Thread-based Workers (Sync Gunicorn/Flask)*: Rejected because 50 concurrent synchronous workers require 50 distinct OS threads/processes, consuming >10 GB of RAM and causing context switching overhead.
- *Celery / External Job Queues (RabbitMQ)*: Rejected per Constitution Principle VII (Law of Parsimony). External task queues add unnecessary deployment complexity when native `asyncio.Semaphore` provides sub-millisecond queuing inside the FastAPI process.

---

## 2. Multi-Turn Session Memory & Context Window Management

### Decision
Implement `RedisSessionStore` using `redis.asyncio` with an automatic fallback to an in-memory LRU store (`InMemorySessionStore`). Each session stores a bounded sliding window of the last 4 dialogue turns (`List[DialogueTurn]`), persisted with a 2-hour Time-To-Live (TTL) that resets on every user interaction.

### Rationale
- **Sub-5ms Lookups**: Redis string/hash operations take < 2ms, comfortably fitting within the session lookup latency budget (< 15ms).
- **Graceful Offline Degradation**: When Redis is offline or during standalone unit test execution, the pipeline seamlessly degrades to `InMemorySessionStore` without crashing or throwing unhandled connection errors.
- **Context Ceiling Protection**: In conversational scientific learning, turns accumulate quickly. Keeping a strict sliding window of 4 turns (~600 tokens) prevents prompt context bloat and ensures downstream synthesis stays under the 2,500 token context budget.
- **Antecedent Resolution**: The stored dialogue history is fed directly into `GatedQueryRewriter`, allowing TypeSafe Jev and Nine Router to resolve pronouns (*آن، این، او*) into standalone entities.

### Alternatives Considered
- *Relational DB Session State (PostgreSQL)*: Rejected for fast turn-by-turn chat history because relational transactions add 15–30ms of query latency compared to Redis in-memory key-value lookups.
- *Unbounded Chat History*: Rejected because passing 10+ previous turns causes prompt bloat, increases LLM cost linearly, and induces "Lost-in-the-Middle" retrieval dilution.

---

## 3. Real-Time Streaming Token Delivery Protocol

### Decision
Implement streaming generation using Server-Sent Events (SSE) via FastAPI's native `StreamingResponse(media_type="text/event-stream")`. The stream progressively emits JSON-formatted data chunks:
`data: {"type": "token", "content": "..."}\n\n`
and concludes with a final metadata event containing verified citations and latency metrics:
`data: {"type": "done", "citations": [...], "ttft_ms": 140.5, "total_ms": 1250.0}\n\n`

### Rationale
- **Sub-1.0s TTFT**: Yields tokens immediately as they arrive from `AdvancedRAGPipeline.query_stream()`, meeting Success Criterion SC-004 (`TTFT < 1.0s`).
- **Structured Termination**: Plain text streams cannot cleanly deliver citation links and performance metrics. Using event-typed SSE frames allows frontend clients (like CopilotKit or standard EventSource) to render text progressively while attaching interactive citation cards at stream completion.
- **Client Disconnect Handling**: An async generator wrapped in a `try...finally` block detects client socket disconnection (`asyncio.CancelledError`), cancelling LLM inference and freeing connection pool slots immediately.

### Alternatives Considered
- *WebSockets*: Rejected because full-duplex communication is unnecessary for single-query requests; WebSockets add connection state complexity and require specialized load balancer configurations compared to standard HTTP SSE.
- *Chunked Transfer Plain Text*: Rejected because parsing out structured citations and latency headers from a raw text stream is brittle.

---

## 4. Operational Telemetry & Production Health Metrics

### Decision
Implement an in-memory `OperationalMetricsTracker` exposing `/api/v1/metrics` and `/api/v1/health`. Track four core production KPI categories:
1. **Concurrency & Queue Metrics**: Active worker count, semaphore queue wait latency (`queue_wait_ms`), p50, p95, and p99 latency.
2. **Circuit Breaker Health**: Total queries, circuit breaker fallback counts, and Circuit Breaker Trip Rate (`CBR = fallback / total`).
3. **Cache Efficiency**: Total queries, cache hits, and Cache Hit Rate (`CHR = hits / total`).
4. **Financial Economics**: Aggregated prompt/completion tokens and estimated Cost per 1,000 queries (USD).

### Rationale
- **Immediate Detection of Silent Fallbacks**: If the CPU cross-encoder times out under 50-user load, `CBR` immediately spikes in `/api/v1/metrics`, alerting operators before users notice relevance degradation.
- **Zero Overhead**: In-memory atomic counters (`threading.Lock` / thread-safe accumulators) have < 0.01ms overhead per request.
- **Prometheus & Dashboard Ready**: Formatted as JSON, ready for ingestion by Prometheus, Datadog, or custom monitoring dashboards.

### Alternatives Considered
- *Offline Log Parsing*: Rejected because log analysis happens post-mortem, whereas live `/api/v1/metrics` allows immediate automated alerts and autoscaling triggers.

---

## 5. Multi-User Concurrency Benchmark Suite (Locust)

### Decision
Create a dedicated Locust load test harness in `backend/tests/load/locustfile.py` and a standalone CLI verification runner `backend/tests/load/run_load_test.py`. Simulate 50 concurrent virtual users hitting `/api/v1/chat/stream` with real questions from `science_textbook.csv` over a 60-to-300 second test window.

### Rationale
- **True Concurrency Verification**: Verifies that 50 concurrent requests run on the event loop without deadlocks, connection exhaustion, or memory spikes.
- **Empirical SLA Validation**: Produces an automated Markdown and JSON report proving `Error Rate < 0.1%`, `p95 Latency < 2.5s`, and `Queue Wait < 200ms` per Success Criteria SC-001 through SC-003.

### Alternatives Considered
- *Sequential pytest loops*: Rejected because running 50 queries sequentially tests latency, not concurrency.
- *Apache Bench (`ab`) / `wrk`*: Rejected because they do not easily support dynamic Persian JSON POST request payloads or SSE stream consumption.

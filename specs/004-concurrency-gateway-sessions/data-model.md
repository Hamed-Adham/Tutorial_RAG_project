# Data Model: Concurrency Gateway, Session Memory & Production Telemetry

**Feature**: `004-concurrency-gateway-sessions`  
**Date**: 2026-10-13  
**Spec Reference**: [spec.md](spec.md)  

---

## 1. Entity Overview

```text
┌────────────────┐          1..*         ┌────────────────┐
│  SessionState  │ ───────────────────── │  DialogueTurn  │
└────────────────┘                       └────────────────┘
        │
        │ references
        ▼
┌────────────────┐                       ┌────────────────┐
│  ChatRequest   │ ───────────────────── │  ChatResponse  │
└────────────────┘                       └────────────────┘
                                                 │
                                                 ▼
                                         ┌────────────────┐
                                         │  StreamEvent   │
                                         └────────────────┘
```

---

## 2. Entities & Field Specifications

### 2.1 `DialogueTurn`
Represents an individual interaction turn within a multi-turn learning session.

| Field | Type | Description | Validation / Constraints |
| :--- | :--- | :--- | :--- |
| `turn_id` | `str` | Unique turn identifier | UUID4 format |
| `role` | `str` | Speaker role | Strictly `"user"` or `"assistant"` |
| `content` | `str` | Natural language text message | Non-empty string |
| `timestamp` | `datetime` | Creation timestamp | ISO 8601 UTC |
| `citations` | `List[PageCitation]` | Citation references for assistant turns | Optional for user turns |
| `latency_profile` | `Dict[str, float]` | Component latencies in milliseconds | E.g. `{"retrieval_ms": 42.1, ...}` |

---

### 2.2 `SessionState`
Container for multi-turn conversation memory stored in Redis or in-memory fallback.

| Field | Type | Description | Validation / Constraints |
| :--- | :--- | :--- | :--- |
| `session_id` | `str` | Unique session identifier | Length 8–64 characters |
| `created_at` | `datetime` | Session creation timestamp | ISO 8601 UTC |
| `last_active_at` | `datetime` | Last user interaction timestamp | ISO 8601 UTC |
| `turns` | `List[DialogueTurn]` | Bounded list of conversation turns | Strictly capped at last 4 turns (sliding window) |
| `selected_chapters` | `Optional[List[int]]` | Active chapter scope if Mode 1 practice | Values between 1 and 15 |
| `ttl_seconds` | `int` | Expiration time-to-live | Default `7200` (2 hours) |

**State Transitions**:
- `Created` &rarr; Upon first query with new `session_id` or absent session.
- `Updated` &rarr; User query and assistant response appended; oldest turn pruned if length > 4; `ttl` refreshed.
- `Expired` &rarr; Key automatically deleted by Redis or LRU eviction after 2 hours of inactivity.

---

### 2.3 `ChatRequest`
HTTP REST/SSE request payload submitted by client applications.

| Field | Type | Description | Validation / Constraints |
| :--- | :--- | :--- | :--- |
| `query` | `str` | Student inquiry in Persian or English | 1 to 1,000 characters |
| `session_id` | `Optional[str]` | Existing session identifier | If omitted, gateway generates a fresh session |
| `selected_chapters`| `Optional[List[int]]` | Filter chapters (Mode 1 scoped practice) | Chapter integers `[1..15]` |
| `top_k` | `int` | Number of final chunks to synthesize | Default `5`, range `1..10` |
| `stream` | `bool` | Whether to stream token response | Default `False` (REST), `True` for SSE |

---

### 2.4 `ChatResponse`
Complete batch response returned by `/api/v1/chat`.

| Field | Type | Description | Validation / Constraints |
| :--- | :--- | :--- | :--- |
| `session_id` | `str` | Session identifier for follow-up turns | Non-empty string |
| `query` | `str` | Original user inquiry | Echoes input |
| `effective_query` | `str` | Disambiguated / rewritten query | If rewritten; else echoes query |
| `answer` | `str` | Synthesized grounded answer text | Markdown string |
| `citations` | `List[PageCitation]` | Formatted textbook page citations | `doc_id`, `filename`, `page`, `score` |
| `is_refusal` | `bool` | True if query was out-of-scope | Default `False` |
| `latency_profile` | `Dict[str, float]` | Detailed stage latencies in ms | `queue_wait_ms`, `total_ms`, etc. |
| `trace_id` | `str` | Langfuse trace correlation ID | String ID |

---

### 2.5 `StreamEvent`
Discrete Server-Sent Event frame emitted by `/api/v1/chat/stream`.

```json
// Token Chunk Event:
{
  "event": "token",
  "data": {
    "content": "چگالی ",
    "seq": 1
  }
}

// Done / Metadata Event:
{
  "event": "done",
  "data": {
    "session_id": "sess_89f1a2c3",
    "effective_query": "فرمول محاسبه چگالی چیست؟",
    "citations": [
      {
        "doc_id": "C706_olom_haftom-7-144-6-11.pdf",
        "filename": "کتاب علوم تجربی هفتم",
        "page": 4,
        "score": 0.884
      }
    ],
    "is_refusal": false,
    "ttft_ms": 140.5,
    "total_ms": 1150.2,
    "trace_id": "0194e8bc-..."
  }
}

// Error Event:
{
  "event": "error",
  "data": {
    "detail": "Rate limit exceeded. Please retry in 5 seconds.",
    "code": "RATE_LIMIT_EXCEEDED"
  }
}
```

---

### 2.6 `OperationalMetrics`
Operational health telemetry exposed at `/api/v1/metrics`.

| Field | Type | Description | Target SLA |
| :--- | :--- | :--- | :--- |
| `active_requests` | `int` | Currently executing concurrent requests | `0..50` |
| `available_semaphore_slots` | `int` | Unused slots in `asyncio.Semaphore` | Max `50` |
| `avg_queue_wait_ms` | `float` | Average time spent in queue before worker execution | `< 200 ms` |
| `total_requests` | `int` | Cumulative lifetime request count | Integer counter |
| `circuit_breaker_trips` | `int` | Queries falling back to raw RRF | `< 2.0%` of total |
| `circuit_breaker_trip_rate` | `float` | Ratio `trips / total_requests` | `< 0.02` |
| `cache_hits` | `int` | Queries resolved from session/exact cache | Integer counter |
| `cache_hit_rate` | `float` | Ratio `cache_hits / total_requests` | `>= 0.20` |
| `latency_p50_ms` | `float` | Median end-to-end response time | `< 1500 ms` |
| `latency_p95_ms` | `float` | 95th percentile response time | `< 2500 ms` |
| `latency_p99_ms` | `float` | 99th percentile response time | `< 4000 ms` |
| `est_cost_per_1k_usd` | `float` | Estimated token cost per 1,000 inquiries | `< $1.50` |

---

### 2.7 `HealthStatus`
Liveness and readiness check exposed at `/api/v1/health`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | `str` | `"healthy"`, `"degraded"`, or `"unhealthy"` |
| `qdrant_connected` | `bool` | True if vector database responds |
| `bm25_loaded` | `bool` | True if in-memory science index is loaded (434 chunks) |
| `redis_connected` | `bool` | True if Redis responded; False if in fallback memory mode |
| `cross_encoder_warm` | `bool` | True if neural reranker model weights are loaded |
| `uptime_seconds` | `float` | Service uptime in seconds |

# OpenAPI Endpoints Contract: Concurrency Gateway & Session Management

**Feature**: `004-concurrency-gateway-sessions`  
**Base URL**: `http://localhost:8000/api/v1`  
**Protocol**: HTTP/1.1 & HTTP/2  

---

## 1. `POST /api/v1/chat`

Synchronous batch execution endpoint for educational science inquiries.

### Request Headers
```http
Content-Type: application/json
Accept: application/json
```

### Request Body (`ChatRequest`)
```json
{
  "query": "چرا در آزمایش‌های علمی باید متغیرها کنترل شوند؟",
  "session_id": "sess_student_1234",
  "selected_chapters": [1, 2],
  "top_k": 5,
  "stream": false
}
```

### Successful Response (HTTP 200 OK)
```json
{
  "session_id": "sess_student_1234",
  "query": "چرا در آزمایش‌های علمی باید متغیرها کنترل شوند؟",
  "effective_query": "چرا در آزمایش‌های علمی باید متغیرها کنترل شوند؟",
  "answer": "برای اینکه مطمئن شویم تغییرات حاصل تنها به دلیل متغیر مستقل است و خطای تصادفی حذف شود.\n\n**Sources:**\n[کتاب علوم تجربی هفتم](C706_olom_haftom-7-144-2-5.pdf#page=2)",
  "citations": [
    {
      "doc_id": "C706_olom_haftom-7-144-2-5.pdf",
      "filename": "کتاب علوم تجربی هفتم",
      "page": 2,
      "score": 0.884,
      "chunk_id": "SEC-0002_c0"
    }
  ],
  "is_refusal": false,
  "latency_profile": {
    "queue_wait_ms": 1.25,
    "rewrite_ms": 8.12,
    "retrieval_ms": 32.40,
    "rerank_ms": 180.50,
    "expansion_ms": 2.10,
    "synthesis_ms": 850.00,
    "total_ms": 1074.37
  },
  "trace_id": "0194e8bc-d95d-7a3b-9e21-..."
}
```

---

## 2. `POST /api/v1/chat/stream`

Real-time streaming generation endpoint via Server-Sent Events (SSE).

### Request Headers
```http
Content-Type: application/json
Accept: text/event-stream
```

### Request Body (`ChatRequest`)
```json
{
  "query": "فرمول چگالی چیست؟",
  "session_id": "sess_student_1234",
  "selected_chapters": null,
  "top_k": 5,
  "stream": true
}
```

### Response Stream (HTTP 200 OK, `text/event-stream`)
```text
event: token
data: {"content": "چگالی ", "seq": 1}

event: token
data: {"content": "مقدار ", "seq": 2}

event: token
data: {"content": "جرمی است ", "seq": 3}

event: done
data: {"session_id": "sess_student_1234", "effective_query": "فرمول چگالی چیست؟", "citations": [{"doc_id": "C706_olom_haftom-7-144-6-11.pdf", "page": 4}], "is_refusal": false, "ttft_ms": 140.5, "total_ms": 1120.0, "trace_id": "0194e8bc-..."}
```

---

## 3. `GET /api/v1/metrics`

Prometheus-compatible operational health telemetry reporting live system KPIs.

### Response (HTTP 200 OK)
```json
{
  "active_requests": 14,
  "available_semaphore_slots": 36,
  "avg_queue_wait_ms": 12.45,
  "total_requests": 1250,
  "circuit_breaker_trips": 6,
  "circuit_breaker_trip_rate": 0.0048,
  "cache_hits": 312,
  "cache_hit_rate": 0.2496,
  "latency_p50_ms": 1120.0,
  "latency_p95_ms": 1850.0,
  "latency_p99_ms": 2340.0,
  "est_cost_per_1k_usd": 1.18
}
```

---

## 4. `GET /api/v1/health`

Service readiness and liveness probe.

### Response (HTTP 200 OK)
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "bm25_loaded": true,
  "redis_connected": true,
  "cross_encoder_warm": true,
  "uptime_seconds": 18450.2
}
```

---

## 5. `DELETE /api/v1/sessions/{session_id}`

Purges active session history explicitly upon user sign-out or conversation reset.

### Response (HTTP 200 OK)
```json
{
  "session_id": "sess_student_1234",
  "deleted": true
}
```

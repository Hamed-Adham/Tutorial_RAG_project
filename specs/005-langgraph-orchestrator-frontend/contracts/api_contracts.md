# API Contracts: Frontend & Gateway Endpoints

**Feature Branch**: `005-langgraph-orchestrator-frontend`  
**Specification**: [spec.md](../spec.md)  
**Date**: 2026-10-14  

---

## 1. Streaming Educational Chat Endpoint

### `POST /api/v1/chat/stream`
Executes real-time progressive response generation via Server-Sent Events (SSE).

* **Headers**: `Content-Type: application/json`
* **Request Body (`ChatRequest`)**:
  ```json
  {
    "query": "فرمول چگالی چیست و چه کاربردی دارد؟",
    "session_id": "sess_m1a2b3c4",
    "selected_chapters": [2],
    "top_k": 5,
    "stream": true
  }
  ```

* **Response Event Stream (`text/event-stream`)**:
  * **Token Event** (`event: token`):
    ```text
    event: token
    data: {"content": "چگالی ", "seq": 1}

    event: token
    data: {"content": "برابر است ", "seq": 2}
    ```
  * **Done Event** (`event: done`):
    ```text
    event: done
    data: {
      "session_id": "sess_m1a2b3c4",
      "effective_query": "فرمول چگالی چیست؟",
      "citations": [
        {
          "doc_id": "C706_olom_haftom-7-144-2-5.pdf",
          "filename": "کتاب علوم تجربی هفتم",
          "page": 4,
          "score": 0.892,
          "chunk_id": "chk_science_01"
        }
      ],
      "is_refusal": false,
      "ttft_ms": 320.5,
      "total_ms": 1420.8
    }
    ```

---

## 2. Batch Chat Endpoint (Fallback)

### `POST /api/v1/chat`
* **Headers**: `Content-Type: application/json`
* **Response Body (`ChatResponse`)**:
  ```json
  {
    "session_id": "sess_m1a2b3c4",
    "query": "فرمول چگالی چیست؟",
    "effective_query": "فرمول چگالی چیست؟",
    "answer": "چگالی برابر است با نسبت جرم به حجم (d = m / V). [کتاب علوم تجربی هفتم (ص ۴)]",
    "citations": [
      {
        "doc_id": "C706_olom_haftom-7-144-2-5.pdf",
        "filename": "کتاب علوم تجربی هفتم",
        "page": 4,
        "score": 0.892,
        "chunk_id": "chk_science_01"
      }
    ],
    "is_refusal": false,
    "latency_profile": {
      "queue_wait_ms": 1.2,
      "rewrite_ms": 45.1,
      "retrieval_ms": 35.8,
      "rerank_ms": 120.4,
      "expansion_ms": 15.2,
      "synthesis_ms": 650.0,
      "total_ms": 867.7
    },
    "trace_id": "tr_9a8b7c6d5e4f"
  }
  ```

---

## 3. Operational Metrics Endpoint

### `GET /api/v1/metrics`
* **Response Body (`OperationalMetrics`)**:
  ```json
  {
    "active_requests": 2,
    "available_semaphore_slots": 48,
    "avg_queue_wait_ms": 4.5,
    "total_requests": 142,
    "circuit_breaker_trips": 0,
    "circuit_breaker_trip_rate": 0.0,
    "cache_hits": 18,
    "cache_hit_rate": 0.126,
    "latency_p50_ms": 450.2,
    "latency_p95_ms": 1250.8,
    "latency_p99_ms": 1850.4,
    "est_cost_per_1k_usd": 0.045
  }
  ```

---

## 4. System Health Probe Endpoint

### `GET /api/v1/health`
* **Response Body (`SystemHealth`)**:
  ```json
  {
    "status": "healthy",
    "qdrant_connected": true,
    "bm25_loaded": true,
    "redis_connected": true,
    "cross_encoder_warm": true,
    "uptime_seconds": 3600
  }
  ```

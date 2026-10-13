# Quickstart Validation Guide: Concurrency Gateway, Session Memory & Production Telemetry

**Feature**: `004-concurrency-gateway-sessions`  
**Date**: 2026-10-13  
**Spec Reference**: [spec.md](spec.md)  
**Data Models**: [data-model.md](data-model.md)  
**Contracts**: [gateway_contracts.py](contracts/gateway_contracts.py)  

---

## 1. Prerequisites & Infrastructure Setup

Start the background infrastructure services:
```bash
# 1. Start Qdrant and Redis services via docker-compose
docker compose up -d qdrant redis

# 2. Verify Redis connectivity
docker compose exec redis redis-cli ping
# Expected: PONG
```

---

## 2. Launch the FastAPI Gateway Server

Run the asynchronous FastAPI application with Uvicorn:
```bash
PYTHONPATH=rag-core/src:backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Expected Output*:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Eagerly warming up AdvancedRAGPipeline and CrossEncoder...
INFO:     Connecting to Redis Session Store at redis://localhost:6379/0...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 3. Scenario 1: Verify Health & Readiness Probes

Check system readiness via HTTP GET:
```bash
curl -s http://localhost:8000/api/v1/health | jq .
```
*Expected Outcome*:
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "bm25_loaded": true,
  "redis_connected": true,
  "cross_encoder_warm": true,
  "uptime_seconds": 12.4
}
```

---

## 4. Scenario 2: Multi-Turn Conversation Continuity (Pronoun Resolution)

### Turn 1: Introduce Concept
```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_student_01",
    "query": "استوانه مدرج چیست و چه کاربردی دارد؟"
  }' | jq '{session_id, query, answer: .answer[:90]}'
```

### Turn 2: Follow-up Inquiry with Pronoun Reference
```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_student_01",
    "query": "نحوه خواندن سطح مایع روی آن چگونه است؟"
  }' | jq '{session_id, effective_query, answer: .answer[:120], citations}'
```
*Expected Outcome*:
- `effective_query` resolves the pronoun *آن* to *استوانه مدرج*.
- Assistant explains reading the meniscus curve on the graduated cylinder.
- Citations point to `C706_olom_haftom-7-144-6-11.pdf#page=4`.

---

## 5. Scenario 3: Real-Time Streaming Token Delivery (TTFT < 1.0s)

Test Server-Sent Events (SSE) streaming delivery:
```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_student_01",
    "query": "فرمول محاسبه چگالی چیست؟",
    "stream": true
  }'
```
*Expected Outcome*:
```text
event: token
data: {"content": "چگالی "}

event: token
data: {"content": "مقدار "}

event: done
data: {"effective_query": "فرمول محاسبه چگالی چیست؟", "citations": [...], "ttft_ms": 145.2, "total_ms": 1120.0}
```
*Validation*: First token is emitted with `ttft_ms < 1000.0`.

---

## 6. Scenario 4: Live Production Telemetry & KPIs

Inspect real-time concurrency and circuit breaker health:
```bash
curl -s http://localhost:8000/api/v1/metrics | jq .
```
*Expected Outcome*:
```json
{
  "active_requests": 0,
  "available_semaphore_slots": 50,
  "avg_queue_wait_ms": 1.4,
  "total_requests": 3,
  "circuit_breaker_trips": 0,
  "circuit_breaker_trip_rate": 0.0,
  "cache_hits": 1,
  "cache_hit_rate": 0.33,
  "latency_p50_ms": 1120.0,
  "latency_p95_ms": 1450.0
}
```

---

## 7. Scenario 5: 50-User Concurrency Load Test (Locust)

Execute the automated multi-user stress benchmark:
```bash
# Run 50 virtual users for 60 seconds against local gateway
.venv/bin/locust -f backend/tests/load/locustfile.py \
  --headless \
  --users 50 \
  --spawn-rate 10 \
  --run-time 60s \
  --host http://localhost:8000 \
  --csv backend/tests/load/results/load_report
```
*Success Verification*:
- Total HTTP failures: `0` (Error rate strictly `< 0.1%`).
- 95th Percentile Response Time: `< 2500ms`.
- Semaphore queue wait time: `< 200ms`.

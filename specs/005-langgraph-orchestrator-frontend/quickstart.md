# Quickstart: LangGraph Orchestrator & Next.js Chatbot UI

**Feature Branch**: `005-langgraph-orchestrator-frontend`  
**Specification**: [spec.md](spec.md)  
**Date**: 2026-10-14  

---

## Prerequisites

1. **Python Virtual Environment**: Active `.venv` with dependencies installed (`pip install -e rag-core -e backend`).
2. **Node.js Environment**: Node.js 20+ and npm installed.
3. **Local Infrastructure**: Docker services running (`Qdrant` on port 6333, `Redis` on port 6379).

---

## 1. Automated Verification Commands

### A. Next.js Production Build Validation
Verify that the frontend compiles cleanly with TypeScript, Tailwind CSS, and local Peyda font assets:
```bash
cd frontend && ./node_modules/.bin/next build
```
* **Expected Output**: `✓ Compiled successfully`, `Generating static pages (4/4)`, `Route /: ~57.6 kB`.

### B. LangGraph Orchestrator Unit Tests
Verify StateGraph node transitions, conditional refusal branching, and Document interoperability:
```bash
PYTHONPATH=rag-core/src:rag-core .venv/bin/pytest rag-core/tests/test_enterprise_graph.py
```
* **Expected Output**: `5 passed in ~2.0s`.

### C. Full Project Regression Suite
Run both backend and rag-core test suites:
```bash
PYTHONPATH=backend:rag-core/src .venv/bin/pytest backend/tests && \
PYTHONPATH=rag-core/src:rag-core .venv/bin/pytest rag-core/tests
```
* **Expected Output**: `18 passed in backend/tests` and `97 passed in rag-core/tests` (Total: 115 passing).

---

## 2. Interactive Local Run Guide

### Step 1: Start Infrastructure & Backend Service
```bash
# 1. Start Docker containers (Qdrant & Redis)
make up

# 2. Start FastAPI Asynchronous Gateway (runs on port 8000)
make serve
```

### Step 2: Start Frontend Application
In a separate terminal:
```bash
cd frontend
npm run dev
```
Open **`http://localhost:3000`** in your browser.

---

## 3. End-to-End Validation Scenarios

### Scenario 1: Curriculum Query & Streaming Verification (User Story 1)
1. Type a scientific inquiry in Persian: `"فرمول چگالی چیست؟"`
2. Press **Enter**.
3. **Expected Outcome**:
   * Initial tokens appear within **1.0 second** (TTFT < 1.0s).
   * Response renders in **Peyda font** with proper RTL text alignment.
   * Citations badge appears at the bottom: `[کتاب علوم تجربی هفتم (ص ۴)]`.

### Scenario 2: Scoped Practice Mode (User Story 2)
1. Click the **"فصل‌های کتاب" (Chapter Scope)** button in the header.
2. Select **فصل ۲: اندازه‌گیری در علوم و ابزارهای آن** and click **تایید و اعمال**.
3. Submit inquiry: `"یکای طول و جرم در SI چیست؟"`
4. **Expected Outcome**: Evidence retrieved is strictly confined to Chapter 2.

### Scenario 3: Deterministic Grounded Refusal (User Story 4)
1. Submit an out-of-scope query: `"قیمت بیت‌کوین چقدر است؟"`
2. **Expected Outcome**:
   * Immediate refusal response rendered: `"متأسفانه سوال شما خارج از حوزه کتاب علوم تجربی هفتم است..."`
   * Latency waterfall badge indicates `0` vector searches and zero-token refusal.

### Scenario 4: Live Operational Telemetry Inspection
1. Click the **"وضعیت سیستم" (Telemetry)** button in the header.
2. **Expected Outcome**: Live modal displays active requests (`/ 50`), queue wait time, cache hit rate, circuit breaker status, and p50/p95/p99 latency percentiles.

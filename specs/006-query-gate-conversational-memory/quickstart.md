# Quickstart & Verification Guide: Feature 006

**Feature**: QueryGate & Conversational Memory Routing  
**Directory**: `specs/006-query-gate-conversational-memory`  
**Date**: 2026-10-14  

---

## 1. Prerequisites & Environment Setup

Ensure environment variables and services are operational:
```bash
# 1. Activate Python virtual environment
source .venv/bin/activate

# 2. Verify SubQ API Key in .env
grep -E "SUBQ_API_KEY|NINE_ROUTER" .env
```

---

## 2. Automated Test Verification

Execute the dedicated unit and contract test suites covering `QueryGate`, SubQ Decision Model 2, and conversational history dispatching:

```bash
# Run unit tests for QueryGate
pytest rag-core/tests/test_query_gate.py -v

# Run integration tests for LangGraph conversational memory nodes
pytest rag-core/tests/test_enterprise_graph.py -v -k "test_query_gate"

# Run backend API contract tests
pytest backend/tests/test_chat_endpoints.py -v
```

---

## 3. Manual Scenario Verification via cURL

### Scenario 1: Completely Irrelevant Gating & Instant Refusal (P1)
Verify that out-of-scope inquiries are intercepted in <25ms without invoking retrieval or synthesis:
```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "نظرت درباره بازی استقلال و پرسپولیس چیه؟",
    "session_id": "test_refusal_sess"
  }' | jq '{intent_tier: .query_gate_decision.intent_tier, is_refusal: .query_gate_decision.is_refusal, timing: .timing}'
```
**Expected Outcome**:
- `intent_tier`: `"completely_irrelevant"`
- `is_refusal`: `true`
- `retrieval_ms`: `0.0`
- `total_ms`: `< 25ms`

---

### Scenario 2: Partially Relevant Query Distillation (P1)
Verify that extraneous chatter is dropped, forwarding only the purified scientific inquiry:
```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "امروز هوا بارونیه ولی بگو چگالی چوب چقدره؟",
    "session_id": "test_distill_sess"
  }' | jq '{effective_query: .effective_query, intent_tier: .query_gate_decision.intent_tier}'
```
**Expected Outcome**:
- `intent_tier`: `"partially_relevant"`
- `effective_query`: Purified to core inquiry (e.g. `"چگالی چوب چقدره؟"`)

---

### Scenario 3: `needs_history_only` Retrieval Bypass & Conversational Badge (P1)
Ask a follow-up requesting a summary of the prior answer:
```bash
# Step A: Prime session with initial inquiry
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "سفره‌های آب زیرزمینی چگونه تشکیل می‌شوند؟",
    "session_id": "sess_history_demo"
  }' > /dev/null

# Step B: Request clarification/summary
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "پاسخ قبلی را خلاصه کن",
    "session_id": "sess_history_demo"
  }' | jq '{badge: .conversational_context_badge, retrieval_ms: .timing.retrieval_ms, citations_count: (.citations | length)}'
```
**Expected Outcome**:
- `badge`: `"پاسخ بر اساس گفتگوی پیشین"`
- `retrieval_ms`: `0.0`
- `citations_count`: `0`

---

### Scenario 4: SubQ Decision Model 2 with Parallel 10-10 Execution (P2)
In a session with antecedent questions, ask a pronoun-dependent follow-up:
```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "آزمایشی که توضیح دادی چه خطاهایی دارد؟",
    "session_id": "sess_history_demo"
  }' | jq '{decision: .query_gate_decision, timing: .timing}'
```
**Expected Outcome**:
- `context_need_tier`: `"needs_history_and_retrieve"`
- Decision Model 2 executes SubQ batch call in parallel (~200ms)
- Antecedent turn defining the target concept is selected

---

### Scenario 5: UI Reply & Manual Context Selection (P3)
1. Open the UI at `http://localhost:3000`.
2. Toggle the Conversation Memory switch to **OFF**.
3. On a prior assistant message, click the **«پاسخ به این پیام / انتخاب به عنوان زمینه»** icon.
4. Verify the floating dismissible quote chip appears above the chat textarea.
5. Submit a question (e.g. *"این بخش را بیشتر باز کن"*).
6. Verify the answer directly references the attached message without enabling global memory.

# API Contracts: Feature 006

**Feature**: QueryGate & Conversational Memory Routing  
**Directory**: `specs/006-query-gate-conversational-memory`  
**Date**: 2026-10-14  

---

## 1. REST Endpoint: `POST /api/v1/chat`

Executes synchronous conversational inquiry with QueryGate routing, SubQ Decision Model 2 context filtering, and session summary caching.

### Request Body
```json
{
  "query": "آزمایشی که توضیح دادی چه خطاهایی دارد؟",
  "session_id": "sess_a8b9c1d2e3f4",
  "selected_chapters": [7],
  "history": [
    { "role": "user", "content": "سفره‌های آب زیرزمینی چگونه تشکیل می‌شوند؟" },
    { "role": "assistant", "content": "سفره‌های آب زیرزمینی یا آبخوان‌ها از دو بخش نفوذپذیر..." }
  ],
  "history_enabled": true,
  "selected_context_turn_id": null
}
```

### Request Headers
- `Content-Type: application/json`
- `Accept: application/json`

### Success Response (`200 OK`)
```json
{
  "response": "آزمایش شبیه‌سازی آبخوان ممکن است خطاهای زیر را داشته باشد:\n۱. ناصاف بودن لایه‌های نفوذناپذیر...\n\n💡 فصل 7: بخش SEC-0020 (صفحه 5)",
  "session_id": "sess_a8b9c1d2e3f4",
  "citations": [
    {
      "chapter_num": 7,
      "chapter_name": "سفر آب درون زمین",
      "page_num": 5,
      "section_id": "SEC-0020",
      "section_title": "سفره‌های آب زیرزمینی",
      "badge": "فصل 7، صفحه 5",
      "pdf_url": "/api/v1/pdf/فصل 7 (سفر آب درون زمین)#page=5"
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "SEC-0020",
      "text": "سفره‌های آب زیرزمینی(آبخوان) به طور کلی دو نوع هستند...",
      "chapter_num": 7,
      "page_num": 5,
      "score": 0.892
    }
  ],
  "effective_query": "خطاهای آزمایش سفره‌های آب زیرزمینی و آبخوان",
  "timing": {
    "total_ms": 780.4,
    "gate_ms": 18.2,
    "decision_model_2_ms": 210.5,
    "retrieval_ms": 85.1,
    "rerank_ms": 42.0,
    "synthesis_ms": 424.6
  },
  "conversational_context_badge": null,
  "q_and_a_summary": "بررسی خطاهای احتمالی در آزمایش سفره‌های آب زیرزمینی آزاد و تحت فشار بر اساس مفاهیم نفوذپذیری فصل هفتم.",
  "query_gate_decision": {
    "intent_tier": "relevant",
    "context_need_tier": "needs_history_and_retrieve",
    "is_refusal": false,
    "bypass_retrieval": false,
    "selected_history_items_count": 1
  }
}
```

---

## 2. Special Scenarios & Badging Responses

### 2.1 Scenario: `needs_history_only` (Option 2)
When user sends *"پاسخ قبلی را خلاصه کن"*:
```json
{
  "response": "به طور خلاصه: آبخوان‌ها به دو دسته آزاد و تحت فشار تقسیم می‌شوند...",
  "session_id": "sess_a8b9c1d2e3f4",
  "citations": [],
  "retrieved_chunks": [],
  "effective_query": "پاسخ قبلی را خلاصه کن",
  "timing": {
    "total_ms": 380.2,
    "gate_ms": 14.5,
    "decision_model_2_ms": 185.0,
    "retrieval_ms": 0.0,
    "rerank_ms": 0.0,
    "synthesis_ms": 180.7
  },
  "conversational_context_badge": "پاسخ بر اساس گفتگوی پیشین",
  "q_and_a_summary": "خلاصه‌سازی توضیحات پیشین درباره انواع آبخوان‌ها.",
  "query_gate_decision": {
    "intent_tier": "relevant",
    "context_need_tier": "needs_history_only",
    "is_refusal": false,
    "bypass_retrieval": true,
    "selected_history_items_count": 1
  }
}
```

### 2.2 Scenario: Completely Irrelevant / Attack Refusal
When user sends *"نظرت درباره بازی استقلال و پرسپولیس چیه؟"*:
```json
{
  "response": "من دستیار تخصصی کتاب علوم تجربی پایه هفتم هستم و تنها می‌توانم به پرسش‌های مرتبط با مباحث و مفاهیم این کتاب پاسخ دهم. لطفاً پرسش خود را در حوزه علوم تجربی مطرح کنید.",
  "session_id": "sess_a8b9c1d2e3f4",
  "citations": [],
  "retrieved_chunks": [],
  "effective_query": "نظرت درباره بازی استقلال و پرسپولیس چیه؟",
  "timing": {
    "total_ms": 12.4,
    "gate_ms": 12.0,
    "decision_model_2_ms": 0.0,
    "retrieval_ms": 0.0,
    "rerank_ms": 0.0,
    "synthesis_ms": 0.0
  },
  "conversational_context_badge": null,
  "q_and_a_summary": null,
  "query_gate_decision": {
    "intent_tier": "completely_irrelevant",
    "context_need_tier": "needs_neither",
    "is_refusal": true,
    "bypass_retrieval": true,
    "selected_history_items_count": 0
  }
}
```

---

## 3. Streaming Endpoint: `POST /api/v1/chat/stream`

SSE stream delivering real-time tokens with event types:
1. `event: gate_decision` -> Emitted once Gate 1 and Decision Model 2 complete.
2. `event: token` -> Streamed markdown chunks.
3. `event: citations` -> Emitted with citation badges (or empty if `needs_history_only`).
4. `event: done` -> Final payload including `conversational_context_badge`, `q_and_a_summary`, and full timing.

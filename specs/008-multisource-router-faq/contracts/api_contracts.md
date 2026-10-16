# Phase 1 API Contracts: Multi-Source Gateway & Chat Endpoints

**Feature Branch**: `008-multisource-router-faq`  
**Feature Directory**: `specs/008-multisource-router-faq`  

---

## 1. REST Endpoint: `/api/v1/chat/stream` & `/api/v1/chat`

### 1.1 Request Payload Extension

```json
{
  "message": "دستگاه گوارش چگونه کار می‌کند و آیا برای اختراع آن می‌توان از صندوق نوآوری تسهیلات گرفت؟",
  "session_id": "sess_89a3f2b1",
  "selected_sources": ["all"],
  "selected_chapters": [],
  "selected_faq_categories": ["تسهیلات", "ضمانت‌نامه"],
  "history_enabled": true
}
```

#### Field Specifications:
- `selected_sources`: `List[str]`
  - `["science_textbook"]`: Explicitly restricts routing to 7th-grade science. Bypasses Level-0 Router.
  - `["administrative_faq"]`: Explicitly restricts routing to INIF FAQ. Bypasses Level-0 Router.
  - `["all"]` or omitted: Enables autonomous Level-0 SubQ Source Router to evaluate relevance across both sources.
- `selected_faq_categories`: `List[str]` (Optional)
  - Pre-filters FAQ retrieval to specified categories (e.g. `["ضمانت‌نامه"]`). Passed down to Qdrant and FAQ BM25 index.

---

### 1.2 Streaming SSE Event Protocol Extensions

#### New Event: `source_routing`
Emitted immediately when Level-0 Source Router finishes evaluation (<25ms):

```json
event: source_routing
data: {
  "routing_tier": "both_domains",
  "active_sources": ["science_textbook", "administrative_faq"],
  "confidence": 0.94,
  "is_refusal": false,
  "latencies": {
    "router_ms": 18.4
  }
}
```

#### Updated Event: `query_gate`
Emitted for active DM1 executions:

```json
event: query_gate
data: {
  "domain": "science_textbook",
  "intent_tier": "relevant",
  "context_need_tier": "needs_retrieve_only",
  "effective_query": "دستگاه گوارش چگونه کار می‌کند"
}
```

#### Updated Event: `citation`
Supports both Science chapters and FAQ categories:

```json
event: citation
data: {
  "citations": [
    {
      "chunk_id": "chunk_science_ch13_p82_01",
      "doc_id": "science_textbook",
      "chapter": 13,
      "page": 82,
      "label": "علوم تجربی هفتم – فصل ۱۳ – ص ۸۲"
    },
    {
      "chunk_id": "faq_item_54",
      "doc_id": "administrative_faq",
      "category": "تسهیلات",
      "row": 54,
      "label": "پرسش‌های متداول – دسته: تسهیلات – ردیف: ۵۴"
    }
  ]
}
```

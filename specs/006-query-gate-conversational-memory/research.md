# Architectural Research & Technical Decisions (006)

**Feature**: QueryGate & Conversational Memory Routing  
**Directory**: `specs/006-query-gate-conversational-memory`  
**Date**: 2026-10-14  

---

## 1. QueryGate Refactoring & System-Wide Renaming

### Decision
Rename all occurrences of `gated_rewriter` to `query_gate` across modules, class names (`QueryGate`), filenames (`rag-core/src/rag_core/query/query_gate.py`), tests (`rag-core/tests/test_query_gate.py`), and LangGraph node identifiers (`query_gate`).

### Rationale
- The term "rewriter" incorrectly implied that every processed query is rewritten by an LLM.
- In reality, the component serves as an intelligent, multi-tier decision gate that classifies curriculum intent, determines whether external retrieval and/or conversation history is required, and routes accordingly.
- Renaming brings conceptual clarity and adheres to Spec-Kit clarity principles.

### Alternatives Considered
- *Retain `GatedQueryRewriter` and add flags*: Rejected because the class now manages 3-tier curriculum gating, 4-tier dispatching, and triggers Decision Model 2; keeping "rewriter" would be misleading and accumulate technical debt.

---

## 2. 3-Tier Curriculum Intent Gating & Partial Query Purification

### Decision
`QueryGate` implements `intends_curriculum_answer` as a 3-tier choice:
1. `completely_irrelevant`: Query has no connection to the 7th-grade science curriculum (e.g. system prompt extraction, general chit-chat, pop culture, sports, or malicious injection). The pipeline immediately routes to `refusal` without calling vector search or synthesis.
2. `partially_relevant`: Query contains out-of-scope fluff combined with a curriculum inquiry (e.g. *"امروز بارونیه ولی بگو بطن‌های قلب چیستند؟"*). The query transformer extracts and purifies only the relevant scientific question **before** any history evaluation or vector retrieval occurs.
3. `relevant`: Direct scientific query that proceeds without modification.

### Rationale
- In real-world educational usage, students frequently prepend friendly remarks or extraneous chatter before their core question.
- Purifying the query *before* Decision Model 2 and Retrieval ensures that:
  - Vector embeddings and BM25 search are not poisoned by conversational fluff.
  - Decision Model 2 historical relevance matching evaluates purely against the scientific question.

### Alternatives Considered
- *Binary yes/no classification*: Existing implementation rejected mixed queries or allowed noisy text to reach dense vector search, reducing context precision.
- *Post-retrieval query filtering*: Incurs unnecessary latency and cost by retrieving chunks based on noisy queries.

---

## 3. 4-Tier Context Need Dispatching (`need_conv_hist_and_retrieve`)

### Decision
Classify queries into four mutually exclusive execution tiers:
- **Tier 1 (`needs_history_and_retrieve`)**: The query relies on prior conversational turns to resolve antecedents AND requires external textbook retrieval. Triggers Decision Model 2, combines needed summaries with the query, and routes to hybrid search.
- **Tier 2 (`needs_history_only`)**: Meta-inquiry requesting clarification, rephrasing, formatting, or summarization of the previous answer (e.g. *"پاسخ قبلی را خلاصه کن"*). Bypasses dense vector search, BM25, and reranking entirely (0 ms retrieval latency). Selected history context is passed directly to synthesis.
- **Tier 3 (`needs_retrieve_only`)**: Fresh standalone scientific question requiring textbook retrieval, with no conversational dependence. Bypasses history evaluation and proceeds directly to hybrid retrieval.
- **Tier 4 (`needs_neither`)**: Pleasantries or non-retrieval conversational pleasantries that can be acknowledged or refused immediately without database or LLM overhead.

### Rationale
- Prior to this architecture, even simple follow-up requests like *"ساده‌تر بگو"* executed expensive vector searches in Qdrant and neural rerankers.
- Bypassing retrieval for Tier 2 drops end-to-end response latency from ~1.8s down to <600ms and reduces vector database load.

### Alternatives Considered
- *Implicit heuristics (e.g. keyword matching for 'خلاصه کن')*: Brittle, fails with colloquial Persian phrasing, and misses subtle conversational intents.

---

## 4. Cold-Start Fallback on Turn 0

### Decision
When an inquiry arrives in a new session where history length is 0:
- If `need_conv_hist_and_retrieve` is classified as `needs_history_and_retrieve` (Option 1), the system auto-coerces the tier to `needs_retrieve_only` (Option 3).
- If classified as `needs_history_only` (Option 2), the system returns a friendly Persian guidance response: *"هنوز گفتگویی در این نشست انجام نشده است. لطفاً پرسش خود را مطرح کنید."*

### Rationale
- First-turn messages can contain ambiguous phrasing that mimics follow-up inquiries.
- Crashing or attempting to read nonexistent history turns would degrade user trust.

---

## 5. SubQ Decision Model 2 Integration & 10-10 Parallel Batching

### Decision
Integrate Decision Model 2 with the **SubQ Decisions API**:
- **Endpoint**: `https://api.subq.ai/v1/systemone`
- **Model**: `subq-snap-preview`
- **Context Window**: 384K tokens (393,216)
- **API Key**: configured via `SUBQ_API_KEY` environment variable
- **Question Schema**: Per-turn `choice` question with 4 options:
  - `not_needed_not_relevant`
  - `not_needed_relevant`
  - `needed_not_relevant`
  - `needed_relevant`

### Batching & Chunking Strategy
- The SubQ Decisions API enforces a limit of **1 to 16 named questions per single API request**.
- Because our candidate history window evaluates up to the last **20 Q&A pairs**, a single request cannot accommodate 20 named questions.
- **Solution**: The candidate turns are split into two parallel batches of up to 10 questions each (`turn_0`..`turn_9` and `turn_10`..`turn_19`) executed concurrently via `asyncio.gather`.
- This ensures compliance with SubQ API limits while maintaining an ultra-low parallel execution latency of ~200-250ms.

### Alternatives Considered
- *Sequential calls for each turn (20 requests)*: Massive network round-trip overhead (>3.0s latency).
- *Single-question multi-select returning comma-separated IDs*: Prone to formatting errors and loses calibrated probabilities and confidence scores provided by SubQ.

---

## 6. Deterministic Missing Summary Backfill

### Decision
1. When history evaluation is triggered, the system deterministically inspects the last up to 20 Q&A turns in the session store.
2. If any turn lacks `q_and_a_summary` in its metadata, all missing pairs are collected and sent in a **single structured batch LLM call** to generate their summaries.
3. The generated summaries are immediately persisted back to the session store under `q_and_a_summary` before invoking Decision Model 2.
4. **Timeout Guard**: If the backfill call takes longer than 2.0s or fails, the system safely falls back to using the first 250 characters of the raw answer text as a pseudo-summary, preventing any stall on the student's turn.

### Rationale
- Steady-state turns generate their summaries incrementally during response synthesis.
- On-demand backfill handles cold cache starts, migrated sessions, or system restarts without requiring manual batch migrations.

---

## 7. Context Fidelity Selection & No Artificial Cap

### Decision
For all historical turns classified by Decision Model 2 as `needed_relevant` or `needed_not_relevant`:
- **No artificial limit** is imposed on the number of selected turns.
- **Recent-5 Message Rule**: Any needed turn among the **5 most recent turns** of the session is injected in **full text** (both full Question and full Answer).
- **Older Turns Rule**: Any needed turn from older history (turns 6 through 20) is injected as its **concise summary**.

### Rationale
- Immediate conversational turns require verbatim fidelity to capture specific formulas, numbers, or step-by-step reasoning.
- Older turns only require thematic and conceptual grounding, where a concise summary saves token budget without losing context.

---

## 8. Incremental Summary Caching in Response Synthesis

### Decision
When response synthesis executes with history enabled:
- The synthesizer produces both the final Persian grounded response and a structured summary (approx. 1 paragraph, <= 6 lines, max 512 tokens).
- The summary is returned in the result payload and persisted under `turn.metadata["q_and_a_summary"]`.

### Rationale
- Amortizes the cost of summarization into runtime generation.
- Eliminates on-the-fly backfill overhead on subsequent conversational turns.

---

## 9. Qwen-Style UI Reply & Manual Context Selection

### Decision
When conversation history is toggled OFF:
1. Each message card in the chat UI displays an action icon: **«پاسخ به این پیام / انتخاب به عنوان زمینه»** (Reply / Attach as Context).
2. Clicking the button mounts a floating, dismissible quote preview chip immediately above the input textarea with the message preview, an icon badge, and a **✕** cancel button.
3. Only **one** past Q&A pair can be attached at a time (selecting another replaces the active chip).
4. When sent, the request payload includes `selected_context_turn_id`.
5. The backend pipeline treats this manually selected turn with **direct inclusion**, bypassing Decision Model 2 and injecting the turn's text directly into the synthesis context.

### Rationale
- Inspired by modern chatbot interfaces (like Qwen Chat), this gives students fine-grained control over context injection in private/stateless mode without needing to enable full automatic memory tracking.

---

## 10. Citation Attribution for `needs_history_only`

### Decision
When retrieval is bypassed for meta-inquiries (`needs_history_only`):
- No fresh textbook citations are returned (`citations: []`).
- The UI displays a dedicated conversational context badge: **«پاسخ بر اساس گفتگوی پیشین»** instead of chapter/page citation badges.

### Rationale
- Prevents confusing students with irrelevant page citations when an answer is merely rephrasing or summarizing prior conversation text.
- Maintains high UI clarity and transparency.

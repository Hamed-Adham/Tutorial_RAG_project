# Feature Specification: QueryGate & Conversational Memory Routing (006)

**Feature Branch**: `006-query-gate-conversational-memory`

**Created**: 2026-10-14  
**Last Updated**: 2026-10-14  

**Status**: Ready for Planning

**Input**: User description: "extend gated_rewriter to query_gate with 3-tier irrelevancy/refusal, 4-tier need_conv_hist_and_retrieve flag, Decision Model 2 for history Q&A relevance filtering across up to 20 turns with missing summary backfill, hybrid retrieval injection, incremental summary generation in synthesis, recent-5 full Q&A vs older summary selection, manual single-turn context selection when history is toggled off, and SubQ Decisions API integration."

---

## Clarifications

### Session 2026-10-14
- Q: What should be the maximum number of historical Q&A turns injected into the context for synthesis and retrieval? (FR-013) → A: No artificial limit. All turns classified as needed by Decision Model 2 are included. For any needed turns among the 5 most recent pairs, include the full Query and Answer text; for older needed turns (turns 6–20), include their stored summary.
- Q: How should the system handle Turn 0 when history is empty? (FR-006) → A: Auto-coerce Option 1 (`needs_history_and_retrieve`) to Option 3 (`needs_retrieve_only`) and Option 2 (`needs_history_only`) to a friendly clarification prompt ("هنوز گفتگویی انجام نشده است، لطفاً پرسش خود را مطرح کنید").
- Q: How should citations appear for `needs_history_only` queries? (FR-008) → A: Return empty citations (`citations: []`) and display a distinct conversational context badge: "پاسخ بر اساس گفتگوی پیشین".
- Q: How should SubQ Decision Model 2 be structured for up to 20 candidate turns? (FR-012) → A: Parallel 10-10 chunking. Split candidate turns into two concurrent requests of up to 10 questions each executed via `asyncio.gather`.
- Q: How does manual context selection work when history is toggled OFF? (FR-016, FR-017) → A: Qwen-style message card Reply button with a dismissible floating preview chip above the input box. Bypasses Decision Model 2 and provides direct antecedent inclusion.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 3-Tier Curriculum Intent Gating & Partial Query Distillation (Priority: P1)

A student enters a query that is either completely irrelevant (e.g. chit-chat, prompt injection, sports), partially relevant (e.g. "امروز بارونیه ولی بگو بطن‌های قلب چیستند؟"), or cleanly relevant. The primary entry gate (`query_gate`) classifies the intent into three distinct tiers:
1. `completely_irrelevant`: Immediately rejected with a polite refusal without wasting vector retrieval or synthesis tokens.
2. `partially_relevant`: The system identifies curriculum-relevant sub-concepts, prompts the query transformer to discard irrelevant noise, and forwards only the purified scientific inquiry before any history evaluation or retrieval occurs.
3. `relevant`: Directly approved for downstream routing.

**Why this priority**: Prevents retrieval poisoning, protects system boundaries against adversarial injection, and saves token budget by eliminating irrelevant queries before retrieval.

**Independent Test**: Send synthetic pure, mixed, and adversarial queries to `QueryGate` and assert exact 3-tier classification and transformed clean strings.

**Acceptance Scenarios**:
1. **Given** a query with mixed intent ("هوا خوبه ولی چگالی چوب چقدره؟"), **When** evaluated by `QueryGate`, **Then** `intends_curriculum_answer` resolves to `partially_relevant`, the query transformer drops "هوا خوبه ولی", and the effective query forwarded is "چگالی چوب چقدره؟".
2. **Given** a completely out-of-scope query ("نظرت درباره بازی استقلال و پرسپولیس چیه؟"), **When** evaluated, **Then** `intends_curriculum_answer` resolves to `completely_irrelevant`, retrieval is bypassed, and an immediate domain refusal is returned.
3. **Given** a direct textbook query ("ساختار یاخته گیاهی چیست؟"), **When** evaluated, **Then** `intends_curriculum_answer` resolves to `relevant` and proceeds without truncation.

---

### User Story 2 - Context Need Dispatching (need_conv_hist_and_retrieve) (Priority: P1)

For every approved query (partially relevant or relevant), `QueryGate` simultaneously evaluates whether answering requires conversation history, external textbook retrieval, both, or neither across 4 exhaustive tiers:
- Tier 1: `needs_history_and_retrieve` (requires prior context to disambiguate pronouns AND retrieval from textbook)
- Tier 2: `needs_history_only` (meta-inquiry explaining, formatting, or simplifying the previous answer; NO textbook retrieval needed)
- Tier 3: `needs_retrieve_only` (standalone scientific question; fresh textbook retrieval required, NO history needed)
- Tier 4: `needs_neither` (conversational pleasantries or non-retrieval chit-chat; immediate acknowledgment/refusal)

**Why this priority**: Drastically cuts latency and database load by bypassing heavy vector search and cross-encoder rerankers when the user merely asks to simplify or reformat the prior answer (`needs_history_only`).

**Independent Test**: Send "پاسخ قبلی را خلاصه کن" and verify that Qdrant/BM25 retrieval latency is 0ms while the previous turn's context is supplied to synthesis and the response is badged as conversational context without textbook citations.

**Acceptance Scenarios**:
1. **Given** a follow-up question referencing an antecedent ("آزمایشی که توضیح دادی چه خطاهایی دارد؟"), **When** evaluated by `QueryGate`, **Then** `need_conv_hist_and_retrieve` resolves to `needs_history_and_retrieve`, triggering history selection and textbook retrieval.
2. **Given** a request to rephrase ("همین را ساده‌تر توضیح بده"), **When** evaluated, **Then** `need_conv_hist_and_retrieve` resolves to `needs_history_only`, bypassing dense/sparse retrieval entirely, generating the simplified explanation from session context, and displaying a "پاسخ بر اساس گفتگوی پیشین" badge with no textbook citations.
3. **Given** a fresh standalone inquiry ("قانون پایستگی انرژی چیست؟"), **When** evaluated, **Then** it resolves to `needs_retrieve_only`, bypassing history selection and retrieving only for the new topic.

---

### User Story 3 - Decision Model 2: History Relevance Filtering Across 20 Q&As (Priority: P2)

When a query requires conversation history (`needs_history_and_retrieve` or `needs_history_only`), the system inspects up to the last 20 Q&A turns of the active session.
1. **Deterministic Summary Check & Single Backfill Call**: The system deterministically checks the existence of `q_and_a_summary` for each of the last 20 candidate Q&A pairs. If any pair lacks a summary, it invokes a single structured batch LLM summarization call for all missing pairs and persists them in the session store before invoking Decision Model 2.
2. **Decision Model 2 Parallel SubQ Execution (10-10 Split)**: Calls the SubQ Decisions API (`https://api.subq.ai/v1/systemone` using model `subq-snap-preview`). When evaluating up to 20 candidate turns, the system chunks the candidate set into two parallel batches of up to 10 named questions each (e.g. `turn_0`..`turn_9` and `turn_10`..`turn_19`) executed concurrently via `asyncio.gather`. For each historical Q&A pair, it evaluates the 4-tier rubric:
   - `not_needed_not_relevant`
   - `not_needed_relevant`
   - `needed_not_relevant`
   - `needed_relevant`
3. **Fidelity Context Selection**: For each turn classified as `needed_relevant` or `needed_not_relevant`:
   - If the turn is within the **5 most recent turns**: include the **full Q&A text** (both Question and full Answer) in the synthesis prompt.
   - If the turn is older than the 5 most recent turns (turns 6-20): include its **concise summary**.

**Why this priority**: Eliminates context window pollution while preserving full fidelity for immediate context and token-efficient summaries for distant context.

**Independent Test**: In a 10-turn session discussing Chapter 2 (density) followed by Chapter 14 (heart), ask a question about density. Verify that only Chapter 2 turns are marked as "needed", ignoring Chapter 14 turns.

**Acceptance Scenarios**:
1. **Given** a session with 10 turns and an antecedent query ("فرمول آن چیست؟"), **When** `Decision Model 2` runs, **Then** only the specific turn defining the target concept is flagged as `needed_relevant`, and unrelated turns are excluded.
2. **Given** historical turns lacking summaries (e.g. legacy turns or worker restart), **When** inspected prior to `Decision Model 2`, **Then** missing pairs are detected, batched into a single LLM call, summarized, and cached in session storage.
3. **Given** a needed turn from Turn N-1 and Turn N-7, **When** passed to synthesis, **Then** Turn N-1 includes full text, and Turn N-7 includes summary text.

---

### User Story 4 - Hybrid Retrieval Augmentation with Needed Summaries (Priority: P2)

When `need_conv_hist_and_retrieve` is `needs_history_and_retrieve`, the retrieval module augments the search query by combining the purified user query with the summaries of the selected historical Q&A turns. Both dense embeddings (BGE-M3) and sparse tokenizers (BM25) receive this enriched query.

**Why this priority**: Guarantees that vector search and BM25 retrieve the exact right textbook pages even when the user's raw query was just a single sentence with context references.

**Independent Test**: Assert that a query like "در این فرایند نقش کبد چیست؟" retrieves digestive system chunks (Chapter 13) when the previous turn established digestion, rather than blood circulation chunks.

**Acceptance Scenarios**:
1. **Given** a transformed query "نقش کبد چیست؟" and a needed summary about digestive enzymes, **When** passed to hybrid retrieval, **Then** candidate chunks from Chapter 13 (سفر غذا) receive top RRF ranks.

---

### User Story 5 - Incremental Turn Summary Caching in Response Synthesis (Priority: P3)

When generating a response in an active conversation where history is enabled, the synthesizer produces both the final grounded answer and a structured summary of the current Q&A pair (approx. 1 paragraph, <= 6 lines, max 512 tokens). This summary is stored under `q_and_a_summary` in the session turn model.

**Why this priority**: Amortizes summarization cost to runtime generation, eliminating future on-the-fly backfill delays.

**Independent Test**: Inspect the session store after a turn and verify that `turn.metadata["q_and_a_summary"]` is populated with a clean Persian paragraph.

**Acceptance Scenarios**:
1. **Given** an answer generated by `ResponseSynthesizer`, **When** the session store commits the turn, **Then** `q_and_a_summary` is saved alongside `citations` and `latencies`.

---

### User Story 6 - Qwen-Style UI Reply & Context Selection When History Is Toggled Off (Priority: P3)

When conversation history is toggled OFF, users still occasionally want to reference a specific past message. Inspired by the reply/quote pattern in modern interfaces like Qwen Chat:
1. Each message card in the chat UI displays a subtle "پاسخ به این پیام / انتخاب به عنوان زمینه" (Reply / Attach as Context) action button alongside copy and reaction icons.
2. Clicking this action displays a compact, floating dismissible quote chip immediately above the message input box with a snippet of the referenced message, an icon indicator ("زمینه انتخابی"), and a clear button ("✕").
3. Only **one** past Q&A pair can be selected at a time (selecting another message replaces the currently attached chip).
4. When the user sends a message with an attached context chip, the client transmits `selected_context_turn_id`.
5. The backend pipeline treats this manually selected turn with **direct inclusion**, bypassing Decision Model 2 and injecting the selected turn directly as antecedent context for synthesis.

**Why this priority**: Gives students full control over context injection in stateless/privacy mode without requiring automatic multi-turn tracking.

**Independent Test**: In the UI, toggle history OFF, click the Reply button on message 3, verify the quote chip appears above the prompt textarea, submit a question, and verify the backend synthesis prompt includes Turn 3 context without invoking Decision Model 2.

**Acceptance Scenarios**:
1. **Given** history toggle is OFF, **When** a user clicks the Reply/Context button on a past message, **Then** a dismissible quote preview chip appears above the chat input box.
2. **Given** an attached quote chip, **When** the user clicks the "✕" button, **Then** the chip is cleared and the query is submitted as a pure single turn.
3. **Given** an attached past message, **When** the user sends their query, **Then** the backend bypasses Decision Model 2 and injects the selected Q&A directly into the synthesis prompt.

---

### Edge Cases & Invariants

- **Cold-Start on Turn 0**: When a session has 0 previous turns, if the classifier predicts `needs_history_and_retrieve`, the system auto-coerces it to `needs_retrieve_only`. If it predicts `needs_history_only`, it returns a friendly clarification prompt asking the student to state their question.
- **Citation Attribution for `needs_history_only`**: When retrieval is bypassed, no textbook citations are generated (`citations: []`). Instead, the UI displays a distinct conversational badge: "پاسخ بر اساس گفتگوی پیشین" to signal that the answer is derived from conversation history rather than freshly retrieved pages.
- **Decision Model 2 SubQ 10-10 Request Chunking**: To respect the SubQ Decisions API limit (up to 16 named questions per request) while evaluating up to 20 historical turns, the system chunks the candidate set into two parallel batches of up to 10 questions each, executed concurrently via `asyncio.gather`.
- **Missing Summary Backfill Timeout Guard**: If the batch backfill LLM call takes longer than 2.0s or fails, the system safely falls back to using the first 250 characters of the raw answer text as a pseudo-summary to prevent stalling user queries.
- **Zero Needed Summaries Selected**: If `Decision Model 2` classifies all historical turns as `not_needed`, the system gracefully proceeds with the transformed query alone.
- **Session History Length < 20**: When a session has fewer than 20 messages (e.g. 3 turns), `Decision Model 2` evaluates only the actual available turns without padding.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST rename all occurrences of `gated_rewriter` to `query_gate` across modules, class names (`QueryGate`), filenames (`query_gate.py`), and imports.
- **FR-002**: `QueryGate` MUST evaluate `intends_curriculum_answer` as a 3-tier choice: `completely_irrelevant`, `partially_relevant`, and `relevant`.
- **FR-003**: If `intends_curriculum_answer` is `completely_irrelevant`, the pipeline MUST halt execution and return a polite refusal without invoking retrieval or response synthesis.
- **FR-004**: If `intends_curriculum_answer` is `partially_relevant`, `QueryGate` MUST invoke query transformation to remove out-of-scope content and emit an effective, cleaned curriculum query before any retrieval or history evaluation occurs.
- **FR-005**: `QueryGate` MUST evaluate `need_conv_hist_and_retrieve` as a 4-tier choice: `needs_history_and_retrieve`, `needs_history_only`, `needs_retrieve_only`, and `needs_neither`.
- **FR-006**: On Turn 0 (empty session), if `need_conv_hist_and_retrieve` is `needs_history_and_retrieve`, it MUST be coerced to `needs_retrieve_only`; if `needs_history_only`, it MUST return a prompt requesting the question without retrieval.
- **FR-007**: If `need_conv_hist_and_retrieve` is `needs_neither`, the pipeline MUST reject/halt without running retrieval or history expansion.
- **FR-008**: If `need_conv_hist_and_retrieve` is `needs_history_only`, the pipeline MUST skip hybrid dense/sparse retrieval and cross-encoder reranking entirely, passing the selected history turns directly to the synthesizer, setting citations to empty, and returning a conversational context badge ("پاسخ بر اساس گفتگوی پیشین").
- **FR-009**: The system MUST proceed to `Decision Model 2` ONLY WHEN `intends_curriculum_answer` in (`partially_relevant`, `relevant`) AND `need_conv_hist_and_retrieve` in (`needs_history_and_retrieve`, `needs_history_only`).
- **FR-010**: Before invoking `Decision Model 2`, the system MUST inspect up to the last 20 Q&A pairs in the active session. If any turn lacks `q_and_a_summary`, it MUST execute a single structured batch LLM call to generate summaries for all missing pairs and persist them in session storage.
- **FR-011**: `Decision Model 2` MUST use SubQ Decisions API (`POST https://api.subq.ai/v1/systemone`, model `subq-snap-preview`, API Key configured via `SUBQ_API_KEY`) and evaluate each candidate Q&A across: `not_needed_not_relevant`, `not_needed_relevant`, `needed_not_relevant`, and `needed_relevant`.
- **FR-012**: If the candidate set exceeds 10 turns (up to 20), the system MUST split the evaluation into two concurrent SubQ requests (up to 10 questions each) using `asyncio.gather`.
- **FR-013**: For all historical turns classified as `needed_relevant` or `needed_not_relevant` (with no artificial numerical limit on total selected turns):
  - If within the 5 most recent turns: include full Q&A text.
  - If older than 5 turns (turns 6–20): include summary text.
- **FR-014**: If `need_conv_hist_and_retrieve` is `needs_history_and_retrieve`, the selected Q&A summaries MUST be combined with the transformed user query and fed into hybrid retrieval (Qdrant + BM25).
- **FR-015**: In response synthesis, when conversation history is enabled, the generation stage MUST produce both the final grounded answer and a structured summary of the current Q&A (maximum 512 tokens / ~6 lines) to be saved in session storage under `q_and_a_summary`.
- **FR-016**: When history is toggled OFF, the UI MUST provide a Qwen-style "Reply" / "Select as Context" button on message cards with a dismissible preview chip above the input box, allowing the user to select at most ONE prior Q&A pair.
- **FR-017**: If a user submits a query with a manually selected past turn, the system MUST bypass Decision Model 2 and directly inject the selected turn's text as antecedent context.
- **FR-018**: Every decision from `QueryGate` and `Decision Model 2` MUST be logged with decision metadata in Langfuse traces.

---

### Key Entities

- **`CurriculumIntentTier`**: Enum (`COMPLETELY_IRRELEVANT`, `PARTIALLY_RELEVANT`, `RELEVANT`).
- **`ContextNeedTier`**: Enum (`NEEDS_HISTORY_AND_RETRIEVE`, `NEEDS_HISTORY_ONLY`, `NEEDS_RETRIEVE_ONLY`, `NEEDS_NEITHER`).
- **`HistoryRelevanceTier`**: Enum (`NOT_NEEDED_NOT_RELEVANT`, `NOT_NEEDED_RELEVANT`, `NEEDED_NOT_RELEVANT`, `NEEDED_RELEVANT`).
- **`QASummaryRecord`**: Structured entity containing `turn_id`, `session_id`, `question_text`, `answer_text`, `summary_text` (max 512 tokens), and `timestamp`.
- **`QueryGateDecision`**: Comprehensive outcome model containing `intent_tier`, `context_need_tier`, `effective_query`, `selected_history_items`, `bypass_retrieval`, and `latency_ms`.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `query_gate` evaluation latency (Model 1) MUST complete in under 25ms when using local Jev System One gateway.
- **SC-002**: `needs_history_only` queries (meta-questions, rephrasing, simplification) MUST achieve a 70%+ reduction in end-to-end response latency (from ~1.8s down to <600ms) by bypassing vector search and reranking.
- **SC-003**: 100% of out-of-scope adversarial and chitchat queries are intercepted at Gate 1 with 0 vector retrieval calls and 0 prompt tokens consumed by the synthesis LLM.
- **SC-004**: For multi-turn conversational inquiries, relevant context precision on follow-up questions improves by at least 15% compared to naive raw query retrieval.
- **SC-005**: Incremental Q&A summary generation adds no more than 150ms to streaming TTFT / synthesis overhead.
- **SC-006**: SubQ Decision Model 2 evaluations complete within 350ms across the full conversational candidate set.

---

## Assumptions & Design Invariants

- **SubQ Decisions API Compatibility**: The SubQ endpoint (`https://api.subq.ai/v1/systemone`) supports `choice` questions with `subq-snap-preview` and a 384K context window.
- **Session Store Persistence**: `SessionStore` (both `InMemorySessionStore` and `RedisSessionStore`) supports persisting metadata dictionaries with `q_and_a_summary` alongside dialogue turns.
- **Graceful Degradation**: If SubQ or Jev is unreachable or times out (>2.5s), the system degrades safely to standalone direct retrieval without failing the user's request.

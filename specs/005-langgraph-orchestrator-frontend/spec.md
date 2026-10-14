# Feature Specification: LangGraph Orchestrator & Modern Next.js Chatbot UI

**Feature Branch**: `005-langgraph-orchestrator-frontend`  
**Created**: 2026-10-14  
**Status**: Approved  
**Input**: User description: "LangGraph orchestrator and Next.js frontend."

---

## Clarifications

### Session 2026-10-14
- Q: How should user chat session history be preserved across browser reloads? (FR-008) → A: Browser LocalStorage with automatic backend session synchronization, ensuring sessions remain available locally while syncing turns to backend session stores.
- Q: What should be the default curriculum scope when a student opens a new chat session? (FR-009) → A: Open global textbook search across all 15 chapters by default, with an optional interactive chapter scope selector for focused practice.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Interactive Conversational Science Learning Interface with Persian RTL (Priority: P1)

As an educational user, 7th-grade science student, or teacher,  
I want a clean, responsive conversational web interface that natively displays Persian text (RTL) with rich markdown formatting, real-time progressive streaming responses, and accessible conversation history,  
So that I can ask scientific questions, read answers progressively without delay, and review previous questions and answers naturally.

**Why this priority**: Core user touchpoint. Without a responsive, native RTL conversational UI, the underlying RAG pipeline is inaccessible to students and educators.

**Independent Test**: Can be fully tested by submitting curriculum inquiries via the web interface, observing sub-second token streaming, rich markdown rendering, and switching between conversational sessions in the sidebar.

**Acceptance Scenarios**:

1. **Given** a student on desktop or mobile, **When** they load the application, **Then** the interface presents a Persian RTL layout with an active chat area, quick suggestion curriculum prompts, and a collapsible session sidebar.
2. **Given** a user submitting a science question, **When** generation begins, **Then** text streams progressively onto the screen with initial tokens visible in under 1.5 seconds.
3. **Given** an ongoing conversation, **When** the user clicks "New Chat", **Then** a fresh dialogue session is initiated while preserving the previous session in the history list.

---

### User Story 2 - Scoped Curriculum Practice vs. Global Open Exploration (Priority: P2)

As a student preparing for chapter-specific exams or daily homework,  
I want to easily switch between targeting specific textbook chapters (Mode 1: Scoped Practice) and asking broad questions across the entire curriculum (Mode 2: Global Search),  
So that my practice is strictly restricted to current class material when studying a specific unit, while allowing holistic connections across all 15 chapters when doing comprehensive revision.

**Why this priority**: Critical educational requirement. Students frequently study specific units (e.g. Chapter 2: Measurement, Chapter 6: Water on Earth); cross-chapter contamination during unit prep confuses learners.

**Independent Test**: Can be tested by opening the Chapter Scope modal, selecting specific chapters (e.g., Chapter 1 and Chapter 2), submitting queries, and verifying that the system restricts evidence to those chapters.

**Acceptance Scenarios**:

1. **Given** a student studying Chapter 2 (*اندازه‌گیری فيزيك*), **When** they select Chapter 2 in the scope selector and ask a question about units of volume, **Then** retrieval and response generation strictly reference Chapter 2.
2. **Given** a chapter-scoped session, **When** the student asks a question unrelated to the selected chapter, **Then** the system notifies the student that the question lies outside the active chapter scope.
3. **Given** a user who deselects all specific chapters, **When** they submit a multi-chapter query, **Then** the system automatically searches across all 15 textbook chapters.

---

### User Story 3 - Evidence Verification, Citations Drawer & Latency Transparency (Priority: P3)

As a student or educator validating AI-generated answers against official school curriculum,  
I want every scientific explanation to display verified textbook page badges, and provide an expandable drawer detailing the source excerpts, relevance confidence, and processing latency waterfall,  
So that I can cross-reference the physical textbook with confidence and verify system responsiveness.

**Why this priority**: Builds pedagogical trust, eliminates hallucinations, and provides transparency into how the AI arrived at its conclusions.

**Independent Test**: Can be tested by generating an answer, clicking the textbook citation badges, and viewing the detailed evidence drawer with page numbers and execution stage timing.

**Acceptance Scenarios**:

1. **Given** an answer generated from the science curriculum, **When** the message renders, **Then** verified citation badges appear referencing the specific chapter and page (e.g. `[کتاب علوم هفتم (ص ۴)]`).
2. **Given** a rendered citation badge, **When** the user clicks the badge or evidence button, **Then** a drawer opens showing the retrieved source excerpts, chapter breadcrumbs, and matching relevance scores.
3. **Given** any completed response, **When** the user inspects the latency indicator, **Then** a stage breakdown shows exact millisecond durations for query understanding, retrieval, reranking, and synthesis.

---

### User Story 4 - Declarative Graph Orchestration with Grounded Refusal & Observability (Priority: P4)

As an educational AI system administrator,  
I want the end-to-end science pipeline to execute through a stateful, declarative graph workflow that enforces early deterministic refusal on out-of-scope inquiries and logs comprehensive execution traces,  
So that out-of-domain questions bypass expensive search stages with zero token overhead, and production telemetry is monitored continuously.

**Why this priority**: Essential for operational cost efficiency, resilience, and maintaining strict educational boundaries.

**Independent Test**: Can be tested by submitting out-of-scope questions (e.g., cryptocurrency, programming) and verifying that the graph branches immediately to refusal without initiating vector searches or generating hallucinations.

**Acceptance Scenarios**:

1. **Given** an inquiry completely unrelated to 7th-grade science, **When** the orchestrator analyzes the intent, **Then** execution branches directly to a polite refusal response, bypassing retrieval and neural reranking.
2. **Given** a valid curriculum inquiry, **When** the graph executes, **Then** state transitions smoothly across query analysis, hybrid retrieval, neural reranking, parent expansion, and citation synthesis.
3. **Given** an administrator monitoring system health, **When** they inspect the live metrics dashboard, **Then** real-time indicators display active requests, queue delay, circuit breaker status, and response latency percentiles.

---

### Edge Cases

- **Network Interruption During Streaming**: If the client disconnects or loses connectivity while tokens are streaming, the server must terminate upstream generation tasks promptly without leaking resources.
- **Ambiguous or Mixed Inquiries**: If a query contains both a valid science question and unrelated chitchat, the system addresses the scientific concept while gently ignoring the irrelevant portion.
- **Empty or Whitespace Prompts**: The interface must disable message dispatch when the input contains only whitespace.
- **Rapid Multi-Message Submissions**: If a user submits a second question before the first completes, the system queues or handles the request according to active concurrency limits without corrupting conversation context.
- **Small Mobile Viewport**: The session sidebar, chapter scope modal, and citations drawer must collapse or adapt smoothly to narrow screen widths without horizontal overflow.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The interface MUST provide native Right-to-Left (RTL) Persian alignment, typography, and responsive layouts for desktop and mobile viewports.
- **FR-002**: The interface MUST support progressive text streaming via real-time events, displaying the first response token within 1.5 seconds.
- **FR-003**: The interface MUST provide an interactive chapter scope selector covering all 15 chapters of the 7th-grade science textbook, supporting both scoped practice (Mode 1) and global search (Mode 2).
- **FR-004**: Generated responses MUST display verified citation badges indicating the source textbook chapter and page number.
- **FR-005**: The interface MUST include an evidence inspection drawer allowing users to review raw source text, relevance scores, and metadata for cited passages.
- **FR-006**: The interface MUST include a latency indicator detailing execution durations across discrete processing stages (query understanding, search, reranking, synthesis).
- **FR-007**: The interface MUST provide a live operational telemetry view showing active request volume, queue delay, and system health status.
- **FR-008**: The interface MUST persist conversation sessions across browser refreshes via browser LocalStorage with automatic backend session synchronization.
- **FR-009**: When initiating a new chat session, the system MUST default to open global textbook search across all 15 chapters, with an optional interactive chapter scope selector for focused practice.
- **FR-010**: The processing graph MUST deterministically refuse out-of-scope or adversarial queries with structured refusal metadata, bypassing search and reranking stages.
- **FR-011**: The system MUST support multi-turn dialogue memory, preserving context across consecutive turns to resolve pronouns and follow-up questions.

---

### Key Entities

- **Learning Session**: Represents an interactive conversational dialogue containing an identifier, title, creation timestamp, and active curriculum scope.
- **Dialogue Turn**: A single conversational exchange comprising the user query, assistant response, cited passages, latency metrics, and timestamp.
- **Curriculum Scope**: Configuration specifying whether retrieval is constrained to specific chapter numbers (Mode 1) or open to all 15 chapters (Mode 2).
- **Textbook Citation**: Provenance record binding an answer claim to a specific chapter title, page number, and original textbook snippet.
- **Graph Pipeline State**: Stateful execution dictionary carrying query transformations, active filters, candidate chunks, reranked passages, expanded parent contexts, and citation links.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 95% of curriculum questions exhibit initial visible response tokens on screen in under 1.5 seconds from user submission.
- **SC-002**: 100% of factual science answers include at least one verified textbook page citation linking back to the source curriculum.
- **SC-003**: 100% of out-of-scope inquiries are politely refused without generating ungrounded claims or executing unnecessary search calls.
- **SC-004**: Users can select and apply a curriculum chapter scope in 3 clicks or fewer from the main chat view.
- **SC-005**: The interface renders without layout distortion, text clipping, or overflow across viewports from 375px mobile to 1920px desktop.
- **SC-006**: Conversational session switching, creation, and deletion execute instantaneously with zero page reloads.

---

## Assumptions

- Users communicate in Persian (Farsi) and access the platform through modern evergreen web browsers (Chrome, Firefox, Safari, Edge).
- The underlying 15-chapter Iranian 7th-grade science textbook knowledge base and multi-stage RAG endpoints are operational.
- Audio and speech input are out of scope for the current version; interaction is text-based.
- User authentication and role-based access control are not required for this educational demonstration interface; sessions are managed per client instance.

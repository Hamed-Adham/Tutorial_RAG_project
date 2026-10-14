# Research & Technical Decisions: LangGraph Orchestrator & Next.js Chatbot UI

**Feature Branch**: `005-langgraph-orchestrator-frontend`  
**Specification**: [spec.md](spec.md)  
**Date**: 2026-10-14  

---

## 1. Declarative Graph Orchestration: LangGraph StateGraph vs. Linear Pipeline

### Decision
Compile the multi-stage science textbook RAG workflow into a declarative LangGraph `StateGraph` (`EnterpriseScienceGraph`) featuring conditional branching for early refusal, token streaming, and native Langfuse `CallbackHandler` integration.

### Rationale
* **Conditional Branching without Overhead**: Out-of-curriculum and adversarial inquiries should never incur vector retrieval or cross-encoder latency. A declarative edge (`should_enterprise_refuse`) routes directly from `gated_rewrite` to `refusal -> END` in < 15ms.
* **LangChain Ecosystem Interoperability**: Conforms to standard `Document` and `BaseRetriever` protocols, making the retriever swappable and auditable.
* **Granular Traceability**: Native callback integration automatically nests node-level spans inside Langfuse root traces.

### Alternatives Considered
* *Hardcoded Class Method Calls (`AdvancedRAGPipeline.query`)*: Simple and effective, but lacks visual inspection, declarative state checkpointing, and graph-level callback lifecycle hooks.
* *LlamaIndex Workflow*: Lacks seamless first-class Langfuse OpenInference tracing and requires extra adapter glue.

---

## 2. Frontend Framework: Next.js 14 App Router vs. Pure Vite SPA

### Decision
Implement the chatbot interface using Next.js 14 (App Router) with Tailwind CSS, Lucide icons, and server rewrites (`next.config.js`).

### Rationale
* **Production-Grade Architecture**: App Router provides clean directory separation, static route pre-rendering (`Route /: 57.6 kB`), and built-in API proxy rewrites to `http://127.0.0.1:8000/api/v1/` eliminating CORS friction in development.
* **Modern UI Patterns**: Inspired by McKay Wrigley's `chatbot-ui`, featuring collapsible session drawers, auto-scrolling message streams, and accessible dark/light modes.
* **Asset Optimization**: Built-in support for serving local WOFF2 font files from `/public/fonts/` with zero layout shift.

### Alternatives Considered
* *Pure React + Vite SPA*: Lightweight, but requires external dev-server proxy configuration, lacks server-rendered metadata optimization, and deviates from enterprise frontend conventions in Constitution Principle I.

---

## 3. Typography: Peyda (پیدا) vs. Generic Web Fonts for Persian RTL

### Decision
Incorporate official **Peyda (پیدا)** font weights (Regular 400, Medium 500, Bold 700) as local WOFF2 assets in `public/fonts/` with RTL-calibrated line height (`line-height: 1.7`).

### Rationale
* **Curriculum Readability**: Iranian 7th-grade science textbook passages contain diacritics, technical terms, and complex Persian compound words. Peyda delivers superior x-height and letter-spacing balance compared to generic system sans-serif.
* **Zero Dependency on External CDNs**: Self-hosting local WOFF2 assets guarantees offline capability, eliminating latency or CDN blocking issues inside restricted networks.

### Alternatives Considered
* *Vazirmatn via Google Fonts CDN*: Clean, but relies on external network requests during initial page render.
* *System Fallback Font Stack (`Tahoma`, `Arial`)*: Highly inconsistent across macOS, Windows, and Linux; poor typography for educational content.

---

## 4. Real-Time Token Delivery: Server-Sent Events (SSE) vs. WebSockets

### Decision
Standardize on Server-Sent Events (SSE) via `POST /api/v1/chat/stream` emitting `event: token` and `event: done` frames.

### Rationale
* **Sub-Second TTFT**: Emits tokens progressively with TTFT < 1.0s while maintaining standard HTTP request/response semantics.
* **Stateless Gateway Routing**: SSE operates over HTTP/1.1 or HTTP/2 without maintaining stateful persistent socket connections on the gateway, easily scaling across 50 concurrent users.
* **Native Client Cancellation**: Web `fetch` with `AbortController` cleanly signals disconnection to FastAPI/Starlette, immediately terminating upstream model generation.

### Alternatives Considered
* *WebSockets*: Bi-directional communication is unnecessary since clients submit a single inquiry per turn and only read downward streams. Adds socket reconnection and proxy complexity.
* *Polling REST*: Unacceptable latency and excessive server load for conversational AI.

---

## 5. Session State Synchronization: Browser LocalStorage + Redis Backend

### Decision
Persist conversation session metadata and messages in browser `LocalStorage` (`unified_rag_sessions`, `unified_rag_messages`) with automatic backend Redis synchronization on every message exchange.

### Rationale
* **Zero Initial Load Latency**: Chat history and active session selections restore instantly from browser cache on page refresh.
* **Backend Sliding-Window Continuity**: The backend `RedisSessionStore` independently maintains the last 4 turns (~600 tokens) for pronoun antecedent resolution (*آن، این*), with 2-hour TTL eviction.

### Alternatives Considered
* *Pure In-Memory Frontend*: Loses conversation history whenever the user reloads the browser tab.
* *Database-Only Query on Every Navigation*: Adds blocking network round-trips for simple sidebar session switching.

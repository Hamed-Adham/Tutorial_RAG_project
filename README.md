# Unified RAG — Enterprise Persian Science Pipeline & Evaluation Cockpit

An enterprise-grade, multi-stage Retrieval-Augmented Generation (RAG) system tailored for the Persian 7th-grade science curriculum (*علوم تجربی پایه هفتم*). Built with **LangGraph orchestrator**, **TypeSafe Jev & SubQ System One decision models**, **hybrid dense + sparse retrieval**, **neural reranking**, **strict grounded citations**, and an interactive **Next.js evaluation and chat cockpit**.

---

## 🌟 Key Features

- **Pre-Retrieval Intent Gating (QueryGate)**: 
  - 3-tier curriculum intent classification (`relevant`, `partially_relevant`, `completely_irrelevant`).
  - Context need classification (`needs_retrieve_only`, `needs_history_and_retrieve`, `needs_history_only`, `needs_neither`).
  - Sub-millisecond refusal of out-of-scope/adversarial queries (0 vector tokens consumed).
  - Sub-query decomposition for complex multi-part scientific questions.
- **Multi-Turn Memory Continuity**:
  - Sliding-window turn history with SubQ Decision Model 2 evaluation.
  - Automatic conversational context injection and direct history-only synthesis path.
- **Hybrid Retrieval & Neural Rerank**:
  - Dense semantic retrieval via **BGE-M3** embeddings in Qdrant (embedded or server mode).
  - Sparse lexical retrieval via custom Persian **BM25**.
  - Reciprocal Rank Fusion (RRF) with **BAAI bge-reranker-v2-m3** Cross-Encoder and 3000ms latency circuit-breaker fallback.
- **Parent Section Expansion & Page Citations**:
  - Hierarchical chapter section catalog (868 parent sections) for context reconstruction.
  - Precise inline clickable page citations linking directly to textbook PDFs.
- **Evaluation Engine & Golden Benchmarks**:
  - 6 verified golden benchmark suites (360 unique, deduplicated test items).
  - Dual-layer evaluation: **Layer 1** deterministic Python code gates (< 5ms) + **Layer 2** SubQ Decisions API judge (`subq-snap-preview`) with Gemini fallback.
  - Native export of structured Markdown reports with pass rates and latency profiles.
- **Interactive Web Interface**:
  - Next.js 14, TailwindCSS, Server-Sent Events (SSE) live streaming with TTFT < 1.0s.
  - Full-featured **Evaluation Cockpit** to run, monitor, and export benchmark suites directly from the UI.
  - Real-time operational telemetry (concurrency slot tracking, token costs, latency breakdowns).

---

## 🏗️ System Architecture

```
                             [ User Query ]
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  QueryGate / Intent Guard   │
                     │   (TypeSafe Jev System One)  │
                     └──────────────┬───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
      [ Out-of-Scope ]      [ History-Only ]        [ Scientific Query ]
            │                       │                       │
     Fast Refusal              Synthesize from              │
     (0 token cost)            Prior Turns                  ▼
                                              ┌───────────────────────────┐
                                              │     Hybrid Retrieval      │
                                              │  Dense (Qdrant / BGE-M3)  │
                                              │      + Sparse (BM25)      │
                                              └─────────────┬─────────────┘
                                                            │
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  Neural Cross-Encoder     │
                                              │  Rerank (bge-reranker)    │
                                              └─────────────┬─────────────┘
                                                            │
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  Parent Section Expansion │
                                              │  + Long-Context Reorder   │
                                              └─────────────┬─────────────┘
                                                            │
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  Grounded Synthesis (LLM) │
                                              │   + Page Citations        │
                                              └───────────────────────────┘
```

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed on your machine:

- **Python**: version `3.11` or higher
- **Node.js**: version `18.x` or higher
- **pnpm**: version `8.x` or higher (`npm install -g pnpm`)
- **Docker & Docker Compose** *(optional, for running Redis, Qdrant server, and Neo4j)*

---

## 🚀 Step-by-Step Quickstart

### Step 1: Clone the Repository

```bash
git clone https://github.com/Hamed-Adham/Tutorial_RAG_project.git
cd Tutorial_RAG_project
```

### Step 2: Set Up Python Virtual Environment

Create and activate a Python 3.11 virtual environment:

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

Install core packages in editable mode:

```bash
pip install --upgrade pip
pip install -e rag-core
pip install -e backend
```

### Step 3: Configure Environment Variables

Copy the provided environment template:

```bash
cp .env.example .env
```

Open `.env` and configure your API keys and endpoints:

- **Local LLM / Router**: Set `NINE_ROUTER_BASE_URL` (default: `http://localhost:20128/v1`) or direct OpenAI-compatible endpoint.
- **TypeSafe Jev Gateway**: Set `JEV_BASE_URL` and `JEV_LOCAL_API_KEY` for System One decision routing.
- **SubQ Decisions API**: Set `SUBQ_API_KEY` for high-speed evaluation judge (`subq-snap-preview`).
- **Vector DB**: `QDRANT_MODE=embedded` (default, zero-configuration local file storage in `data_index/qdrant_db`) or `QDRANT_MODE=server` for Docker.
- **Langfuse (Optional)**: Enable tracing by setting `LANGFUSE_ENABLED=true` with your keys.

### Step 4: (Optional) Start Infrastructure via Docker

If you prefer using Docker for Qdrant, Redis, Postgres, and Neo4j:

```bash
make up
```

*(If using `QDRANT_MODE=embedded`, Docker is not strictly required for local testing).*

### Step 5: Build BM25 Index & Vector Store

Generate the lexical BM25 index and Qdrant embeddings from the textbook dataset:

```bash
# 1. Build lexical BM25 index
make build-bm25

# 2. Build Qdrant vector index
make index-science-qdrant
```

### Step 6: Start the Backend API

Start the FastAPI application on port `8000`:

```bash
make serve
```

The backend API will be available at:
- **API URL**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/v1/health`

### Step 7: Start the Frontend Web UI

In a new terminal window, install frontend dependencies and launch the Next.js development server:

```bash
cd frontend
pnpm install
pnpm run dev
```

*(Alternatively, run `make ui-dev` from the repository root).*

Open your browser at **`http://localhost:3000`**.

---

## 🧪 Running Tests & Quality Gates

The codebase enforces a multi-tier testing and verification strategy:

```bash
# Run Tier 1 unit & component tests (< 1s)
make test

# Run Tier 2 deterministic code gates (< 2s, 0 tokens)
make test-gates

# Run Tier 1 + Tier 2 pre-commit check
make check

# Run all backend integration tests
PYTHONPATH=backend:rag-core/src .venv/bin/pytest backend/tests/
```

---

## 📊 Benchmark Suites & Evaluation

The repository includes **6 standardized golden benchmark suites** (360 verified test items) located in `rag-core/src/rag_core/evaluation/datasets/`:

| Benchmark Suite | Test Items | Scope & Validation |
| :--- | :---: | :--- |
| **Science Curriculum** | 120 | Textbook concepts across all 15 chapters (page citations, groundedness) |
| **Administrative FAQ** | 80 | Administrative questions, loan/guarantee policies, admission rules |
| **Conversational Memory** | 60 | Multi-turn dialogues, pronoun resolution, antecedent references |
| **Subquery Decomposition** | 50 | Multi-concept comparative queries requiring multi-hop retrieval |
| **Adversarial Refusal** | 50 | Out-of-scope, prompt injection, and adversarial ungrounded probes |
| **Domain Routing** | 20 | Boundary classification between curriculum and administrative domains |

### Running Evals from Terminal

```bash
# Fast 5-item smoke evaluation
make eval-smoke

# Full evaluation on dev split (90 items) using SubQ judge
make eval-dev

# Generate Markdown evaluation report
make eval-report
```

### Running Evals from Web UI

1. Open `http://localhost:3000`.
2. Click **ارزیابی پایپ‌لاین** (Pipeline Evaluation) in the header.
3. Select any benchmark suite or the full master suite.
4. Choose your evaluator (`subq-snap-preview` or `gemini-cli`).
5. Run the evaluation and inspect real-time pass rates, latency profiles, and failure breakdowns.

---

## 📁 Repository Directory Structure

```text
Unified_RAG/
├── backend/                       # FastAPI async service & session gateway
│   ├── app/
│   │   ├── api/v1/endpoints/      # REST & SSE endpoints (chat, sessions, evaluation)
│   │   ├── core/                  # Gateway concurrency semaphore & telemetry
│   │   ├── schemas/               # Pydantic request/response schemas
│   │   └── services/              # Orchestration & Redis/In-Memory session store
│   └── tests/                     # Gateway and endpoint unit tests
├── rag-core/                      # Core algorithmic RAG library
│   ├── src/rag_core/
│   │   ├── query/                 # QueryGate, SubQ client, summarizer
│   │   ├── retrieval/             # Dense Qdrant + Sparse BM25 retriever
│   │   ├── fusion/                # Neural Cross-Encoder reranker & RRF
│   │   ├── postprocessing/        # Parent section expander & citation builder
│   │   ├── generation/            # Grounded Persian synthesis & prompt templates
│   │   ├── graph/                 # Native LangGraph StateGraph nodes & pipeline
│   │   ├── evaluation/            # Datasets, code gates, SubQJudge, report generator
│   │   └── observability/         # Langfuse distributed tracing & telemetry
│   └── tests/                     # Core library test suites
├── frontend/                      # Next.js 14 web application
│   ├── src/app/                   # App router pages & layouts
│   ├── src/components/            # UI components (Chat, Modals, Sidebar, Telemetry)
│   └── src/lib/                   # Utilities & constants
├── data/                          # 7th-grade science curriculum & markdown extractions
├── docs/                          # Architecture & mentor KPI documentation
├── specs/                         # Feature specifications & contract designs
├── docker-compose.yml             # Local infrastructure services
├── Makefile                       # Developer command shortcuts
└── README.md                      # Project documentation
```

---

## 🛡️ License

This project is developed for educational and research purposes.

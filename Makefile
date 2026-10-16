.PHONY: help dev test test-gates check eval-smoke eval eval-force eval-llm eval-test eval-graph eval-modular eval-legacy eval-science-advanced eval-dev eval-report build-bm25 index-science-qdrant kg-status build up down clean serve load-test ui-dev ui-build

PYTHON := .venv/bin/python
PYTHONPATH := rag-core/src

help:
	@echo "Unified RAG - Management Commands"
	@echo "  make up           Start Docker infrastructure services (Qdrant, Neo4j, Postgres, Redis)"
	@echo "  make down         Stop Docker infrastructure services"
	@echo "  make dev          Run local development environment"
	@echo "  make test         Run Tier 1 unit and component tests (< 1s, 0 tokens)"
	@echo "  make test-gates   Run Tier 2 deterministic gates across 100 benchmark items (< 2s, 0 tokens)"
	@echo "  make check        Run Tier 1 + Tier 2 pre-commit gate (test + test-gates in < 2s)"
	@echo "  make kg-status    Display current Codebase Memory Knowledge Graph status"
	@echo "  make eval-smoke   Run Tier 2.5 fast smoke evaluation (5 items, Jev System One)"
	@echo "  make eval         Run Tier 2.5 full baseline evaluation on dev split (75 items, Jev default)"
	@echo "  make eval-force   Run Tier 2.5 full evaluation on dev split (forced live, 0 cache)"
	@echo "  make eval-llm     Run Tier 3 deep evaluation using local generative LLM (Persian CoT)"
	@echo "  make eval-test    Run Tier 2.5 evaluation on held-out test split (25 items)"
	@echo "  make eval-graph   Run evaluation through LangGraph compiled pipeline"
	@echo "  make eval-modular Run modular eval harness with gates only"
	@echo "  make eval-legacy  Run legacy naive/ runner (regression comparison)"
	@echo "  make build        Build rag-core and backend packages"
	@echo "  make clean        Clean temporary cache and build artifacts"


up:
	docker compose up -d

down:
	docker compose down

dev:
	@echo "Starting local services and checking gateway..."
	docker compose up -d
	@echo "Ready for development. Local Nine Router should run on port 20128."

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s rag-core/tests

test-gates:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/naive/run_baseline_eval.py --gates-only

check: test test-gates

# ── Evaluation (Tier 2.5 Jev + Tier 3 LLM) ──────────────────────────────────
# All eval targets invoke the canonical runner at rag_core/evaluation/.
# Use eval-legacy to compare against the original naive/ runner.

eval-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/evaluation/run_baseline_eval.py --split dev --max-items 5 --force --judge jev

eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/evaluation/run_baseline_eval.py --split dev --judge jev

eval-force:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/evaluation/run_baseline_eval.py --split dev --force --judge jev

eval-llm:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/evaluation/run_baseline_eval.py --split dev --force --judge llm

eval-test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/evaluation/run_baseline_eval.py --split test --judge jev

eval-graph:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval --pipeline langgraph --judge jev

eval-modular:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval --gates-only

eval-retrieval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval --retrieval-only

eval-gates:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval --gates-only

eval-dev:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval --split dev --judge subq --export-report

eval-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval --split dev --export-report specs/007-comprehensive-benchmarks-subq-judge/evaluation_report.md

eval-full:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval

sync-datasets:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/evaluation/build_standardized_benchmarks.py

sync-langfuse:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/evaluation/migrate_datasets_to_langfuse.py

eval-legacy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/naive/run_baseline_eval.py --split dev --judge jev

build-bm25:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/indexing/build_science_bm25.py

index-science-qdrant:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) rag-core/src/rag_core/indexing/index_science_qdrant.py

eval-science-advanced:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_core.evaluation.run_eval --pipeline advanced --suite science_textbook

serve:
	PYTHONPATH=$(PYTHONPATH):backend $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

load-test:
	PYTHONPATH=$(PYTHONPATH):backend $(PYTHON) backend/tests/load/run_load_test.py 50 --mock

ui-dev:
	cd frontend && rm -rf .next && pnpm run dev

ui-build:
	cd frontend && pnpm run build



# ── Codebase Knowledge Graph ─────────────────────────────────────────────────

kg-status:
	@$(PYTHON) -c "import json, os; p='.codebase-memory/artifact.json'; data=json.load(open(p)) if os.path.exists(p) else {}; print('KG Status: %s nodes, %s edges, compressed %s bytes, last indexed: %s' % (data.get('nodes','N/A'), data.get('edges','N/A'), data.get('compressed_size','N/A'), data.get('indexed_at','N/A')))"


build:
	@echo "Building rag-core..."
	cd rag-core && $(PYTHON) -m pip install -e .
	@echo "Building backend..."
	cd backend && $(PYTHON) -m pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info

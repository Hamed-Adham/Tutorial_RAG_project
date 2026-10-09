#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for Unified RAG.
Reads from environment variables and root .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Determine workspace root (three levels up from this file)
CURRENT_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = CURRENT_FILE.parents[3]
ENV_FILE = WORKSPACE_ROOT / ".env"

def load_env_file(env_path: Path):
    """Simple parser for .env file to avoid external dependency requirement."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = val

load_env_file(ENV_FILE)

# Synchronize Langfuse environment variables for SDK, LangChain CallbackHandler, and external tools
if "LANGFUSE_BASE_URL" in os.environ and "LANGFUSE_HOST" not in os.environ:
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
elif "LANGFUSE_HOST" in os.environ and "LANGFUSE_BASE_URL" not in os.environ:
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]

@dataclass
class Settings:
    workspace_root: Path = WORKSPACE_ROOT
    data_dir: Path = WORKSPACE_ROOT / "data"
    
    # Nine Router Settings (Local LLM & Embedding Proxy)
    nine_router_base_url: str = os.getenv("NINE_ROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1"))
    nine_router_api_key: str = os.getenv("NINE_ROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "sk-83c65355ea1ab6f8-h7nd68-fcf94cfe"))
    llm_model: str = os.getenv("LLM_MODEL", "gemini-cli")

    # OmniRoute Settings (Local Gateway on Port 20130)
    omniroute_base_url: str = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20130/v1")
    omniroute_api_key: str = os.getenv("OMNIROUTE_API_KEY", "sk-3331134bf0368e24-91670a-36c86189")
    omniroute_model: str = os.getenv("OMNIROUTE_MODEL", "deepseek-web2api")

    # ── Jev / TypeSafe System One ────────────────────────────────────────────
    # PRIMARY source is the LOCAL Nine Router gateway, which serves System One
    # natively on port 20128. The legacy TypeSafe.ai endpoint in .env is kept
    # only as a fallback and is not reachable with the local credentials.
    #
    # Override any of these with JEV_BASE_URL / JEV_MODEL / JEV_LOCAL_API_KEY.
    # Set JEV_LOCAL_API_KEY="" to disable the local gateway and force the
    # legacy TypeSafe.ai keys below.
    jev_base_url: str = os.getenv("JEV_BASE_URL", "http://localhost:20128/v1/systemone")
    jev_model: str = os.getenv("JEV_MODEL", "oc/jev-1.13-free")
    jev_local_api_key: Optional[str] = os.getenv(
        "JEV_LOCAL_API_KEY", "sk-83c65355ea1ab6f8-qd9giv-ec31eac7"
    )

    # Legacy TypeSafe.ai credentials — FALLBACK only.
    jev_api_key: Optional[str] = os.getenv("JEV_API_KEY", None)
    system_one_api_key: Optional[str] = os.getenv("SYSTEM_ONE_API_KEY", None)

    @property
    def resolved_jev_api_key(self) -> Optional[str]:
        """Resolve the Jev credential: local gateway key first, then legacy keys."""
        return self.jev_local_api_key or self.jev_api_key or self.system_one_api_key

    # Embeddings (Defaulting to Nine Router's bge-m3)
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "un/gpustack/bge-m3-GGUF")
    embedding_model_path: str = os.getenv(
        "EMBEDDING_MODEL_PATH",
        os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
    )

    @property
    def openai_base_url(self) -> str:
        return self.nine_router_base_url

    @property
    def openai_api_key(self) -> str:
        return self.nine_router_api_key

    # Langfuse
    langfuse_host: str = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"))
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-local")
    langfuse_enabled: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "yes")


    # Vector store (Qdrant)
    qdrant_mode: str = os.getenv("QDRANT_MODE", "embedded").lower()  # "embedded" or "server"
    qdrant_url: Optional[str] = os.getenv("QDRANT_URL", None)
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    qdrant_path: str = os.getenv("QDRANT_PATH", str(WORKSPACE_ROOT / "data_index" / "qdrant_db"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION", "unified_rag_chunks")

    # Knowledge Graph (Neo4j)
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Benchmark Scaling & Synthetic Evaluation Settings
    benchmark_model: str = os.getenv("BENCHMARK_MODEL", "gemini-cli")
    benchmark_temperature: float = float(os.getenv("BENCHMARK_TEMPERATURE", "0.7"))
    benchmark_target_total: int = int(os.getenv("BENCHMARK_TARGET_TOTAL", "400"))
    benchmark_faq_target: int = int(os.getenv("BENCHMARK_FAQ_TARGET", "200"))
    benchmark_science_target: int = int(os.getenv("BENCHMARK_SCIENCE_TARGET", "180"))

settings = Settings()


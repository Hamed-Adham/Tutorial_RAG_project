#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGE-M3 embedding wrapper for Naive RAG.
Supports high-performance embedding via Nine Router API (bge-m3 / un/gpustack/bge-m3-GGUF)
with seamless fallback to local snapshot weights.
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Union, Optional
import numpy as np

from rag_core.config import settings

class BgeM3Embedder:
    """Wrapper around BGE-M3 with Nine Router API support and local fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_model_path: Optional[str] = None
    ):
        self.model_name = model_name or settings.embedding_model_name
        self.base_url = (base_url or settings.nine_router_base_url).rstrip("/")
        self.api_key = api_key or settings.nine_router_api_key
        self.local_model_path = local_model_path or settings.embedding_model_path
        self._local_model = None
        self._device = None
        self._use_api = True

    def _embed_batch_via_api(self, texts: List[str]) -> Optional[np.ndarray]:
        """Attempt to embed a batch of texts via Nine Router /v1/embeddings."""
        if not self._use_api:
            return None

        # Candidate models to try in case of alias differences
        candidates = [self.model_name]
        if "un/gpustack/bge-m3-GGUF" not in candidates:
            candidates.append("un/gpustack/bge-m3-GGUF")
        if "bge-m3" not in candidates:
            candidates.append("bge-m3")

        endpoint = f"{self.base_url}/embeddings"

        for model_id in candidates:
            payload = {
                "model": model_id,
                "input": texts if len(texts) > 1 else texts[0]
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    items = data.get("data", [])
                    if items:
                        vecs = [item["embedding"] for item in items]
                        self.model_name = model_id
                        return np.array(vecs, dtype=np.float32)
            except urllib.error.HTTPError as e:
                # If bad request / invalid format, try next candidate
                continue
            except Exception as e:
                # Connection refused or offline
                print(f"Nine Router embedding connection failed: {e}. Switching to local fallback.")
                self._use_api = False
                return None

        return None

    def _load_local_model(self):
        if self._local_model is not None:
            return

        try:
            import torch
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"

        # Attempt sentence_transformers first
        from pathlib import Path
        model_target = self.local_model_path if (self.local_model_path and Path(self.local_model_path).exists()) else "BAAI/bge-m3"
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local BGE-M3 from {model_target} on {self._device}...")
            self._local_model = SentenceTransformer(model_target, device=self._device)
            return
        except ImportError:
            pass

        # Fallback to transformers
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            print(f"Loading local BGE-M3 via transformers ({model_target}) on {self._device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_target)
            self._hf_model = AutoModel.from_pretrained(model_target).to(self._device)
            self._hf_model.eval()
            self._local_model = "transformers"
            return
        except Exception as e:
            raise RuntimeError(f"Could not load local BGE-M3 model from {model_target}: {e}")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of text documents into dense vectors (API first, local fallback)."""
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        # 1. Try Nine Router API first
        if self._use_api:
            all_vectors = []
            api_success = True
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vecs = self._embed_batch_via_api(batch)
                if batch_vecs is not None and len(batch_vecs) == len(batch):
                    all_vectors.append(batch_vecs)
                else:
                    api_success = False
                    break

            if api_success and all_vectors:
                return np.vstack(all_vectors).astype(np.float32)

        # 2. Local fallback
        print("Using local BGE-M3 model for embeddings...")
        self._load_local_model()

        if hasattr(self._local_model, "encode"):
            embeddings = self._local_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50,
                normalize_embeddings=True
            )
            return np.array(embeddings, dtype=np.float32)
        else:
            import torch
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._hf_model(**encoded)
                    attention_mask = encoded["attention_mask"].unsqueeze(-1)
                    token_embeddings = outputs.last_hidden_state
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    all_embeddings.append(normalized.cpu().numpy())

            return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into a 1D dense vector."""
        vectors = self.embed_documents([query], batch_size=1)
        return vectors[0]

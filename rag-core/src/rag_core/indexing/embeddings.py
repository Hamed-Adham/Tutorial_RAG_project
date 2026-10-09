#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular Embedding Provider for Unified RAG (Stage 4).
Wraps BGE-M3 dense multilingual embeddings with query prefix prepending and quantization options.
"""

from typing import List, Union, Optional
import numpy as np
from rag_core.naive.embedder import BgeM3Embedder as BaseBgeM3Embedder
from rag_core.config import settings

# BGE-M3 recommended query instruction prefix for asymmetrical retrieval
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BgeM3Embedder(BaseBgeM3Embedder):
    """
    Modular BGE-M3 embedding loader supporting query prefixes and int8 quantization.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_model_path: Optional[str] = None,
        int8_quantization: bool = True
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            local_model_path=local_model_path
        )
        self.int8_quantization = int8_quantization

    def embed_query(self, query: str) -> List[float]:
        """Embed a single user query, prepending the asymmetrical retrieval instruction prefix."""
        prefixed_query = f"{QUERY_PREFIX}{query}" if not query.startswith(QUERY_PREFIX) else query
        emb = self.embed_text(prefixed_query)
        if isinstance(emb, np.ndarray):
            return emb.tolist()
        return list(emb)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a batch of document passages without query instruction prefix."""
        embs = self.embed_texts(documents)
        if isinstance(embs, np.ndarray):
            return embs.tolist()
        return [list(e) for e in embs]

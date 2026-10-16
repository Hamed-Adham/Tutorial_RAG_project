"""
In-Memory BM25 Indexer with Hazm Persian Text Normalization and Disk Serialization.
Pre-tokenizes and indexes 7th-grade science curriculum chunks for sub-5ms lexical retrieval.
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi

try:
    from hazm import Normalizer, Stemmer, word_tokenize
    _HAZM_AVAILABLE = True
except ImportError:
    _HAZM_AVAILABLE = False
    Normalizer, Stemmer, word_tokenize = None, None, None

from rag_core.indexing.science_schema import ScienceChunk, RetrievalCandidate


class PersianTextProcessor:
    """Handles Persian text normalization, stemming, and tokenization."""

    def __init__(self):
        if _HAZM_AVAILABLE:
            self.normalizer = Normalizer()
            self.stemmer = Stemmer()
            # Eagerly warmup internal regex and affix tables to eliminate first-query cold start
            try:
                _ = self.tokenize("گرم")
            except Exception:
                pass
        else:
            self.normalizer = None
            self.stemmer = None

    def tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize Persian text into stems."""
        if not text:
            return []

        # 1. Hazm normalization or fallback regex normalization
        if self.normalizer is not None:
            normalized = self.normalizer.normalize(text)
            tokens = word_tokenize(normalized)
            tokens = [self.stemmer.stem(t) for t in tokens if len(t.strip()) > 1]
        else:
            # Fallback regex tokenization
            normalized = re.sub(r'[\u200c\s]+', ' ', text)
            tokens = [t for t in re.split(r'\W+', normalized) if len(t) > 1]

        # Filter punctuation & single character artifacts
        return [t.lower() for t in tokens if not re.match(r'^\W+$', t)]


class ScienceBM25Index:
    """In-memory BM25 lexical index with pickle serialization."""

    def __init__(self):
        self.processor = PersianTextProcessor()
        self.chunks: List[ScienceChunk] = []
        self.chunk_lookup: Dict[str, ScienceChunk] = {}
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_chunks(self, chunks: List[ScienceChunk]) -> None:
        """Add chunks to index and build tokenized corpus."""
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_lookup[chunk.chunk_id] = chunk
            # Tokenize breadcrumbs (boosted) + chunk text
            indexed_text = f"{chunk.breadcrumbs} {chunk.breadcrumbs} {chunk.text}"
            tokens = self.processor.tokenize(indexed_text)
            self.tokenized_corpus.append(tokens)

    def build(self) -> None:
        """Fit BM25Okapi over tokenized corpus."""
        if not self.tokenized_corpus:
            raise ValueError("Cannot build BM25 index on empty corpus.")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 50,
        chapter_filter: Optional[List[int]] = None,
        category_filter: Optional[List[str]] = None
    ) -> List[Tuple[Any, float]]:
        """Search query against BM25 index returning top_k (chunk, score) tuples with optional chapter or category pre-filtering."""
        if self.bm25 is None:
            raise ValueError("BM25 index is not built. Call build() or load() first.")

        query_tokens = self.processor.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        candidate_indices = range(len(scores))
        if chapter_filter:
            candidate_indices = [
                i for i in candidate_indices 
                if getattr(self.chunks[i], "chapter", None) in chapter_filter
            ]
        if category_filter:
            candidate_indices = [
                i for i in candidate_indices 
                if getattr(self.chunks[i], "category", None) in category_filter
            ]

        top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.chunks[idx], score))
        return results

    def save(self, filepath: str | Path) -> None:
        """Persist index state to disk artifact."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "tokenized_corpus": self.tokenized_corpus,
                "bm25": self.bm25
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str | Path) -> "ScienceBM25Index":
        """Load serialized BM25 index artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        index = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
            index.chunks = data["chunks"]
            index.tokenized_corpus = data["tokenized_corpus"]
            index.bm25 = data["bm25"]
            index.chunk_lookup = {c.chunk_id: c for c in index.chunks}
        return index

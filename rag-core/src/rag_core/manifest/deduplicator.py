"""
Stage 1: Document Deduplication

Two-stage deduplication strategy:
  1. **Exact deduplication** — SHA-256 byte-level hash collapse.
     Documents with identical hashes are dropped immediately.
  2. **Near-duplicate clustering** — MinHash + LSH (Locality-Sensitive Hashing)
     to find documents that are substantially similar (e.g., different headers
     on the same whitepaper, or minor edits to a policy document).

Constitution v1.2.0 compliance:
- Framework-independent: pure Python + optional `datasketch`.
- Graceful fallback: MinHash clustering degrades to exact-only when
  `datasketch` is not installed.
- Law of Parsimony: exact SHA-256 is always run first (O(1) lookup).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationResult:
    """Result of a deduplication check for a single document."""

    is_exact_duplicate: bool
    """True when SHA-256 matches an already-seen document exactly."""

    is_near_duplicate: bool
    """True when MinHash similarity exceeds the configured threshold."""

    duplicate_of: Optional[str] = None
    """``doc_id`` of the canonical (first-seen) document, if duplicate."""

    similarity_score: float = 0.0
    """Estimated Jaccard similarity (0.0–1.0); 1.0 = exact byte match."""

    cluster_id: Optional[str] = None
    """Cluster label assigned to the near-duplicate group (if any)."""

    message: str = ""
    """Human-readable explanation of the deduplication decision."""


@dataclass
class DocumentFingerprint:
    """Lightweight fingerprint computed from a document's raw bytes."""

    doc_id: str
    """Stable doc_id from the manifest catalog."""

    sha256: str
    """Exact SHA-256 hash of raw bytes."""

    shingles: Set[str] = field(default_factory=set)
    """Character-level 5-shingles extracted from normalised text."""


# ---------------------------------------------------------------------------
# Helpers — text normalisation & shingling
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    """
    Normalise document text before shingling.

    Steps:
    1. Lowercase.
    2. Strip punctuation.
    3. Collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shingle(text: str, k: int = 5) -> Set[str]:
    """
    Return the set of character-level k-shingles for *text*.

    Shingling converts a document into a set representation suitable
    for Jaccard similarity estimation via MinHash.

    Args:
        text: Input string (should be normalised first).
        k:    Shingle size in characters (default 5).

    Returns:
        Set of k-length substrings.
    """
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Compute exact Jaccard similarity between two shingle sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# MinHash wrapper (with datasketch fallback)
# ---------------------------------------------------------------------------


class _MinHashEstimator:
    """
    Wrapper around ``datasketch.MinHash`` with a pure-Python fallback.

    If ``datasketch`` is not installed the fallback computes exact Jaccard
    similarity over the raw shingle sets, which is correct but O(n*m) for
    large corpora.  Install ``datasketch`` for production use::

        pip install datasketch
    """

    def __init__(self, num_perm: int = 128) -> None:
        self._num_perm = num_perm
        try:
            import datasketch  # noqa: F401

            self._use_datasketch = True
            logger.debug("MinHash: using datasketch backend (num_perm=%d)", num_perm)
        except ImportError:
            self._use_datasketch = False
            logger.warning(
                "datasketch not installed. Near-duplicate detection will use "
                "exact Jaccard similarity (slower for large corpora). "
                "Install with: pip install datasketch"
            )

    def compute(self, shingles: Set[str]) -> "MinHashHandle":
        """Compute a MinHash signature for the given shingle set."""
        if self._use_datasketch:
            from datasketch import MinHash

            m = MinHash(num_perm=self._num_perm)
            for s in shingles:
                m.update(s.encode("utf-8"))
            return _DatasketchHandle(m, shingles)
        else:
            return _FallbackHandle(shingles)

    def jaccard(self, a: "MinHashHandle", b: "MinHashHandle") -> float:
        """Estimate Jaccard similarity between two MinHash handles."""
        return a.jaccard(b)


class MinHashHandle:
    """Abstract handle returned by ``_MinHashEstimator.compute``."""

    def jaccard(self, other: "MinHashHandle") -> float:
        raise NotImplementedError


class _DatasketchHandle(MinHashHandle):
    def __init__(self, minhash: Any, shingles: Set[str]) -> None:
        self._mh = minhash
        self._shingles = shingles

    def jaccard(self, other: "_DatasketchHandle") -> float:  # type: ignore[override]
        return self._mh.jaccard(other._mh)


class _FallbackHandle(MinHashHandle):
    def __init__(self, shingles: Set[str]) -> None:
        self._shingles = shingles

    def jaccard(self, other: "_FallbackHandle") -> float:  # type: ignore[override]
        return _jaccard(self._shingles, other._shingles)


# ---------------------------------------------------------------------------
# Any-type alias (avoid importing typing.Any at top for Python 3.8 compat)
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402 — keep local to avoid shadowing builtins


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class DocumentDeduplicator:
    """
    Two-stage document deduplication engine.

    Stage 1 — **Exact SHA-256 collapse**:
        O(1) hash lookup.  Documents with identical SHA-256 hashes are
        immediately classified as exact duplicates regardless of MinHash.

    Stage 2 — **MinHash near-duplicate clustering**:
        Uses LSH-approximated Jaccard similarity to detect near-duplicates
        (documents sharing ≥ ``near_dup_threshold`` of their 5-shingles).
        Documents above the threshold are grouped into the same cluster and
        only the **first-seen** document in each cluster is retained.

    Usage::

        dedup = DocumentDeduplicator(near_dup_threshold=0.85)

        # Register canonical docs
        dedup.add_document("doc-1", sha256="abc...", text="Full document text...")

        # Check incoming document
        result = dedup.check("doc-2", sha256="xyz...", text="Slightly modified text...")
        if result.is_near_duplicate:
            print(f"Near-dup of {result.duplicate_of} (sim={result.similarity_score:.2f})")
    """

    def __init__(
        self,
        near_dup_threshold: float = 0.85,
        shingle_size: int = 5,
        num_perm: int = 128,
    ) -> None:
        """
        Args:
            near_dup_threshold: Jaccard similarity threshold above which two
                documents are considered near-duplicates (0.0–1.0).
            shingle_size: Character n-gram size for shingling (default 5).
            num_perm: Number of permutations for MinHash (higher = more accurate).
        """
        self._threshold = near_dup_threshold
        self._k = shingle_size
        self._estimator = _MinHashEstimator(num_perm=num_perm)

        # Storage
        self._sha256_to_doc_id: Dict[str, str] = {}
        self._fingerprints: Dict[str, DocumentFingerprint] = {}
        self._minhashes: Dict[str, MinHashHandle] = {}
        self._clusters: Dict[str, str] = {}  # doc_id -> cluster_id (canonical doc_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, sha256: str, text: str) -> None:
        """
        Register a document as a canonical (non-duplicate) entry.

        This must be called **after** ``check()`` returns ``is_exact_duplicate=False``
        and ``is_near_duplicate=False`` to build the deduplication index.

        Args:
            doc_id: Stable doc_id from the manifest catalog.
            sha256: SHA-256 hash of the raw file bytes.
            text:   Plain-text content for shingling.
        """
        normalised = _normalise_text(text)
        shingles = _shingle(normalised, k=self._k)
        fp = DocumentFingerprint(doc_id=doc_id, sha256=sha256, shingles=shingles)

        self._sha256_to_doc_id[sha256] = doc_id
        self._fingerprints[doc_id] = fp
        self._minhashes[doc_id] = self._estimator.compute(shingles)
        self._clusters[doc_id] = doc_id  # canonical of itself

        logger.debug("Deduplicator: indexed doc_id=%s (shingles=%d)", doc_id, len(shingles))

    def check(
        self, doc_id: str, sha256: str, text: str
    ) -> DeduplicationResult:
        """
        Check whether a candidate document is an exact or near-duplicate.

        Does **not** modify internal state — call ``add_document`` separately
        if the check returns a non-duplicate result.

        Args:
            doc_id: Candidate document's stable doc_id.
            sha256: SHA-256 of the candidate's raw bytes.
            text:   Plain-text content of the candidate.

        Returns:
            ``DeduplicationResult`` describing the deduplication decision.
        """
        # --- Stage 1: Exact SHA-256 ---
        if sha256 in self._sha256_to_doc_id:
            canonical_id = self._sha256_to_doc_id[sha256]
            return DeduplicationResult(
                is_exact_duplicate=True,
                is_near_duplicate=False,
                duplicate_of=canonical_id,
                similarity_score=1.0,
                cluster_id=self._clusters.get(canonical_id),
                message=(
                    f"Exact SHA-256 match with doc_id={canonical_id}"
                ),
            )

        # --- Stage 2: MinHash near-duplicate ---
        normalised = _normalise_text(text)
        shingles = _shingle(normalised, k=self._k)
        candidate_mh = self._estimator.compute(shingles)

        best_sim = 0.0
        best_id: Optional[str] = None

        for existing_id, existing_mh in self._minhashes.items():
            sim = self._estimator.jaccard(candidate_mh, existing_mh)
            if sim > best_sim:
                best_sim = sim
                best_id = existing_id

        if best_sim >= self._threshold and best_id is not None:
            canonical_id = self._clusters.get(best_id, best_id)
            logger.info(
                "Near-duplicate detected: doc_id=%s sim=%.3f canonical=%s",
                doc_id,
                best_sim,
                canonical_id,
            )
            return DeduplicationResult(
                is_exact_duplicate=False,
                is_near_duplicate=True,
                duplicate_of=canonical_id,
                similarity_score=best_sim,
                cluster_id=canonical_id,
                message=(
                    f"Near-duplicate (Jaccard={best_sim:.3f}) of doc_id={canonical_id}"
                ),
            )

        return DeduplicationResult(
            is_exact_duplicate=False,
            is_near_duplicate=False,
            similarity_score=best_sim,
            message="No duplicate detected",
        )

    def cluster_summary(self) -> Dict[str, List[str]]:
        """
        Return a dict mapping each canonical ``doc_id`` to the list of
        near-duplicate ``doc_id`` values in its cluster.

        Only includes clusters with at least one near-duplicate member.
        """
        from collections import defaultdict

        clusters: Dict[str, List[str]] = defaultdict(list)
        for doc_id, cluster_id in self._clusters.items():
            if doc_id != cluster_id:
                clusters[cluster_id].append(doc_id)
        return dict(clusters)

    def contains(self, sha256: str) -> bool:
        """Return ``True`` if an exact SHA-256 match exists in the index."""
        return sha256 in self._sha256_to_doc_id

    def indexed_count(self) -> int:
        """Return the number of canonical documents indexed."""
        return len(self._fingerprints)

    def clear(self) -> None:
        """Clear all deduplication state (test teardown only)."""
        self._sha256_to_doc_id.clear()
        self._fingerprints.clear()
        self._minhashes.clear()
        self._clusters.clear()


# ---------------------------------------------------------------------------
# Convenience function — deduplicate a batch
# ---------------------------------------------------------------------------


def deduplicate_batch(
    documents: List[Tuple[str, str, bytes, str]],
    near_dup_threshold: float = 0.85,
) -> Iterator[Tuple[str, DeduplicationResult]]:
    """
    Deduplicate a batch of documents and yield ``(doc_id, result)`` pairs.

    Documents are processed in order.  The first occurrence of any content is
    always yielded as non-duplicate; subsequent duplicates are flagged.

    Args:
        documents: List of ``(doc_id, filename, content_bytes, text)`` tuples.
        near_dup_threshold: MinHash Jaccard threshold for near-duplicates.

    Yields:
        ``(doc_id, DeduplicationResult)`` for each document in *documents*.

    Example::

        for doc_id, result in deduplicate_batch(docs):
            if not result.is_exact_duplicate and not result.is_near_duplicate:
                catalog.update_status(doc_id, DocumentStatus.INDEXED)
    """
    dedup = DocumentDeduplicator(near_dup_threshold=near_dup_threshold)

    for doc_id, _filename, content_bytes, text in documents:
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        result = dedup.check(doc_id=doc_id, sha256=sha256, text=text)

        if not result.is_exact_duplicate and not result.is_near_duplicate:
            dedup.add_document(doc_id=doc_id, sha256=sha256, text=text)

        yield doc_id, result

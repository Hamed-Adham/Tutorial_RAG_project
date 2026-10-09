# `rag-core` Package Interface Contracts

This document defines the formal Python Protocol contracts and interfaces exposed by the `rag-core` package according to Principle I (Three-Layer Decoupled Architecture).

---

## 1. Document Catalog & Manifest Interface (`rag_core.manifest.catalog`)

```python
from typing import Protocol, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ManifestRecord:
    doc_id: str
    filename: str
    sha256_hash: str
    status: str
    page_count: int
    total_chunks: int

class DocumentCatalog(Protocol):
    def register_document(
        self,
        file_path: Path,
        tenant_id: str = "default",
        acl_roles: Optional[List[str]] = None
    ) -> ManifestRecord:
        """Calculate SHA-256, verify deduplication, and catalog record."""
        ...

    def cascade_delete_document(self, doc_id: str) -> bool:
        """
        Synchronously purge document from Postgres, Qdrant, BM25, and Neo4j.
        Rolls back if any secondary store fails.
        """
        ...
```

---

## 2. Parsing & Canonical IR Protocol (`rag_core.parsing.router`)

```python
from typing import Protocol, Union
from pathlib import Path
from rag_core.parsing.schema import ParsedDocument

class DocumentParser(Protocol):
    def parse(
        self,
        source: Union[str, Path, bytes],
        doc_id: str,
        filename: str
    ) -> ParsedDocument:
        """Probe MIME type and parse into Canonical IR Block objects."""
        ...
```

---

## 3. Structure-Aware Chunker Protocol (`rag_core.chunking.hierarchical`)

```python
from typing import Protocol, List
from rag_core.parsing.schema import ParsedDocument
from rag_core.chunking.hierarchical import Chunk

class DocumentChunker(Protocol):
    def chunk(self, doc: ParsedDocument) -> List[Chunk]:
        """
        Split Canonical IR blocks into hierarchical chunks with breadcrumb prepending.
        Target: ~512 tokens (ceiling 1024 for semantic sections).
        """
        ...
```

---

## 4. Multi-Pronged Retriever Protocol (`rag_core.retrieval`)

```python
from typing import Protocol, List, Optional
from dataclasses import dataclass

@dataclass
class RetrievalCandidate:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    breadcrumb: str
    page: int

class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        acl_roles: Optional[List[str]] = None
    ) -> List[RetrievalCandidate]:
        """Execute search with ACL pre-filtering and return scored candidates."""
        ...
```

---

## 5. Evaluation Harness Protocol (`rag_core.evaluation.harness`)

```python
from typing import Protocol, Dict, Any, List
from dataclasses import dataclass

@dataclass
class EvalReport:
    suite_name: str
    total_evaluated: int
    gate_pass_rate: float
    refusal_accuracy: float
    domain_routing_accuracy: float
    metrics: Dict[str, float]
    average_judge_latency: float

class EvaluationHarness(Protocol):
    def run_suite(
        self,
        suite: str = "all",
        gates_only: bool = False,
        limit: Optional[int] = None
    ) -> EvalReport:
        """Cascade from Layer 1 Code Gates to Layer 2 LLM Judge."""
        ...
```

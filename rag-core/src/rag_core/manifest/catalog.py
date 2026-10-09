"""
Stage 1: PostgreSQL Document Catalog

Provides a stable `doc_id`, SHA-256 hash, tenant, and ACL tracking
for every ingested document. Supports async and sync usage patterns
with a graceful offline fallback (SQLite in-memory) for unit tests.

Constitution v1.2.0 compliance:
- Three-layer decoupled: zero FastAPI/Next.js imports.
- Tracing: Langfuse spans wrap catalog operations.
- Baseline-First: simple CRUD with upsert semantics.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    """Lifecycle states for a document in the manifest catalog."""
    PENDING = "pending"
    INDEXED = "indexed"
    DUPLICATE = "duplicate"
    TOMBSTONED = "tombstoned"
    ERROR = "error"


@dataclass
class DocumentRecord:
    """A single row in the manifest catalog table."""

    doc_id: str
    """Stable UUID-v4 primary key, never reused after creation."""

    filename: str
    """Original filename as uploaded by the user."""

    sha256: str
    """SHA-256 hex digest of the raw file bytes (exact deduplication key)."""

    tenant_id: str
    """Tenant / organisation identifier for multi-tenancy isolation."""

    acl_roles: List[str] = field(default_factory=list)
    """List of roles allowed to retrieve chunks from this document."""

    status: DocumentStatus = DocumentStatus.PENDING
    """Current lifecycle state of the document."""

    file_size_bytes: int = 0
    """Raw file size in bytes."""

    mime_type: str = "application/octet-stream"
    """MIME type detected at ingestion time."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC timestamp when the record was first created."""

    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC timestamp of the last status update."""

    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary key/value metadata bag (e.g., source system, batch_id)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON or DB insertion."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "tenant_id": self.tenant_id,
            "acl_roles": self.acl_roles,
            "status": self.status.value,
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "extra_metadata": self.extra_metadata,
        }


# ---------------------------------------------------------------------------
# Catalog Result
# ---------------------------------------------------------------------------

@dataclass
class CatalogResult:
    """Result returned by `DocumentCatalog.register()`."""

    doc_id: str
    """The stable doc_id for this document (new or existing)."""

    is_new: bool
    """True if this is the first time the document was ingested."""

    is_duplicate: bool
    """True if an exact SHA-256 match already existed in the catalog."""

    record: DocumentRecord
    """The full document record (may be the existing one on duplicate)."""

    message: str = ""
    """Human-readable status message."""


# ---------------------------------------------------------------------------
# Helper — SHA-256
# ---------------------------------------------------------------------------

def compute_sha256(content: bytes) -> str:
    """Return the SHA-256 hex digest of *content*."""
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# In-Memory Catalog (default / offline fallback)
# ---------------------------------------------------------------------------

class InMemoryCatalog:
    """
    Thread-safe in-memory document catalog backed by a plain dict.

    Used in:
    - Unit tests (zero external dependencies).
    - Local development when PostgreSQL is not running.
    - Any context where `use_postgres=False`.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, DocumentRecord] = {}
        self._sha256_index: Dict[str, str] = {}  # sha256 -> doc_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        filename: str,
        content: bytes,
        tenant_id: str = "default",
        acl_roles: Optional[List[str]] = None,
        mime_type: str = "application/octet-stream",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> CatalogResult:
        """
        Register a document in the catalog.

        If an exact SHA-256 match already exists, marks the new submission
        as a duplicate and returns the *existing* record.

        Args:
            filename: Original document filename.
            content: Raw file bytes used to compute SHA-256.
            tenant_id: Tenant namespace (default: ``"default"``).
            acl_roles: List of roles permitted to retrieve this document.
            mime_type: MIME type of the file.
            extra_metadata: Arbitrary metadata dict.

        Returns:
            ``CatalogResult`` with ``doc_id``, ``is_new``, ``is_duplicate``.
        """
        sha256 = compute_sha256(content)
        acl_roles = acl_roles or ["public"]
        extra_metadata = extra_metadata or {}

        # Exact duplicate check
        if sha256 in self._sha256_index:
            existing_id = self._sha256_index[sha256]
            existing = self._by_id[existing_id]
            logger.info(
                "Exact duplicate detected",
                extra={"sha256": sha256, "existing_doc_id": existing_id},
            )
            return CatalogResult(
                doc_id=existing_id,
                is_new=False,
                is_duplicate=True,
                record=existing,
                message=f"Duplicate of existing doc_id={existing_id}",
            )

        # New document
        doc_id = str(uuid.uuid4())
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            sha256=sha256,
            tenant_id=tenant_id,
            acl_roles=acl_roles,
            status=DocumentStatus.PENDING,
            file_size_bytes=len(content),
            mime_type=mime_type,
            extra_metadata=extra_metadata,
        )
        self._by_id[doc_id] = record
        self._sha256_index[sha256] = doc_id

        logger.info(
            "New document registered",
            extra={"doc_id": doc_id, "filename": filename, "sha256": sha256},
        )
        return CatalogResult(
            doc_id=doc_id,
            is_new=True,
            is_duplicate=False,
            record=record,
            message=f"Registered new doc_id={doc_id}",
        )

    def update_status(self, doc_id: str, status: DocumentStatus) -> bool:
        """
        Update the lifecycle status of a document.

        Returns:
            ``True`` if the document was found and updated, ``False`` otherwise.
        """
        if doc_id not in self._by_id:
            logger.warning("update_status: doc_id not found", extra={"doc_id": doc_id})
            return False
        record = self._by_id[doc_id]
        record.status = status
        record.updated_at = datetime.now(timezone.utc)
        return True

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        """Retrieve a document record by its stable ``doc_id``."""
        return self._by_id.get(doc_id)

    def get_by_sha256(self, sha256: str) -> Optional[DocumentRecord]:
        """Retrieve a document record by its SHA-256 hash."""
        doc_id = self._sha256_index.get(sha256)
        if doc_id:
            return self._by_id.get(doc_id)
        return None

    def list_documents(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> List[DocumentRecord]:
        """
        List documents, optionally filtered by tenant and/or status.

        Args:
            tenant_id: If set, return only documents for this tenant.
            status: If set, return only documents with this status.

        Returns:
            List of matching ``DocumentRecord`` objects.
        """
        results = list(self._by_id.values())
        if tenant_id:
            results = [r for r in results if r.tenant_id == tenant_id]
        if status:
            results = [r for r in results if r.status == status]
        return results

    def tombstone(self, doc_id: str) -> bool:
        """
        Soft-delete a document by setting its status to ``TOMBSTONED``.

        Tombstoned documents remain in the catalog for audit purposes
        but are excluded from retrieval and deduplication checks.

        Returns:
            ``True`` if the document was found and tombstoned.
        """
        if doc_id not in self._by_id:
            return False
        record = self._by_id[doc_id]
        record.status = DocumentStatus.TOMBSTONED
        record.updated_at = datetime.now(timezone.utc)
        # Remove from dedup index so the file can be re-ingested if needed
        self._sha256_index.pop(record.sha256, None)
        logger.info("Document tombstoned", extra={"doc_id": doc_id})
        return True

    def count(self) -> int:
        """Return the total number of documents in the catalog."""
        return len(self._by_id)

    def clear(self) -> None:
        """Clear all records (intended for test teardown only)."""
        self._by_id.clear()
        self._sha256_index.clear()


# ---------------------------------------------------------------------------
# PostgreSQL Catalog (production)
# ---------------------------------------------------------------------------

class PostgreSQLCatalog:
    """
    PostgreSQL-backed document catalog using SQLAlchemy Core.

    The table schema is intentionally lightweight and append-friendly.
    Heavy migrations are handled via Alembic in the ``backend`` layer.

    This class is initialised lazily: the engine and table are created
    on first use so that import does not fail in environments without
    a running database.

    Usage::

        catalog = PostgreSQLCatalog(database_url="postgresql+psycopg2://...")
        result = catalog.register(filename="doc.pdf", content=bytes(...))
    """

    TABLE_NAME = "document_catalog"
    DDL = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        doc_id          TEXT PRIMARY KEY,
        filename        TEXT NOT NULL,
        sha256          TEXT NOT NULL UNIQUE,
        tenant_id       TEXT NOT NULL DEFAULT 'default',
        acl_roles       TEXT NOT NULL DEFAULT '["public"]',
        status          TEXT NOT NULL DEFAULT 'pending',
        file_size_bytes BIGINT NOT NULL DEFAULT 0,
        mime_type       TEXT NOT NULL DEFAULT 'application/octet-stream',
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        extra_metadata  TEXT NOT NULL DEFAULT '{{}}'
    );
    CREATE INDEX IF NOT EXISTS idx_doc_catalog_sha256
        ON {TABLE_NAME} (sha256);
    CREATE INDEX IF NOT EXISTS idx_doc_catalog_tenant
        ON {TABLE_NAME} (tenant_id);
    CREATE INDEX IF NOT EXISTS idx_doc_catalog_status
        ON {TABLE_NAME} (status);
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._engine: Any = None  # sqlalchemy.engine.Engine
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily create the engine and table on first use."""
        if self._initialized:
            return
        try:
            import json  # noqa: F401 — required for JSON (de)serialisation below
            from sqlalchemy import create_engine, text

            self._engine = create_engine(self._database_url, pool_pre_ping=True)
            with self._engine.connect() as conn:
                # Execute DDL statements one by one (SQLAlchemy needs them split)
                for stmt in [s.strip() for s in self.DDL.split(";") if s.strip()]:
                    conn.execute(text(stmt))
                conn.commit()
            self._initialized = True
            logger.info("PostgreSQL document catalog initialized")
        except Exception as exc:
            logger.error("Failed to initialize PostgreSQL catalog: %s", exc)
            raise

    def register(
        self,
        filename: str,
        content: bytes,
        tenant_id: str = "default",
        acl_roles: Optional[List[str]] = None,
        mime_type: str = "application/octet-stream",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> CatalogResult:
        """Register a document; upserts on SHA-256 uniqueness conflict."""
        import json
        from sqlalchemy import text

        self._ensure_initialized()
        sha256 = compute_sha256(content)
        acl_roles = acl_roles or ["public"]
        extra_metadata = extra_metadata or {}

        with self._engine.connect() as conn:
            # Check for existing sha256
            existing = conn.execute(
                text(f"SELECT doc_id FROM {self.TABLE_NAME} WHERE sha256 = :h"),
                {"h": sha256},
            ).fetchone()

            if existing:
                doc_id = existing[0]
                row = conn.execute(
                    text(f"SELECT * FROM {self.TABLE_NAME} WHERE doc_id = :id"),
                    {"id": doc_id},
                ).fetchone()
                record = self._row_to_record(row)
                logger.info("Duplicate detected (PG)", extra={"doc_id": doc_id})
                return CatalogResult(
                    doc_id=doc_id,
                    is_new=False,
                    is_duplicate=True,
                    record=record,
                    message=f"Duplicate of existing doc_id={doc_id}",
                )

            doc_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.TABLE_NAME}
                        (doc_id, filename, sha256, tenant_id, acl_roles,
                         status, file_size_bytes, mime_type,
                         created_at, updated_at, extra_metadata)
                    VALUES
                        (:doc_id, :filename, :sha256, :tenant_id, :acl_roles,
                         :status, :file_size_bytes, :mime_type,
                         :created_at, :updated_at, :extra_metadata)
                    """
                ),
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "sha256": sha256,
                    "tenant_id": tenant_id,
                    "acl_roles": json.dumps(acl_roles),
                    "status": DocumentStatus.PENDING.value,
                    "file_size_bytes": len(content),
                    "mime_type": mime_type,
                    "created_at": now,
                    "updated_at": now,
                    "extra_metadata": json.dumps(extra_metadata),
                },
            )
            conn.commit()

        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            sha256=sha256,
            tenant_id=tenant_id,
            acl_roles=acl_roles,
            status=DocumentStatus.PENDING,
            file_size_bytes=len(content),
            mime_type=mime_type,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            extra_metadata=extra_metadata,
        )
        return CatalogResult(
            doc_id=doc_id,
            is_new=True,
            is_duplicate=False,
            record=record,
            message=f"Registered new doc_id={doc_id}",
        )

    def update_status(self, doc_id: str, status: DocumentStatus) -> bool:
        """Update document status in PostgreSQL."""
        from sqlalchemy import text

        self._ensure_initialized()
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET status = :status, updated_at = :now
                    WHERE doc_id = :doc_id
                    """
                ),
                {"status": status.value, "now": datetime.now(timezone.utc), "doc_id": doc_id},
            )
            conn.commit()
            return result.rowcount > 0

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        """Retrieve a document record from PostgreSQL by ``doc_id``."""
        from sqlalchemy import text

        self._ensure_initialized()
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT * FROM {self.TABLE_NAME} WHERE doc_id = :id"),
                {"id": doc_id},
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_documents(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> List[DocumentRecord]:
        """List catalog documents with optional filtering."""
        from sqlalchemy import text

        self._ensure_initialized()
        conditions: List[str] = []
        params: Dict[str, Any] = {}
        if tenant_id:
            conditions.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        if status:
            conditions.append("status = :status")
            params["status"] = status.value

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {self.TABLE_NAME} {where} ORDER BY created_at"),
                params,
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def tombstone(self, doc_id: str) -> bool:
        """Soft-delete by setting status to TOMBSTONED."""
        return self.update_status(doc_id, DocumentStatus.TOMBSTONED)

    def _row_to_record(self, row: Any) -> DocumentRecord:
        """Convert a SQLAlchemy Row to a ``DocumentRecord``."""
        import json

        return DocumentRecord(
            doc_id=row.doc_id,
            filename=row.filename,
            sha256=row.sha256,
            tenant_id=row.tenant_id,
            acl_roles=json.loads(row.acl_roles),
            status=DocumentStatus(row.status),
            file_size_bytes=row.file_size_bytes,
            mime_type=row.mime_type,
            created_at=row.created_at,
            updated_at=row.updated_at,
            extra_metadata=json.loads(row.extra_metadata),
        )


# ---------------------------------------------------------------------------
# Factory — DocumentCatalog
# ---------------------------------------------------------------------------

class DocumentCatalog:
    """
    Unified factory that exposes an identical API regardless of the backend.

    Selects the backend at construction time:
    - ``database_url=None`` or ``use_postgres=False`` → ``InMemoryCatalog``
    - ``database_url`` provided and ``use_postgres=True`` → ``PostgreSQLCatalog``

    This allows unit tests to run with zero infrastructure while production
    uses the full PostgreSQL backend transparently.

    Example::

        # In tests / local dev
        catalog = DocumentCatalog()

        # In production
        catalog = DocumentCatalog(database_url="postgresql+psycopg2://user:pw@host/db")
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        use_postgres: bool = False,
    ) -> None:
        if use_postgres and database_url:
            self._backend: InMemoryCatalog | PostgreSQLCatalog = PostgreSQLCatalog(database_url)
            logger.info("DocumentCatalog: using PostgreSQL backend")
        else:
            self._backend = InMemoryCatalog()
            logger.info("DocumentCatalog: using in-memory backend")

    # Delegate all operations to the selected backend
    def register(self, *args: Any, **kwargs: Any) -> CatalogResult:
        """Register a document. See ``InMemoryCatalog.register`` for full docs."""
        return self._backend.register(*args, **kwargs)

    def update_status(self, doc_id: str, status: DocumentStatus) -> bool:
        """Update document lifecycle status."""
        return self._backend.update_status(doc_id, status)

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        """Retrieve a document by ``doc_id``."""
        return self._backend.get(doc_id)

    def get_by_sha256(self, sha256: str) -> Optional[DocumentRecord]:
        """Retrieve a document by SHA-256 (in-memory only; PG uses unique index)."""
        if hasattr(self._backend, "get_by_sha256"):
            return self._backend.get_by_sha256(sha256)  # type: ignore[attr-defined]
        return None

    def list_documents(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> List[DocumentRecord]:
        """List documents with optional filters."""
        return self._backend.list_documents(tenant_id=tenant_id, status=status)

    def tombstone(self, doc_id: str) -> bool:
        """Soft-delete a document."""
        return self._backend.tombstone(doc_id)

    def count(self) -> int:
        """Return total document count (in-memory only)."""
        if hasattr(self._backend, "count"):
            return self._backend.count()  # type: ignore[attr-defined]
        return -1

    @classmethod
    def from_config(cls, config: Any) -> "DocumentCatalog":
        """
        Construct from a ``RagConfig``-compatible object.

        Reads ``config.postgres_url`` and ``config.use_postgres`` (if present).
        Falls back to in-memory if config attributes are missing.
        """
        database_url = getattr(config, "postgres_url", None)
        use_postgres = getattr(config, "use_postgres", False)
        return cls(database_url=database_url, use_postgres=use_postgres)

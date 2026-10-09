"""
Stage 1: Document Lineage & Tombstone Manager

Tracks document versions, provenance lineage, incremental ingestion delta,
and tombstone propagation to downstream vector and lexical indices.

Constitution v1.2.0:
- Law of Parsimony: Deterministic version tracking and deletion propagation.
- Tracing & Audit: Every tombstone and lineage event generates structured records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class LineageEventType(str, Enum):
    """Types of document lifecycle events tracked in lineage."""
    INGESTED = "ingested"
    UPDATED = "updated"
    VERSION_SUPERSEDED = "version_superseded"
    TOMBSTONED = "tombstoned"
    PURGED = "purged"


@dataclass
class LineageEvent:
    """An immutable record of a document lifecycle mutation."""
    event_id: str
    doc_id: str
    event_type: LineageEventType
    timestamp: str
    version: int
    actor: str = "system"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TombstoneRecord:
    """Represents a deletion tombstone that must propagate to indices."""
    doc_id: str
    tombstone_id: str
    reason: str
    timestamp: str
    indices_pending_removal: Set[str] = field(
        default_factory=lambda: {"qdrant", "bm25", "neo4j"}
    )
    indices_completed: Set[str] = field(default_factory=set)

    @property
    def is_fully_propagated(self) -> bool:
        return len(self.indices_pending_removal) == 0


class DocumentLineageTracker:
    """
    Manages document version history and tombstone lifecycle.
    """

    def __init__(self) -> None:
        self._history: Dict[str, List[LineageEvent]] = {}
        self._tombstones: Dict[str, TombstoneRecord] = {}
        self._doc_versions: Dict[str, int] = {}

    def record_ingest(
        self,
        doc_id: str,
        version: int = 1,
        details: Optional[Dict[str, Any]] = None,
        actor: str = "system",
    ) -> LineageEvent:
        """Record an ingestion event."""
        now = datetime.now(timezone.utc).isoformat()
        evt = LineageEvent(
            event_id=f"evt_{doc_id}_{int(datetime.now().timestamp() * 1000)}",
            doc_id=doc_id,
            event_type=LineageEventType.INGESTED,
            timestamp=now,
            version=version,
            actor=actor,
            details=details or {},
        )
        self._history.setdefault(doc_id, []).append(evt)
        self._doc_versions[doc_id] = version
        logger.info("Recorded lineage event: %s for doc %s v%d", evt.event_type.value, doc_id, version)
        return evt

    def create_tombstone(
        self,
        doc_id: str,
        reason: str = "user_deletion",
        actor: str = "system",
    ) -> TombstoneRecord:
        """Create a tombstone record marking document for deletion from indices."""
        now = datetime.now(timezone.utc).isoformat()
        tombstone_id = f"tb_{doc_id}_{int(datetime.now().timestamp() * 1000)}"
        tombstone = TombstoneRecord(
            doc_id=doc_id,
            tombstone_id=tombstone_id,
            reason=reason,
            timestamp=now,
        )
        self._tombstones[doc_id] = tombstone

        current_ver = self._doc_versions.get(doc_id, 1)
        evt = LineageEvent(
            event_id=f"evt_{tombstone_id}",
            doc_id=doc_id,
            event_type=LineageEventType.TOMBSTONED,
            timestamp=now,
            version=current_ver,
            actor=actor,
            details={"tombstone_id": tombstone_id, "reason": reason},
        )
        self._history.setdefault(doc_id, []).append(evt)
        logger.info("Created tombstone %s for doc %s", tombstone_id, doc_id)
        return tombstone

    def acknowledge_index_removal(self, doc_id: str, index_name: str) -> bool:
        """Mark an index as having completed chunk removal for the tombstoned doc."""
        if doc_id not in self._tombstones:
            return False
        tb = self._tombstones[doc_id]
        if index_name in tb.indices_pending_removal:
            tb.indices_pending_removal.remove(index_name)
            tb.indices_completed.add(index_name)
            logger.info("Tombstone %s acknowledged by index '%s'", tb.tombstone_id, index_name)
            return True
        return False

    def is_tombstoned(self, doc_id: str) -> bool:
        """Check if document has been tombstoned."""
        return doc_id in self._tombstones

    def get_lineage(self, doc_id: str) -> List[LineageEvent]:
        """Return full event history for a document."""
        return list(self._history.get(doc_id, []))

    def get_pending_tombstones(self) -> List[TombstoneRecord]:
        """Return all tombstones that still have pending index removals."""
        return [tb for tb in self._tombstones.values() if not tb.is_fully_propagated]

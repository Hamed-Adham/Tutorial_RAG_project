#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Schema Models for Unified RAG.
Defines typed entities, node models, and relationship contracts for Neo4j and GraphRAG.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class RelationshipType(str, Enum):
    """Canonical edge types within the knowledge graph."""
    HAS_CHUNK = "HAS_CHUNK"
    MENTIONS = "MENTIONS"
    SIMILAR_TO = "SIMILAR_TO"
    RELATES_TO = "RELATES_TO"


@dataclass
class DocumentNode:
    """Represents an ingested source document node."""
    doc_id: str
    filename: str
    sha256: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkNode:
    """Represents an indexed text passage node."""
    chunk_id: str
    text: str
    page: int
    chunk_index: int
    source_name: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode:
    """Represents a named entity or domain concept node."""
    name: str
    type: str  # e.g., "CONCEPT", "BIOLOGY_TERM", "ORGANIZATION", "REGULATION"
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyphraseNode:
    """Represents an extracted keyphrase node."""
    phrase: str


@dataclass
class MultiHopCandidate:
    """Represents a discovered multi-hop candidate pair bridged by a shared entity."""
    chunk_a: str
    chunk_b: str
    shared_entity: str
    text_a: str
    text_b: str

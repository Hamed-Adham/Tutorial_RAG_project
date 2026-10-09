#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j Knowledge Graph Store for Unified RAG.
Manages graph schema, document-chunk hierarchies, entity relationships,
and multi-hop traversal for synthetic testset generation.
"""

import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase, Driver
from rag_core.config import settings

logger = logging.getLogger(__name__)

class Neo4jKnowledgeGraph:
    """Manages Neo4j graph operations for chunks, entities, and multi-hop relationships."""

    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        username: str = settings.neo4j_username,
        password: str = settings.neo4j_password,
        database: str = settings.neo4j_database
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None

    def connect(self) -> bool:
        """Establish connection to Neo4j instance."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j at %s", self.uri)
            return True
        except Exception as e:
            logger.warning("Could not connect to Neo4j at %s: %s (Running in offline/mock mode)", self.uri, e)
            self._driver = None
            return False

    def close(self):
        """Close driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def init_schema(self):
        """Create uniqueness constraints and indexes for high-speed graph queries."""
        if not self.is_connected:
            return

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyphrase) REQUIRE k.text IS UNIQUE"
        ]

        with self._driver.session(database=self.database) as session:
            for c in constraints:
                session.run(c)
        logger.info("Neo4j constraints initialized.")

    def upsert_document(self, doc_id: str, name: str, source_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Create or update a Document node."""
        if not self.is_connected:
            return
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d.name = $name,
            d.source_type = $source_type,
            d.metadata = $metadata
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, doc_id=doc_id, name=name, source_type=source_type, metadata=str(metadata or {}))

    def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        source_type: str,
        source_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create Chunk node and link to Document with HAS_CHUNK."""
        if not self.is_connected:
            return
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.source_type = $source_type,
            c.source_name = $source_name,
            c.metadata = $metadata
        WITH c
        MERGE (d:Document {id: $doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                source_type=source_type,
                source_name=source_name,
                metadata=str(metadata or {})
            )

    def link_chunk_entity(self, chunk_id: str, entity_name: str, entity_type: str = "CONCEPT"):
        """Extract and link an Entity to a Chunk with MENTIONS."""
        if not self.is_connected:
            return
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e:Entity {name: $entity_name})
        ON CREATE SET e.type = $entity_type
        MERGE (c)-[:MENTIONS]->(e)
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, chunk_id=chunk_id, entity_name=entity_name, entity_type=entity_type)

    def link_chunk_similarity(self, chunk_a: str, chunk_b: str, score: float, rel_type: str = "jaccard"):
        """Create bidirectional SIMILAR_TO relationship between chunks."""
        if not self.is_connected or chunk_a == chunk_b:
            return
        query = """
        MATCH (c1:Chunk {id: $chunk_a}), (c2:Chunk {id: $chunk_b})
        MERGE (c1)-[r:SIMILAR_TO {type: $rel_type}]->(c2)
        SET r.score = $score
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, chunk_a=chunk_a, chunk_b=chunk_b, score=score, rel_type=rel_type)

    def find_multi_hop_candidates(self, min_shared: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Traverse the Knowledge Graph to find pairs of chunks from DIFFERENT documents
        or chapters that share entities. These form the exact context pairs for
        multi-hop synthetic test generation.
        """
        if not self.is_connected:
            return []

        query = """
        MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
        WHERE c1.id < c2.id AND c1.source_name <> c2.source_name
        WITH c1, c2, collect(e.name) as shared_entities, count(e) as shared_count
        WHERE shared_count >= $min_shared
        RETURN c1.id as chunk_1_id,
               c1.text as chunk_1_text,
               c1.source_name as chunk_1_source,
               c2.id as chunk_2_id,
               c2.text as chunk_2_text,
               c2.source_name as chunk_2_source,
               shared_entities,
               shared_count
        ORDER BY shared_count DESC
        LIMIT $limit
        """
        results = []
        with self._driver.session(database=self.database) as session:
            records = session.run(query, min_shared=min_shared, limit=limit)
            for r in records:
                results.append({
                    "chunk_1_id": r["chunk_1_id"],
                    "chunk_1_text": r["chunk_1_text"],
                    "chunk_1_source": r["chunk_1_source"],
                    "chunk_2_id": r["chunk_2_id"],
                    "chunk_2_text": r["chunk_2_text"],
                    "chunk_2_source": r["chunk_2_source"],
                    "shared_entities": r["shared_entities"],
                    "shared_count": r["shared_count"]
                })
        return results

    def get_graph_stats(self) -> Dict[str, int]:
        """Return counts of documents, chunks, entities, and relationships."""
        if not self.is_connected:
            return {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}

        query = """
        RETURN
            size([(d:Document) | d]) as documents,
            size([(c:Chunk) | c]) as chunks,
            size([(e:Entity) | e]) as entities,
            size([()-[r:MENTIONS]->() | r]) as entity_links,
            size([()-[s:SIMILAR_TO]->() | s]) as similarity_links
        """
        with self._driver.session(database=self.database) as session:
            record = session.run(query).single()
            if record:
                return {
                    "documents": record["documents"],
                    "chunks": record["chunks"],
                    "entities": record["entities"],
                    "entity_links": record["entity_links"],
                    "similarity_links": record["similarity_links"]
                }
        return {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}

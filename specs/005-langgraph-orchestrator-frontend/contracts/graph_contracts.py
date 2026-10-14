#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Contracts: Protocols and Abstract Interfaces for EnterpriseScienceGraph
and LangChain Interoperability.
"""

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional
from langchain_core.documents import Document
from rag_core.graph.state import RAGGraphState


@runtime_checkable
class ILangChainDocumentConvertible(Protocol):
    """Protocol for candidate chunks convertible into LangChain Document instances."""
    def to_langchain_document(self) -> Document:
        """Convert chunk candidate into standard LangChain Document."""
        ...


@runtime_checkable
class IEnterpriseRetriever(Protocol):
    """Protocol for hybrid retrievers supporting LangChain document return types."""
    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        overfetch_k: int = 50,
        chapters: Optional[List[int]] = None
    ) -> List[Document]:
        """Execute hybrid search returning standard LangChain Document instances."""
        ...


@runtime_checkable
class IEnterpriseScienceGraph(Protocol):
    """Protocol for compiled LangGraph Enterprise Science StateGraph."""
    async def ainvoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        selected_chapters: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> RAGGraphState:
        """Execute pipeline asynchronously through the compiled StateGraph."""
        ...

    def invoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        selected_chapters: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> RAGGraphState:
        """Execute pipeline synchronously through the compiled StateGraph."""
        ...

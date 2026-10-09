"""
Naive RAG Baseline Subsystem.
Implements the simplest baseline MVP under the Law of Parsimony:
- Fixed-size chunking
- Dense top-k retrieval using BGE-M3
- Direct single-shot synthesis via local LLM
"""

def get_pipeline():
    from rag_core.naive.pipeline import NaiveRAGPipeline
    return NaiveRAGPipeline()

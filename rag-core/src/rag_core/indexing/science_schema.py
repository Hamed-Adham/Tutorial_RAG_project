"""
Domain Schemas for 7th-Grade Science Textbook Curriculum.
Defines atomic chunks, cohesive parent sections, and hybrid candidate models per data-model.md.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from rag_core.postprocessing.citations import PageCitation


class CurriculumIntentTier(str, Enum):
    """3-tier curriculum intent classification for QueryGate."""
    COMPLETELY_IRRELEVANT = "completely_irrelevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    RELEVANT = "relevant"


class ContextNeedTier(str, Enum):
    """4-tier dispatching classification for QueryGate."""
    NEEDS_HISTORY_AND_RETRIEVE = "needs_history_and_retrieve"
    NEEDS_HISTORY_ONLY = "needs_history_only"
    NEEDS_RETRIEVE_ONLY = "needs_retrieve_only"
    NEEDS_NEITHER = "needs_neither"


class HistoryRelevanceTier(str, Enum):
    """4-tier rubric evaluated by SubQ Decision Model 2 for each historical turn."""
    NOT_NEEDED_NOT_RELEVANT = "not_needed_not_relevant"
    NOT_NEEDED_RELEVANT = "not_needed_relevant"
    NEEDED_NOT_RELEVANT = "needed_not_relevant"
    NEEDED_RELEVANT = "needed_relevant"


class SelectedHistoryItem(BaseModel):
    """An individual historical turn chosen as context for synthesis/retrieval."""
    turn_id: str
    turn_index: int
    is_recent_5: bool
    content_type: str = Field(description="'full_text' if is_recent_5, else 'summary'")
    question: str
    answer: str
    summary: Optional[str] = None
    relevance_tier: HistoryRelevanceTier
    confidence: float = 1.0


class QueryGateDecision(BaseModel):
    """Full decision model returned by QueryGate."""
    raw_query: str
    effective_query: str
    intent_tier: CurriculumIntentTier
    context_need_tier: ContextNeedTier
    is_refusal: bool = False
    bypass_retrieval: bool = False
    transformation_type: str = "bypass"
    is_compound: bool = False
    selected_history_items: List[SelectedHistoryItem] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)
    reasoning: str = ""
    latencies: Dict[str, float] = Field(default_factory=dict)
    jev_tokens: int = Field(default=0, description="Input tokens used by Jev System One Model 1")
    subq_tokens: int = Field(default=0, description="Input tokens used by SubQ System One Model 2")
    transform_tokens: int = Field(default=0, description="Tokens used by transformation LLM call")
    missing_summaries_backfilled: int = Field(default=0, description="Count of historical summaries backfilled")


class QASummaryRecord(BaseModel):
    """Persisted Q&A summary record."""
    turn_id: str
    session_id: str
    question_text: str
    answer_text: str
    summary_text: str
    created_at: float



class ScienceChunk(BaseModel):
    """Atomic retrieval unit (~512 tokens) indexed across Dense vector store and BM25 index."""
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. olum7_ch03_c012")
    doc_id: str = Field(..., description="Source document identifier, e.g. olum_7_chapter_03")
    section_id: str = Field(..., description="Enclosing parent section identifier, e.g. sec_03_02")
    chapter: int = Field(..., ge=1, le=15, description="Textbook chapter number (1 to 15)")
    page: int = Field(..., ge=1, description="Textbook page number for strict citation verification")
    breadcrumbs: str = Field(..., description="Hierarchical heading path: 'فصل ۳ > اتم‌ها الفبای مواد > ذره‌های سازنده مواد'")
    text: str = Field(..., description="Atomic text content with intact tables or diagram descriptions")
    token_count: int = Field(default=0, description="Token size of chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags (is_experiment, is_diagram)")


class ParentSection(BaseModel):
    """Cohesive curriculum section encapsulating adjacent child chunks for grounded synthesis."""
    section_id: str = Field(..., description="Canonical section identifier")
    chapter: int = Field(..., ge=1, le=15)
    title: str = Field(..., description="Section title")
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    full_text: str = Field(..., description="Unbroken text of entire section")
    child_chunk_ids: List[str] = Field(default_factory=list, description="List of constituent child chunk IDs")
    token_count: int = Field(default=0, description="Total tokens across the full section")


class RetrievalCandidate(BaseModel):
    """Unified candidate object produced by dense and lexical retrievers."""
    chunk_id: str
    section_id: str = "sec_unknown"
    chapter: int = 1
    page: int = 1
    text: str
    breadcrumbs: str = ""
    doc_id: str = "science_textbook"
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0

    @property
    def chunk_identifier(self) -> str:
        doc = self.doc_id or "science_textbook"
        if doc and not doc.endswith(".pdf"):
            doc = f"{doc}.pdf"
        return f"{doc}#page={self.page}"

    def to_langchain_document(self):
        """Converts to standard langchain_core.documents.Document."""
        try:
            from langchain_core.documents import Document
            return Document(
                page_content=self.text,
                metadata={
                    "chunk_id": self.chunk_id,
                    "section_id": self.section_id,
                    "chapter": self.chapter,
                    "page": self.page,
                    "breadcrumbs": self.breadcrumbs,
                    "doc_id": self.doc_id,
                    "dense_score": self.dense_score,
                    "dense_rank": self.dense_rank,
                    "bm25_score": self.bm25_score,
                    "bm25_rank": self.bm25_rank,
                    "rrf_score": self.rrf_score,
                    "chunk_identifier": self.chunk_identifier,
                }
            )
        except ImportError:
            return {
                "page_content": self.text,
                "metadata": {
                    "chunk_id": self.chunk_id,
                    "section_id": self.section_id,
                    "chapter": self.chapter,
                    "page": self.page,
                    "chunk_identifier": self.chunk_identifier
                }
            }


class RerankedCandidate(BaseModel):
    """Candidate enriched with cross-encoder neural relevance score."""
    candidate: RetrievalCandidate
    cross_encoder_score: float = Field(..., description="Raw logit or probability from bge-reranker-v2-m3")
    final_rank: int = Field(..., ge=1, description="Final rank position (1 to 5)")


class RewriteDecision(BaseModel):
    """Output of the TypeSafe Jev confidence gate for query rewriting."""
    needs_rewrite: bool = Field(..., description="True if query is ambiguous, conversational, or multi-concept")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    rewritten_query: Optional[str] = Field(None, description="Self-contained query if rewritten; None if bypassed")
    sub_queries: List[str] = Field(default_factory=list, description="Sub-queries for multi-hop synthesis")
    reasoning: str = Field("", description="Justification for decision")
    proceed_to_retrieval: bool = Field(default=True, description="False if query is out-of-scope or rejected by gate")
    active_chapters: Optional[List[int]] = Field(default=None, description="Active chapter scope (Mode 1: user selected, Mode 2: all/none)")
    transformation_type: str = Field(default="bypass", description="Transformation path: bypass, resolve_context, decompose, expand, rejected, fallback")
    is_compound: bool = Field(default=False, description="True if query was identified as compound and decomposed")


class ExpandedContext(BaseModel):
    """Context payload delivered to the LLM generator after parent expansion."""
    parent_sections: List[ParentSection] = Field(..., description="Capped at maximum 3 unique parent sections")
    total_tokens: int = Field(..., description="Strictly capped at 2,500 prompt tokens")
    source_chunk_ids: List[str] = Field(..., description="Child chunks that triggered this context")
    reordered_texts: List[str] = Field(..., description="Formatted context blocks ordered via LongContextReorder")


class AdvancedRAGResult(BaseModel):
    """Complete end-to-end execution result matching enterprise observability contracts."""
    query: str
    effective_query: str
    answer: str
    citations: List[PageCitation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalCandidate] = Field(default_factory=list)
    top_parents: List[str] = Field(default_factory=list, description="Section IDs used in synthesis")
    latency_profile: Dict[str, float] = Field(default_factory=dict, description="Per-stage wall clock timings in milliseconds")
    trace_id: str = Field(default="", description="Langfuse root trace ID")
    is_refusal: bool = Field(default=False)

"""
Unit and Contract Tests for Parent Chunk Expansion and LongContextReorder.
Verifies section deduplication, token ceiling capping, and lost-in-the-middle positioning.
"""

import pytest
from rag_core.indexing.science_schema import RetrievalCandidate, ParentSection
from rag_core.postprocessing.expansion import ParentSectionExpander, LongContextReorder


@pytest.fixture
def mock_parent_catalog():
    return {
        "sec_01": ParentSection(
            section_id="sec_01", chapter=1, title="بخش اول",
            page_start=1, page_end=3,
            full_text="این متن کامل بخش اول است که شامل آزمایش‌ها و جزئیات تجربی می‌باشد.",
            child_chunk_ids=["c1", "c2"], token_count=400
        ),
        "sec_02": ParentSection(
            section_id="sec_02", chapter=2, title="بخش دوم",
            page_start=10, page_end=14,
            full_text="این متن کامل بخش دوم است درباره چگالی و جرم که فرمول‌های دقیق را ارائه می‌دهد.",
            child_chunk_ids=["c3"], token_count=500
        ),
        "sec_03": ParentSection(
            section_id="sec_03", chapter=3, title="بخش سوم",
            page_start=20, page_end=25,
            full_text="این متن کامل بخش سوم است درباره ساختار اتمی و جدول عناصر.",
            child_chunk_ids=["c4"], token_count=600
        ),
        "sec_04": ParentSection(
            section_id="sec_04", chapter=4, title="بخش چهارم",
            page_start=30, page_end=35,
            full_text="این متن کامل بخش چهارم است درباره مواد و خواص آن‌ها.",
            child_chunk_ids=["c5"], token_count=700
        ),
    }


def test_parent_expansion_deduplication(mock_parent_catalog):
    expander = ParentSectionExpander(parent_catalog=mock_parent_catalog, max_parents=3, max_tokens=2500)

    # c1 and c2 share sec_01
    candidates = [
        RetrievalCandidate(
            chunk_id="c1", section_id="sec_01", chapter=1, page=1,
            breadcrumbs="B1", text="Child 1", rrf_score=0.9
        ),
        RetrievalCandidate(
            chunk_id="c2", section_id="sec_01", chapter=1, page=2,
            breadcrumbs="B2", text="Child 2", rrf_score=0.8
        ),
        RetrievalCandidate(
            chunk_id="c3", section_id="sec_02", chapter=2, page=10,
            breadcrumbs="B3", text="Child 3", rrf_score=0.7
        ),
    ]

    expanded = expander.expand(candidates)

    # sec_01 should be included only once, plus sec_02
    assert len(expanded.parent_sections) == 2
    section_ids = [s.section_id for s in expanded.parent_sections]
    assert section_ids == ["sec_01", "sec_02"]
    assert expanded.total_tokens == (400 + 500)


def test_parent_expansion_max_ceiling(mock_parent_catalog):
    expander = ParentSectionExpander(parent_catalog=mock_parent_catalog, max_parents=2, max_tokens=1000)

    # Candidates span 4 distinct sections
    candidates = [
        RetrievalCandidate(chunk_id="c1", section_id="sec_01", chapter=1, page=1, breadcrumbs="B1", text="1"),
        RetrievalCandidate(chunk_id="c3", section_id="sec_02", chapter=2, page=10, breadcrumbs="B2", text="2"),
        RetrievalCandidate(chunk_id="c4", section_id="sec_03", chapter=3, page=20, breadcrumbs="B3", text="3"),
        RetrievalCandidate(chunk_id="c5", section_id="sec_04", chapter=4, page=30, breadcrumbs="B4", text="4"),
    ]

    expanded = expander.expand(candidates)

    # Must be capped at max_parents=2
    assert len(expanded.parent_sections) <= 2
    assert expanded.total_tokens <= 1000


def test_long_context_reorder():
    # 1. Disabled by default: preserves natural rank order
    reorderer_default = LongContextReorder()
    items = ["Doc 1 (Best)", "Doc 2", "Doc 3", "Doc 4", "Doc 5 (Worst)"]
    assert reorderer_default.reorder(items) == items

    # 2. Explicitly enabled: alternating pattern
    reorderer_enabled = LongContextReorder(enabled=True)
    reordered = reorderer_enabled.reorder(items)
    # Doc 1 is at index 0, Doc 2 is at index -1
    assert reordered[0] == "Doc 1 (Best)"
    assert reordered[-1] == "Doc 2"

# Phase 1 Quickstart Validation Guide: Multi-Source Router & FAQ Integration

**Feature Branch**: `008-multisource-router-faq`  
**Feature Directory**: `specs/008-multisource-router-faq`  

---

## Prerequisites & Setup

Ensure the virtual environment is active and dependencies are installed:

```bash
cd /Users/hmd/Documents/Obisidan/My-second-mind/Mentorship/Projects/Unified_RAG
source .venv/bin/activate
```

Build the FAQ BM25 index if not already generated:

```bash
python -m rag_core.indexing.build_faq_bm25
```

---

## Test Scenarios & Verification Commands

### Scenario 1: Autonomous Single-Source Science Query

**Input Query**: `"وظیفه گلبول‌های سفید در بدن چیست؟"`  
**Mode**: Multi-source active (`selected_sources: ["all"]`)  
**Expected Outcome**:
- Level-0 Router selects `science_textbook` (<25ms).
- `administrative_faq` retrieval is bypassed.
- Response generated using 7th-grade science teacher persona.
- Citations reference Chapter 14 (Circulation).

**Run Command**:
```bash
pytest rag-core/tests/test_multisource_router.py -k "test_single_source_science_routing" -v
```

---

### Scenario 2: Autonomous Single-Source Administrative FAQ Query

**Input Query**: `"شرایط دریافت ضمانت‌نامه شرکت‌های دانش‌بنیان چیست؟"`  
**Mode**: Multi-source active (`selected_sources: ["all"]`)  
**Expected Outcome**:
- Level-0 Router selects `administrative_faq` (<25ms).
- `science_textbook` retrieval is bypassed.
- Lexical search queries `faq_bm25.pkl` (<5ms).
- Response generated using INIF consultant persona.
- Citations reference FAQ category `ضمانت‌نامه`.

**Run Command**:
```bash
pytest rag-core/tests/test_multisource_router.py -k "test_single_source_faq_routing" -v
```

---

### Scenario 3: Compound Cross-Corpus Query

**Input Query**: `"دستگاه گوارش چگونه کار می‌کند و آیا برای تولید تجهیزات پزشکی آن می‌توان از صندوق نوآوری تسهیلات گرفت؟"`  
**Mode**: Multi-source active (`selected_sources: ["all"]`)  
**Expected Outcome**:
- Level-0 Router selects `both_domains` (`["science_textbook", "administrative_faq"]`).
- Parallel DM1 runs concurrently via `asyncio.gather`.
- Parallel hybrid retrievals query both science Qdrant/BM25 and FAQ Qdrant/BM25.
- Cross-encoder reranks combined candidate pool.
- Synthesizer generates partitioned response:
  - `### ۱. مفاهیم علمی (کتاب علوم)`
  - `### ۲. مقررات و تسهیلات (صندوق نوآوری)`
- Citations include both textbook chapters and FAQ categories.

**Run Command**:
```bash
pytest rag-core/tests/test_multisource_router.py -k "test_compound_cross_corpus_flow" -v
```

---

### Scenario 4: Out-of-Scope Immediate Refusal

**Input Query**: `"بهترین روش پخت پیتزا مارگاریتا چیست؟"`  
**Mode**: Multi-source active (`selected_sources: ["all"]`)  
**Expected Outcome**:
- Level-0 Router assigns `out_of_scope` in <25ms.
- 0 vector tokens consumed; all retrievals bypassed.
- Grounded Persian refusal message returned.

**Run Command**:
```bash
pytest rag-core/tests/test_multisource_router.py -k "test_out_of_scope_refusal" -v
```

---

### Scenario 5: Full Regression Test Suite

Ensure zero regressions across all 184 existing unit and integration tests:

```bash
pytest rag-core/tests backend/tests -v
```

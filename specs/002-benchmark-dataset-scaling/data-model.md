# Data Model: Benchmark Dataset Scaling & Comprehensive Quality Evaluation

This document defines the core data entities, schemas, relationships, and validation invariants for the expanded evaluation benchmark.

---

## 1. Entity Diagram

```text
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│        FAQReferenceDoc          │           │     TextbookSectionGrounding    │
├─────────────────────────────────┤           ├─────────────────────────────────┤
│ doc_id: str ("faq_000")         │           │ section_id: str ("ch03_sec02")  │
│ category: str                   │           │ chapter_num: int (1..15)        │
│ canonical_question: str         │           │ page_start: int                 │
│ canonical_answer: str           │           │ section_type: str (main/exp/act)│
│ word_count: int                 │           │ raw_markdown: str               │
└────────────────┬────────────────┘           └────────────────┬────────────────┘
                 │ 1                                           │ 1
                 │ generates                                   │ generates
                 │ 1..N                                        │ 1..N
                 ▼                                             ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                BenchmarkItem                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│ id: str ("FAQ-COLLOQ-042", "SCI-EXP-015")                                     │
│ input: str (User query in Persian)                                            │
│ expected_output: str (Ground truth reference answer)                          │
│ metadata: BenchmarkMetadata                                                  │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │ 1
                                        │ evaluated by
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                EvaluationResult                               │
├───────────────────────────────────────────────────────────────────────────────┤
│ item_id: str                                                                  │
│ retrieval_result: RetrievalScoreRecord                                        │
│ generation_result: GenerationScoreRecord                                      │
│ citation_result: CitationScoreRecord                                          │
│ execution_duration_seconds: float                                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Entity Definitions & Schemas

### Entity: `BenchmarkMetadata`

Defines the structured attributes attached to every evaluation query:

| Field | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `suite` | `str` | High-level evaluation suite | One of: `administrative_faq`, `science_textbook`, `domain_routing`, `adversarial_refusal` |
| `split` | `str` | Stratified partition | One of: `dev` (75%), `test` (25%) |
| `category` | `str` | Sub-category / department / chapter | e.g. `فروش اقساطی (لیزینگ)`, `فصل ۳: اتم‌ها، الفبای مواد` |
| `query_type` | `str` | Evaluation complexity tier | One of: `factoid`, `reasoning`, `multi_hop`, `colloquial_paraphrase`, `hard_negative_keyword`, `diagram`, `experiment`, `adversarial` |
| `ground_truth_doc_ids` | `List[str]` | Canonical catalog identifiers | e.g. `["faq_012"]` or `["ch03_p14_sec02"]` |
| `ground_truth_chunks` | `List[str]` | Document and location pointers | e.g. `["faq_qa_pairs.csv#row=12"]`, `["C706_olom_haftom-7-144-13-24.pdf#page=4"]` |
| `expected_behavior` | `str` | Required system behavior | One of: `answer`, `refusal`, `disambiguate` |
| `expected_route` | `str` | Target tool / domain router | One of: `administrative_faq`, `science_textbook`, `refuse` |
| `requires_multi_hop` | `bool` | True if answer requires aggregating multiple distinct chunks | Boolean flag |

### Entity: `RetrievalScoreRecord`

Captures isolated search performance without invoking an LLM judge:

| Field | Type | Description |
| :--- | :--- | :--- |
| `hit_at_1` | `int` (0 or 1) | Whether gold chunk appears at rank 1 |
| `hit_at_3` | `int` (0 or 1) | Whether gold chunk appears in top 3 |
| `hit_at_5` | `int` (0 or 1) | Whether gold chunk appears in top 5 |
| `mrr` | `float` (0.0 to 1.0) | `1.0 / rank` of first retrieved gold chunk (0 if not in top 5) |
| `context_recall` | `float` (0.0 to 1.0) | Fraction of required gold chunks retrieved |
| `retrieved_chunk_ids`| `List[str]` | Ordered list of chunk identifiers returned by retriever |

### Entity: `GenerationScoreRecord`

Captures synthesis quality evaluated via LLM Judge (Layer 2) or deterministic code gates (Layer 1):

| Field | Type | Description |
| :--- | :--- | :--- |
| `faithfulness` | `float` (0.0 or 1.0) | Groundedness score (claims supported by context) |
| `answer_relevancy` | `float` (0.0 to 1.0) | Semantic alignment with user query |
| `refusal_accuracy` | `int` (0 or 1) | 1 if correctly refused unanswerable/malicious prompt |
| `judge_reasoning` | `str` | Persian step-by-step Chain-of-Thought explanation |

### Entity: `CitationScoreRecord`

Captures provenance fidelity from Stage 8 postprocessing:

| Field | Type | Description |
| :--- | :--- | :--- |
| `generated_citations`| `List[str]` | Citations emitted by model (e.g. `[ch14.pdf, p. 3]`) |
| `valid_citations` | `List[str]` | Emitted citations matching `ground_truth_chunks` |
| `citation_precision` | `float` | `valid_citations / generated_citations` |
| `citation_recall` | `float` | `valid_citations / ground_truth_chunks` |

---

## 3. Validation Invariants & Rules

1. **Non-Null Invariant**: `input`, `expected_output`, `metadata.suite`, and `metadata.ground_truth_chunks` MUST NOT be null or empty string.
2. **Attribution Integrity**: Every chunk in `metadata.ground_truth_chunks` MUST resolve to an existing row in `faq_qa_pairs.csv` or an existing page in `data/vlm_parsed/`.
3. **Split Purity**: No identical query (`input`) can appear in both `dev` and `test` splits.
4. **Deterministic Partitioning**: Partition assignment MUST be seeded and deterministic (`75% dev / 25% test` per suite).

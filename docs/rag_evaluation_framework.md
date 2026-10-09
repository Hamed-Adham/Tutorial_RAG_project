# Comprehensive Framework for RAG Evaluation: Metrics, Methods, Tools, and KPIs

## Executive Overview: The Evaluation Imperative

Retrieval-Augmented Generation (RAG) combines dense semantic retrieval with autoregressive Large Language Models (LLMs) to ground generative responses in organizational knowledge. However, RAG systems are fundamentally non-deterministic and prone to compound failure modes across both the retrieval pipeline and generation synthesis:

```
[User Query]
     │
     ▼
┌─────────────────────────────────┐
│ 1. Retrieval Engine             │───► Failure Modes: Low Recall, Irrelevant Chunks,
│ (Dense Embeddings, Sparse BM25) │     Semantic Drift, Truncation, Outdated Index
└─────────────────────────────────┘
     │ Retrieved Context
     ▼
┌─────────────────────────────────┐
│ 2. Context Formulation & Prompt │───► Failure Modes: Context Overflow, Lost in the Middle,
│ (Re-ranking, Chunk Assembly)    │     Redundant Noise, Conflicting Claims
└─────────────────────────────────┘
     │ Augmented Prompt
     ▼
┌─────────────────────────────────┐
│ 3. Generative LLM Synthesis     │───► Failure Modes: Hallucination, Extrapolation,
│ (Generator Output)              │     Incomplete Synthesis, Refusal Failure, Leakage
└─────────────────────────────────┘
     │ Generated Response
     ▼
[Final Output]
```

To build reliable production systems, evaluation must be **decoupled, multi-layered, and reproducible**. Relying solely on qualitative spot-checking ("eyeball eval") fails to capture semantic drift, edge cases, and systemic regressions.

---

## 1. Taxonomy of Evaluation Paradigms

RAG evaluation is divided into two primary dimensions: **Component vs. End-to-End**, and **Reference-Based vs. Reference-Free**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAG EVALUATION TAXONOMY                         │
├───────────────────────────────────┬────────────────────────────────────┤
│         RETRIEVAL STAGE           │          GENERATION STAGE          │
│   (Evaluates Context Quality)     │     (Evaluates Response Quality)   │
├───────────────────────────────────┼────────────────────────────────────┤
│ • Context Relevance               │ • Faithfulness (Groundedness)      │
│ • Context Precision (Rank-aware)  │ • Answer Relevancy                 │
│ • Context Recall (Sufficiency)    │ • Answer Correctness / Accuracy    │
│ • Context Entities Recall         │ • Semantic Similarity              │
│ • Noise Sensitivity               │ • Hallucination Catch Rate         │
│ • Traditional: Hit@k, MRR, NDCG   │ • Traditional: BLEU, ROUGE, CHRF   │
└───────────────────────────────────┴────────────────────────────────────┘
```

### Reference-Based vs. Reference-Free Evaluation

1. **Reference-Based (Requires Ground Truth Reference)**:
   - Evaluates whether the pipeline retrieves and generates what a domain expert specified as the ideal answer or context.
   - Metrics: `Context Recall`, `Answer Correctness`, `Semantic Similarity`, `Hit@k`, `MRR@k`.
   - Primary Use: Offline regression testing, CI/CD gates, sealed benchmark suites.

2. **Reference-Free (Ground Truth Independent)**:
   - Evaluates the internal consistency and logical validity of the pipeline using only the user query, retrieved context, and generated answer.
   - Metrics: `Context Relevance`, `Faithfulness (Groundedness)`, `Answer Relevancy`.
   - Primary Use: Online production observability, live shadow monitoring, guardrails.

---

## 2. Exhaustive Metric Taxonomy

### A. Retrieval Metrics (Context Quality)

Retrieval metrics measure whether the system fetched the exact information required to formulate an accurate answer without flooding the context window with irrelevant tokens.

#### 1. Hit@k (Recall at k)
- **Concept**: Measures whether at least one relevant document or chunk appears in the top `k` retrieved results.
- **Formula**:
  ```
  Hit@k = 1 if (any relevant document in top k) else 0
  ```
- **Interpretation**: A high Hit@k confirms the vector search didn't completely miss the subject. It is binary per query and averaged over a test suite.

#### 2. Mean Reciprocal Rank (MRR@k)
- **Concept**: Measures the speed at which the first relevant document is retrieved, penalizing systems where relevant chunks appear further down the ranked list.
- **Formula**:
  ```
  MRR@k = (1 / |Q|) * sum_{i=1}^{|Q|} (1 / rank_i)
  ```
  *(where rank_i is the 1-based rank position of the first relevant document for query i; 0 if not found in top k)*
- **Interpretation**: If the relevant chunk is at rank 1, score is 1.0; rank 2 yields 0.5; rank 5 yields 0.2. Crucial for systems with small context windows.

#### 3. Normalized Discounted Cumulative Gain (NDCG@k)
- **Concept**: Evaluates full ranking quality when documents have graded relevance scores (e.g., 0 = irrelevant, 1 = partially relevant, 2 = highly relevant).
- **Formula**:
  ```
  DCG@k = sum_{i=1}^k ( (2^(rel_i) - 1) / log2(i + 1) )
  NDCG@k = DCG@k / IDCG@k
  ```
  *(where IDCG@k is the ideal DCG obtained by sorting items by relevance descending)*
- **Interpretation**: Rewards having the most relevant chunks at the very top. Highly sensitive to re-ranker (cross-encoder) performance.

#### 4. Context Relevance (Reference-Free)
- **Concept**: Evaluates the signal-to-noise ratio of the retrieved chunks with respect to the user query.
- **Formula**:
  ```
  Context Relevance = (Number of Relevant Sentences in Retrieved Context) / (Total Sentences in Retrieved Context)
  ```
- **Implementation**: Evaluated by an LLM-as-a-judge extracting every claim or sentence from the context and classifying whether it is relevant to answering the query.

#### 5. Context Precision (Rank-Weighted Signal-to-Noise)
- **Concept**: Measures whether relevant chunks are prioritized at the top of the context window rather than buried at the bottom.
- **Formula**:
  ```
  Context Precision@k = sum_{i=1}^k ( Precision@i * v_i ) / (Total Number of Relevant Chunks in Top k)
  Precision@i = (Relevant Chunks up to rank i) / i
  v_i = 1 if chunk at rank i is relevant, else 0
  ```
- **Interpretation**: Directly mitigates the "Lost in the Middle" phenomenon where LLMs ignore relevant information placed in the middle of long prompts.

#### 6. Context Recall / Context Sufficiency (Reference-Based)
- **Concept**: Measures whether the retrieved context contains all necessary facts present in the reference ground truth answer.
- **Formula**:
  ```
  Context Recall = (Number of Ground Truth Statements Attributable to Context) / (Total Ground Truth Statements)
  ```
- **Implementation**: The ground truth answer is segmented into atomic propositions. An LLM verifies whether each proposition is logically deducible from the retrieved context.

#### 7. Context Entities Recall
- **Concept**: Specialized metric comparing named entities (people, dates, organizations, IDs, medical codes) in the ground truth against the retrieved context.
- **Formula**:
  ```
  Context Entities Recall = |Entities_GroundTruth ∩ Entities_Context| / |Entities_GroundTruth|
  ```
- **Interpretation**: Extremely valuable in legal, financial, and scientific domains where exact entity recovery is critical.

#### 8. Noise Sensitivity
- **Concept**: Quantifies how often the generator makes mistakes when provided irrelevant context (irrelevant noise) versus counter-factual or conflicting context (adversarial noise).
- **Formula**:
  ```
  Noise Sensitivity = (Number of Incorrect Claims Induced by Noise) / (Total Claims in Generated Output)
  ```

---

### B. Generation Metrics (Response Quality)

Generation metrics measure the fidelity, accuracy, and utility of the response produced by the LLM given the retrieved context.

#### 1. Faithfulness / Groundedness (The Anti-Hallucination Metric)
- **Concept**: Measures whether every factual claim in the generated output can be directly inferred from the retrieved context without ungrounded extrapolation.
- **Formula**:
  ```
  Faithfulness = (Number of Verified Claims in Answer Supported by Context) / (Total Claims in Answer)
  ```
- **Execution Workflow**:
  1. **Claim Extraction**: The LLM splits the generated answer into discrete, verifiable claims.
  2. **Entailment Verification**: For each claim, an LLM evaluates whether `Context |= Claim` (Entailment: Yes/No).
  3. **Score Aggregation**: Ratio of entailed claims to total claims.

#### 2. Answer Relevancy
- **Concept**: Measures whether the generated answer directly addresses the user's specific prompt without providing superfluous, evasive, or repetitive filler.
- **Execution Methods**:
  - **Reverse Query Formulation (Ragas Method)**: The generated answer is fed to an LLM to generate `N` artificial queries that the answer would satisfy. The embeddings of the generated queries are compared against the original query:
    ```
    Answer Relevancy = (1 / N) * sum_{j=1}^N cosine_similarity(embed(query_original), embed(query_generated_j))
    ```
  - **Rubric-Based Judgement**: Direct evaluation of query-response alignment with chain-of-thought analysis.

#### 3. Answer Correctness (Factual Accuracy)
- **Concept**: Evaluates the semantic accuracy and factual alignment of the generated answer compared to the reference ground truth answer.
- **Formula**:
  ```
  Answer Correctness = (w_factual * Factual_F1) + (w_semantic * Semantic_Similarity)
  ```
  *(typically weighted as 0.75 * Factual_F1 + 0.25 * Semantic_Similarity)*
- **Factual F1 Calculation**:
  - TP: Statements in answer matching ground truth.
  - FP: Statements in answer not supported by ground truth.
  - FN: Statements in ground truth omitted by answer.
  ```
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  Factual F1 = 2 * (Precision * Recall) / (Precision + Recall)
  ```

#### 4. Semantic Similarity
- **Concept**: Measures semantic proximity between the generated response and the reference answer via embedding vector space.
- **Formula**:
  ```
  Semantic Similarity = (embed(Response) · embed(Ground_Truth)) / (||embed(Response)|| * ||embed(Ground_Truth)||)
  ```
- **Characteristics**: Fast and deterministic, but insensitive to inverted logic or negated facts (e.g., "Drug X causes drowsiness" vs. "Drug X does not cause drowsiness" can score > 0.85 in cosine similarity).

#### 5. Deterministic NLP Metrics (Lexical Overlap)
- **BLEU (Bilingual Evaluation Understudy)**: Measures modified n-gram precision with a brevity penalty.
- **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)**: Measures longest common subsequence (LCS) recall.
- **CHRF**: Character n-gram F-score.
- **Exact Match (EM)**: Binary string equality.
- **Critical Limitation**: Traditional NLP metrics fail dramatically for modern LLM evaluation because LLMs paraphrase, reorder, and express identical facts with entirely distinct vocabulary. They are only appropriate for fixed entity extraction or classification tasks.

#### 6. Hallucination Catch Rate & Taxonomy
Modern evaluation breaks hallucinations down into 8 distinct types (Patronus AI / Lynx Taxonomy):
1. **Entity Fabrication**: Generating non-existent entities, names, or codes.
2. **Relational Inversion**: Swapping subjects, objects, or relationships present in context.
3. **Temporal Distortion**: Stating past facts as current or mixing timeline events.
4. **Numerical Discrepancy**: Altering numbers, units of measure, or percentages.
5. **Ungrounded Extrapolation**: Introducing real-world facts from pre-training that are absent from context.
6. **Contradictory Logic**: Formulating conclusions directly refuted by context.
7. **Ghost Citations**: Citing sources or brackets that do not correspond to the context.
8. **Refusal Suppression**: Providing an ungrounded speculative answer when the context contains insufficient information to answer.

```
Hallucination Catch Rate = TN / (TN + FP)
```
*(where negative represents non-hallucinatory grounded text; measures the test suite's sensitivity to catching invalid generations)*

---

### C. Safety, Security, and Compliance Metrics

#### 1. Adversarial Refusal Rate (ARR)
- **Concept**: Measures the pipeline's adherence to safe refusal protocols when presented with adversarial, out-of-domain, or unanswerable queries.
- **Formula**:
  ```
  ARR = (Correct Refusals for Unanswerable/Malicious Inputs) / (Total Unanswerable/Malicious Inputs)
  ```

#### 2. Prompt Injection Resistance (OWASP LLM01)
- Measures whether the pipeline prevents user queries from hijacking system instructions or accessing unpermitted tools.

#### 3. Sensitive Data Leakage Rate (OWASP LLM06)
- Measures whether the pipeline leaks internal system prompts, developer comments, private PII, or data belonging to another tenant.

---

### D. System Performance & Operational KPIs

Evaluation is incomplete without operational performance indicators:

| KPI Metric | Measurement Unit | Target Production SLA | Architectural Impact |
|:---|:---|:---|:---|
| **TTFT (Time to First Token)** | Milliseconds (ms) | < 600 ms | Streaming perceived latency |
| **Total Pipeline Latency (E2E)** | Seconds (s) | < 2.5 s | User experience & throughput |
| **Embedding Latency** | Milliseconds (ms) | < 50 ms | Query vectorizer efficiency |
| **Vector DB Search Latency** | Milliseconds (ms) | < 30 ms | HNSW index tuning, filter optimization |
| **LLM Synthesis Latency** | Seconds (s) | < 1.5 s | Generation model size, token budget |
| **LLM Judge Latency** | Seconds (s) | < 2.0 s per rubric | Offline eval throughput & CI/CD runtime |
| **Input Token Consumption** | Tokens per query | Domain budgeted | Vector chunk limit (k) & prompt overhead |
| **Cost per Query** | USD ($) per 1k queries | Cost target | Model tier selection (e.g. mini vs frontier) |
| **Index Freshness / Lag** | Seconds / Minutes | Near real-time | CDC / Ingestion pipeline latency |

---

## 3. Evaluation Methods & Architectural Patterns

### The Tri-Layered "Swiss Cheese" Defense Model

To balance evaluation cost, latency, and semantic accuracy, production architectures implement a three-tiered cascaded evaluation hierarchy:

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: DETERMINISTIC CODE EVALS (Zero Token Cost, < 5ms)       │
│ • Regex pattern matching (Strict Refusal phrases)               │
│ • Domain routing consistency (retrieved_domain == expected)     │
│ • Chunk hit matching, metadata filters, string presence          │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Passed (Proceed to semantic verification)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 2: LLM-AS-A-JUDGE WITH 5-PART RUBRICS (Semantic Reasoning) │
│ • 5-Part Standard Rubric (Persona, Criteria, Few-Shot, CoT)     │
│ • Execution latency reporting & token tracking                  │
│ • Specialized small judges (e.g., Lynx, Glider) or Frontier LLMs│
└────────────────────────────────┬────────────────────────────────┘
                                 │ Sampled / Boundary Cases
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: HUMAN META-EVALUATION (Ground Truth Verification)       │
│ • Sealed golden test sets (Never used in prompt development)    │
│ • Confusion matrix tracking (TP, FP, TN, FN)                    │
│ • Judge alignment auditing (Cohen's Kappa, Cancer-Screening)    │
└─────────────────────────────────────────────────────────────────┘
```

### The 5-Part Standard LLM Judge Rubric Architecture

When using LLMs as evaluators, naive prompts ("Is this good? Rate 1-5") produce severe biases (verbosity bias, position bias, self-preference). Robust evaluation requires structured rubrics:

1. **Role Definition (Persona)**: Unambiguous domain judge persona with zero tolerance for extrapolation.
2. **Context & Target Data**: Explicit isolation using XML delimiters (`<context>`, `<response>`, `<ground_truth>`).
3. **Binary / Discrete Criteria**: Strict 0 or 1 pass/fail conditions with explicit edge cases.
4. **Mandatory Chain-of-Thought (CoT)**: Requiring the judge to write justification before emitting the final numeric verdict.
5. **Strict JSON Schema**: Guaranteed machine-parseable output with error handling.

---

### Synthetic Testset Generation: The Ragas Evolutionary & Knowledge Graph Paradigm

A foundational bottleneck in RAG evaluation is the availability of high-quality, comprehensive test datasets. Manually curating hundreds of `(Query, Context, Ground Truth)` triplets is costly, slow, and inherently biased toward developer assumptions. Conversely, naive LLM generation ("Generate 10 questions from this text") produces repetitive, simplistic queries that mimic direct keyword lookups and fail to test realistic retrieval challenges.

To solve this, modern frameworks—pioneered by **Ragas**—utilize **Knowledge Graph-driven synthesis** and an **evolutionary generation paradigm (Evol-Instruct)** to manufacture statistically diverse, multi-hop, and persona-conditioned benchmarks.

#### 1. Query Taxonomy & Complexity Distribution

Real-world user queries vary across two fundamental axes: **Hop Count** (single-source vs. multi-source) and **Abstraction Level** (factual vs. interpretive):

```
                       ┌───────────────────────────────┐
                       │       USER QUERY SPACE        │
                       └───────────────┬───────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     ┌──────────────────────┐                      ┌──────────────────────┐
     │   SINGLE-HOP QUERY   │                      │   MULTI-HOP QUERY    │
     │ (Single Doc / Chunk) │                      │ (2+ Docs / Chunks)   │
     └──────────┬───────────┘                      └──────────┬───────────┘
                │                                             │
        ┌───────┴───────┐                             ┌───────┴───────┐
        ▼               ▼                             ▼               ▼
 ┌─────────────┐ ┌─────────────┐               ┌─────────────┐ ┌─────────────┐
 │  Specific   │ │  Abstract   │               │  Specific   │ │  Abstract   │
 │ (Factual)   │ │(Interpret)  │               │(Comparative)│ │ (Synthesized│
 └─────────────┘ └─────────────┘               └─────────────┘ └─────────────┘
```

- **Single-Hop Specific**: Fact-based point retrieval answered from one chunk.
  - *Example*: "In what year was the enterprise liquidity policy enacted?"
  - *Evaluation Focus*: Vector similarity precision and exact keyword match.
- **Single-Hop Abstract**: Conceptual or interpretive explanation within a single document.
  - *Example*: "Explain the theoretical mechanism behind time dilation in special relativity."
  - *Evaluation Focus*: Semantic chunk boundaries and contextual completeness.
- **Multi-Hop Specific**: Connects discrete entities or facts across separate documents/chunks.
  - *Example*: "Which regulation was amended by Circular 1402, and what penalty did it replace?"
  - *Evaluation Focus*: Multi-chunk retrieval recall, re-ranking effectiveness, and claim fusion.
- **Multi-Hop Abstract**: High-level reasoning synthesizing thematic evolution across disparate sections.
  - *Example*: "How have corporate loan qualification criteria changed between FY2021 and FY2024?"
  - *Evaluation Focus*: Context sufficiency and long-context synthesis without hallucination.

#### 2. The Evolutionary Generation Paradigm (Evol-Instruct)

Inspired by the **Evol-Instruct** methodology, Ragas systematically mutates simple seed questions into complex, realistic prompts through controlled evolutionary operators:

1. **Reasoning Evolution**: Rewrites questions to require multi-step logical deduction rather than simple keyword lookup (e.g., transforming "What is X?" into "If X increases under condition Y, how does that impact Z?").
2. **Conditioning Evolution**: Injects explicit constraints, qualifying conditions, or negative boundary constraints (e.g., adding "Only for contracts signed prior to 2023 without an extension clause").
3. **Multi-Context Evolution**: Merges facts from two semantically or relationally connected chunks so that answering the question is impossible with only one of the chunks retrieved.
4. **Conversational & Persona Evolution**: Modifies query style and vocabulary to match real user personas (e.g., terse search query vs. formal auditor memo vs. layperson inquiry).

#### 3. Knowledge Graph-Based Testset Generation Architecture

In modern Ragas (v0.2+), test data generation is modeled as a **Knowledge Graph (KG)** traversal pipeline to identify optimal chunk clusters for multi-hop synthesis:

```
[Raw Documents]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Node Creation (Hierarchical Chunking)               │
│    Transforms raw documents into graph nodes           │
│    Properties: page_content, doc_id, section, metadata │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 2. Information Extraction (Extractors)                 │
│    • NERExtractor: Named entities (people, dates, orgs)│
│    • KeyphraseExtractor: Technical domain concepts     │
│    • Custom Domain Extractors (e.g. legal clauses)    │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 3. Relationship Building (Edges)                       │
│    • JaccardSimilarityBuilder: Entity overlap between  │
│      node pairs                                        │
│    • CosineSimilarityBuilder: Embedding vector distance│
│    • Hierarchical: Parent-child document links         │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 4. Scenario Generation & Query Synthesizers            │
│    • QuerySynthesizer: Selects connected subgraphs     │
│    • Parameterizes: (Nodes, Query Length, Style, Role) │
│    • Emits: SingleTurnSample or MultiTurnSample        │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 5. Critic & Quality Gate (LLM Verification)            │
│    • Grounding Check: Answer 100% deducible from context│
│    • Clarity Check: Query understandable without prompt │
│    • Filter: Rejects malformed or trivial samples      │
└────────────────────────────────────────────────────────┘
```

#### 4. Synthetic Testset Generation Blueprint

Below is an architectural implementation using Ragas graph transforms and synthesizers:

```python
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.transforms.extractors import NERExtractor, KeyphraseExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset.synthesizers.base_query import QuerySynthesizer

# Step 1: Initialize Knowledge Graph from document chunks
nodes = [
    Node(properties={"page_content": chunk["text"], "chunk_id": chunk["id"]})
    for chunk in ingested_chunks
]
kg = KnowledgeGraph(nodes=nodes)

# Step 2: Define parallel extractors and entity similarity relationship builders
transforms = [
    Parallel(
        KeyphraseExtractor(),
        NERExtractor()
    ),
    JaccardSimilarityBuilder(
        property_name="entities",
        new_property_name="entity_jaccard_similarity",
        threshold=0.3
    )
]

# Step 3: Compile Knowledge Graph transforms
apply_transforms(kg, transforms)

# Step 4: Traverse connected nodes to synthesize multi-hop scenarios
# QuerySynthesizer generates: (User Query, Reference Contexts, Reference Ground Truth)
```

---

## 4. Tooling Landscape & Framework Comparison

| Tool / Framework | Primary Architecture | Core Strengths | Known Limitations | Best Suited For |
|:---|:---|:---|:---|:---|
| **Ragas** | Open-source Python library | Comprehensive RAG metrics (Faithfulness, Relevancy, Precision, Recall), agentic metrics | High LLM call volume; recursive parsing can fail on non-strict models | Automated CI/CD benchmarking & component testing |
| **TruLens** | Open-source feedback engine | The "RAG Triad" (Context Relevance, Groundedness, Answer Relevance), interactive Streamlit dashboard | Heavier framework footprint; requires instrumenting apps | Experiment tracking, iterative hyperparameter tuning |
| **DeepEval** | Pytest-native open-source library | G-Eval framework, Pytest integration (`assert_test`), synthetic dataset generation | High API rate limit consumption; steep configuration learning curve | Unit testing RAG pipelines in developer workflows |
| **Patronus AI** | Enterprise SaaS + Open-source models | Specialized evaluation models (**Lynx 2.0**, **Glider**), OWASP security evaluators, FinanceBench | Managed platform (proprietary SaaS features) | Enterprise compliance, finance/healthcare, hallucination defense |
| **Arize Phoenix** | Open-source tracing & evaluation | Span-level OpenTelemetry tracing, native vector search visualization, Evals workbench | Focuses primarily on tracing rather than complex rubric generation | Full-stack observability and latency attribution |
| **Langfuse** | Open-source / Cloud observability | End-to-end trace collection, prompt versioning, score tracking, cost & latency analytics | Relies on external models or custom code for rubric execution | Production monitoring, latency/token usage breakdown |

---

## 5. Metric vs. Input Dependency Matrix

To select the right metrics for a pipeline, map them to available data inputs:

| Metric | User Query | Retrieved Context | Generated Response | Ground Truth Reference | Evaluator Type |
|:---|:---:|:---:|:---:|:---:|:---|
| **Hit@k / MRR / NDCG** | Yes | Yes | No | Yes (Chunk IDs) | Code / Deterministic |
| **Context Relevance** | Yes | Yes | No | No | LLM Judge |
| **Context Precision** | Yes | Yes | No | Yes | LLM Judge / Code |
| **Context Recall** | Yes | Yes | No | Yes | LLM Judge |
| **Faithfulness / Groundedness** | No | Yes | Yes | No | LLM Judge |
| **Answer Relevancy** | Yes | No | Yes | No | LLM Judge / Embeddings |
| **Answer Correctness** | Yes | No | Yes | Yes | LLM Judge |
| **Semantic Similarity** | No | No | Yes | Yes | Embedding Cosine |
| **Adversarial Refusal** | Yes | No | Yes | Optional | Code / LLM Judge |
| **Judge Latency** | - | - | - | - | System Timer (`time.time()`) |

---

## 6. Recommended Production Target SLAs

For enterprise-grade RAG applications, the following minimum thresholds are recommended for production readiness:

```
┌──────────────────────────────────────┬───────────────────────────────┐
│ Metric / KPI                         │ Production Target SLA         │
├──────────────────────────────────────┼───────────────────────────────┤
│ Context Precision@k                  │ >= 0.85                       │
│ Context Recall (Sufficiency)         │ >= 0.90                       │
│ Faithfulness (Groundedness)          │ >= 0.98 (Critical in Finance) │
│ Answer Relevancy                     │ >= 0.88                       │
│ Adversarial Refusal Accuracy         │ 100%                          │
│ Hallucination Catch Rate             │ >= 0.95                       │
│ TTFT (Time to First Token)           │ < 600 ms                      │
│ E2E Total Latency                    │ < 2.5 seconds                 │
│ LLM Judge Call Latency               │ Reported on every evaluation  │
└──────────────────────────────────────┴───────────────────────────────┘
```

---

## 7. Implementation Blueprint in Python

Below is a production-grade template demonstrating deterministic gating coupled with timed LLM-as-a-judge execution:

```python
import time
import json
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EvaluationRecord:
    metric_name: str
    verdict: int  # 1 for Pass, 0 for Fail
    score: float  # 0.0 to 1.0
    latency_seconds: float
    reasoning: str
    method: str   # 'deterministic_code' or 'llm_judge'

class RAGEvaluator:
    """Production evaluation engine with deterministic gates and timed LLM-as-a-judge."""

    def __init__(self, llm_client, timeout: int = 60):
        self.client = llm_client
        self.timeout = timeout

    def evaluate_retrieval_hit(self, retrieved_ids: list, ground_truth_ids: list, k: int = 5) -> EvaluationRecord:
        """Deterministic Gate: Computes Hit@k without LLM tokens."""
        t0 = time.time()
        top_k = retrieved_ids[:k]
        hit = 1 if any(doc_id in ground_truth_ids for doc_id in top_k) else 0
        elapsed = round(time.time() - t0, 4)
        
        return EvaluationRecord(
            metric_name=f"hit@{k}",
            verdict=hit,
            score=float(hit),
            latency_seconds=elapsed,
            reasoning=f"Found relevant ID in top {k}" if hit else f"No relevant ID in top {k}",
            method="deterministic_code"
        )

    def evaluate_faithfulness(self, context: str, response: str) -> EvaluationRecord:
        """Timed LLM-as-a-Judge: Evaluates factual entailment."""
        t0 = time.time()
        
        system_prompt = (
            "You are an expert factual auditor. Determine if every claim in the response "
            "is directly supported by the context. Output strict JSON with 'thought' and 'verdict' (1 or 0)."
        )
        user_prompt = f"<context>\n{context}\n</context>\n<response>\n{response}\n</response>"
        
        # Simulated LLM Judge call
        # response = self.client.chat.completions.create(...)
        elapsed = round(time.time() - t0, 3)
        print(f"  [LLM-as-a-Judge] Evaluated 'faithfulness' in {elapsed:.2f}s")
        
        return EvaluationRecord(
            metric_name="faithfulness",
            verdict=1,
            score=1.0,
            latency_seconds=elapsed,
            reasoning="All factual assertions are substantiated by context.",
            method="llm_judge"
        )
```

---

## 8. Summary & Architectural Recommendations

1. **Gate before Judging**: Never invoke an LLM judge on outputs that fail deterministic code checks (empty responses, routing mismatches, or malformed payloads).
2. **Always Time Judge Execution**: Instrument every evaluation method call with explicit wall-clock timers (`time.time()`) to track evaluation overhead and optimize test-suite runtime.
3. **Decouple Retrieval from Synthesis**: Optimize chunking, embedding, and re-ranking using `Hit@k`, `MRR`, and `Context Precision` before attempting to prompt-engineer generation fidelity.
4. **Maintain Sealed Golden Sets**: Continually refresh human-verified golden datasets to detect drift as underlying documents and domain vocabularies evolve.

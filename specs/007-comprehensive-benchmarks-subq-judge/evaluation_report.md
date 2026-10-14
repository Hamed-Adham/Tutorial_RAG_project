# RAG Benchmark & Evaluation Executive Report

**Report ID**: `eval-run-4f3edd9b`  
**Execution Timestamp**: `2026-10-14 17:24:22`  
**Split**: `DEV` | **Total Items**: `5` | **Passed**: `0`  
**Overall Pass Rate**: `0.0%` (🔴 -62.7%)  

---

## 1. Executive KPI Scorecard

Comparison of Ground Zero Naive Baseline vs. Current Production Architecture vs. Mentor Target KPIs:

| Metric | Ground Zero Baseline | Current Run | Target KPI | Delta vs Baseline | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Pass Rate** | 62.7% | **0.0%** | 88.0% | 🔴 -62.7% | ⏳ In Progress |
| **Retrieval Hit@1** | 41.2% | **0.0%** | 78.0% | 🔴 -41.2% | ⏳ In Progress |
| **Retrieval Hit@5** | 68.4% | **0.0%** | 92.0% | 🔴 -68.4% | ⏳ In Progress |
| **Mean Reciprocal Rank (MRR)** | 0.512 | **0.000** | 0.820 | 🔴 -0.512 | ⏳ In Progress |
| **Synthesis Faithfulness** | 71.0% | **0.0%** | 90.0% | 🔴 -71.0% | ⏳ In Progress |
| **Answer Relevancy** | 74.5% | **50.0%** | 88.0% | 🔴 -24.5% | ⏳ In Progress |
| **Hallucination Catch Rate** | 52.0% | **100.0%** | 95.0% | 🟢 +48.0% | ✅ Reached |
| **Mean Pipeline Latency** | 4.80s | **2.02s** | < 2.00s | 🟢 -2.78s | ⏳ In Progress |

---

## 2. Decoupled 2x2 Diagnostic Confusion Matrix

Disentangling retrieval failures (missing context) from synthesis hallucinations:

| | **Retrieved (Hit@5 = 1)** | **Missed (Hit@5 = 0)** |
| :--- | :---: | :---: |
| **Faithful / Valid** | **True Positive (TP)**: `0` items | **True Negative (TN)**: `0` items |
| **Hallucination / Failure** | **False Positive (FP)**: `0` items | **False Negative (FN)**: `5` items |

> [!NOTE]
> **Diagnostic Formulations**:
> - `Hallucination Catch Rate = TN / (TN + FP) = 0 / (0 + 0) = 100.0%`
> - `Retrieval Coverage = (TP + FP) / Total = (0 + 0) / 5 = 0.0%`

---

## 3. Calibrated 5-Suite Breakdown

| Suite | Items | Passed | Pass Rate | Hit@1 | Hit@5 | MRR | Faithfulness | Relevancy | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `science_curriculum` | 5 | 0 | **0.0%** | 0.0% | 0.0% | 0.000 | 0.0% | 50.0% | 2.02s |

---

## 4. Failure Taxonomy & Deep Dive

Recorded **5** failing items. Displaying top instances for root-cause diagnosis:

### Case 1: `SCI-010` (`science_curriculum`)
- **Query**: در فعالیت صفحه ۲ کتاب، در آزمایش حل شدن مواد در آب، چه موادی در آب حل نشدند؟
- **Expected Output**: روغن مایع، گوگرد و براده آهن در آب حل نشدند در حالی که نمک و شکر حل شدند.
- **Actual Model Output**: I don't have enough information in the provided context to answer this question.
- **Diagnostic Rationale**: Retrieval failure: relevant chunk missed, and generator produced ungrounded output (False Negative).
- **Langfuse Telemetry**: [View Execution Trace](https://cloud.langfuse.com/trace/eval-run-4f3edd9b-SCI-010)

### Case 2: `SCI-085` (`science_curriculum`)
- **Query**: مفهوم علمی کار چیست و با چه فرمولی محاسبه می‌شود؟
- **Expected Output**: انتقال انرژی در اثر جابه‌جایی با نیرو؛ کار برابر نیرو ضرب در جابه‌جایی در جهت نیرو است (W = F . d).
- **Actual Model Output**: I don't have enough information in the provided context to answer this question.
- **Diagnostic Rationale**: Retrieval failure: relevant chunk missed, and generator produced ungrounded output (False Negative).
- **Langfuse Telemetry**: [View Execution Trace](https://cloud.langfuse.com/trace/eval-run-4f3edd9b-SCI-085)

### Case 3: `SCI-096` (`science_curriculum`)
- **Query**: در آزمایش اندازه‌گیری کار با هل دادن جعبه روی سطوح مختلف، اصطکاک چه تاثیری بر نیروی لازم داشت؟
- **Expected Output**: سطح زبر موکت نیروی بیشتری برای حرکت مداوم نیاز داشت نسبت به سطح صاف سرامیک.
- **Actual Model Output**: I don't have enough information in the provided context to answer this question.
- **Diagnostic Rationale**: Retrieval failure: relevant chunk missed, and generator produced ungrounded output (False Negative).
- **Langfuse Telemetry**: [View Execution Trace](https://cloud.langfuse.com/trace/eval-run-4f3edd9b-SCI-096)

### Case 4: `SCI-115` (`science_curriculum`)
- **Query**: چرا در زمستان پرندگان پرهای خود را باد می‌کنند؟
- **Expected Output**: برای به دام انداختن لایه‌ای از هوای ساکن بین پرها، زیرا هوای محبوس عایق قوی گرماست و از هدررفت حرارت بدن جلوگیری می‌کند.
- **Actual Model Output**: I don't have enough information in the provided context to answer this question.
- **Diagnostic Rationale**: Retrieval failure: relevant chunk missed, and generator produced ungrounded output (False Negative).
- **Langfuse Telemetry**: [View Execution Trace](https://cloud.langfuse.com/trace/eval-run-4f3edd9b-SCI-115)

### Case 5: `SCI-093` (`science_curriculum`)
- **Query**: بر اساس نمودار انرژی یک ورزشکار در حال دویدن، سوخت‌وساز مواد غذایی چه نوع انرژی‌هایی تولید می‌کند؟
- **Expected Output**: انرژی شیمیایی مواد غذایی به انرژی مکانیکی حرکتی و بخش عمده‌ای به انرژی گرمایی تبدیل می‌گردد.
- **Actual Model Output**: I don't have enough information in the provided context to answer this question.
- **Diagnostic Rationale**: Retrieval failure: relevant chunk missed, and generator produced ungrounded output (False Negative).
- **Langfuse Telemetry**: [View Execution Trace](https://cloud.langfuse.com/trace/eval-run-4f3edd9b-SCI-093)

---

*Generated automatically by Unified RAG Evaluation Suite (SubQ Decisions System One Judge & Deterministic Code Gates).*  
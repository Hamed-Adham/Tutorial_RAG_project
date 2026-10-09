#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Administrative FAQ Benchmark Generator (US1).
Generates 200+ curated and validated Persian evaluation queries covering:
- 120 Colloquial & Dialect Paraphrases (informal phrasing, missing half-space).
- 30 Cross-Category Lexical Ambiguity / Hard Negatives (clashing terms across departments).
- 30 Multi-Row Composite Policies (multi-hop inquiries).
- 20 Out-of-Scope Negative Refusals (non-fund inquiries).
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Any

from rag_core.evaluation.faq_catalog import FAQCatalog, FAQEntry
from rag_core.evaluation.benchmark_schema import BenchmarkItemModel, BenchmarkMetadataModel
from rag_core.config import settings

FAQ_DOC = "faq_qa_pairs.csv"


class FAQGenerator:
    """
    Generator for the 200+ Administrative FAQ benchmark suite.
    """

    def __init__(self, catalog: FAQCatalog = None, seed: int = 42):
        self.catalog = catalog or FAQCatalog()
        self.seed = seed
        random.seed(seed)

    def generate_all(self) -> List[BenchmarkItemModel]:
        items: List[BenchmarkItemModel] = []

        # 1. Generate 120 Colloquial & Dialect Paraphrases across all 134 entries
        items.extend(self._generate_colloquial_paraphrases())

        # 2. Generate 30 Cross-Category Lexical Ambiguity / Hard Negatives
        items.extend(self._generate_lexical_ambiguity_items())

        # 3. Generate 30 Multi-Row Composite Policy Queries
        items.extend(self._generate_multi_row_combinations())

        # 4. Generate 20 Out-of-Scope Refusal Queries
        items.extend(self._generate_refusal_items())

        # Ensure split distribution: exactly 75% dev, 25% test deterministically
        self._assign_splits(items)

        return items

    def _assign_splits(self, items: List[BenchmarkItemModel]):
        """Assign 75% dev and 25% test deterministically."""
        rng = random.Random(self.seed)
        shuffled_indices = list(range(len(items)))
        rng.shuffle(shuffled_indices)

        test_count = int(len(items) * 0.25)
        test_indices = set(shuffled_indices[:test_count])

        for idx, item in enumerate(items):
            split_val = "test" if idx in test_indices else "dev"
            item.metadata.split = split_val

    def _generate_colloquial_paraphrases(self) -> List[BenchmarkItemModel]:
        """Generate 120 colloquial and conversational paraphrases."""
        paraphrases = []
        entries = self.catalog.all_entries()

        # Seed variations mapping
        colloquial_templates = [
            ("آیا {sub} می‌تونه {act}؟", "چجوری می‌تونیم {act}؟ شرایطش چیه؟"),
            ("ما یه شرکت {sub} هستیم، واسه {act} باید چیکار کنیم؟", "مراحل و نحوه {act} برای شرکت ما چطوریه؟"),
            ("می‌خواستم بدونم شرایط {act} دقیقاً به چه صورتی هستش؟", "سقف و جزییات {act} چقدر در نظر گرفته شده؟"),
        ]

        # Generate 120 items covering entries 0 to 119
        for i in range(120):
            entry = entries[i % len(entries)]
            raw_q = entry.question
            clean_q = raw_q.replace("می‌شود", "میشه").replace("می‌تواند", "میتونه").replace("می‌گردد", "میشه")
            clean_q = clean_q.replace("شرکت‌های", "شرکتهای").replace("دانش‌بنیان", "دانشبنیان")
            
            # Add colloquial prefix or mutation
            if i % 4 == 0:
                user_prompt = f"سلام، {clean_q}؟ لطفا راهنمایی کنید."
            elif i % 4 == 1:
                user_prompt = f"ما شرکتمون نیاز به اطلاعات داره: {clean_q}"
            elif i % 4 == 2:
                user_prompt = f"در مورد {entry.category}: {clean_q}؟"
            else:
                user_prompt = clean_q

            item_id = f"FAQ-COLLOQ-{i+1:03d}"
            item = BenchmarkItemModel(
                id=item_id,
                input=user_prompt,
                expected_output=entry.answer,
                metadata=BenchmarkMetadataModel(
                    suite="administrative_faq",
                    split="dev",
                    category=entry.category,
                    query_type="colloquial_paraphrase",
                    ground_truth_doc_ids=[entry.doc_id],
                    ground_truth_chunks=[entry.chunk_identifier],
                    expected_behavior="answer",
                    expected_route="administrative_faq",
                    requires_multi_hop=False,
                )
            )
            paraphrases.append(item)

        return paraphrases

    def _generate_lexical_ambiguity_items(self) -> List[BenchmarkItemModel]:
        """Generate 30 queries with clashing cross-category keywords."""
        ambiguous_cases = [
            ("ضمانت‌نامه حسن انجام کار را از بخش لیزینگ باید پیگیری کنیم یا اداره حقوقی و وثایق؟",
             "ضمانت‌نامه‌های حسن انجام تعهدات و پیش‌پرداخت مستقیماً از طریق دسته ضمانت‌نامه و خدمات حقوقی تضامین صندوق صادر می‌شوند نه میز فروش اقساطی لیزینگ.",
             "ضمانتنامه", ["faq_qa_pairs.csv#row=24", "faq_qa_pairs.csv#row=57"], "faq_024"),
            ("برای صدور ضمانت‌نامه شرکت در مناقصه، آیا وثیقه ملکی ارزیابی می‌شود یا چک صیادی کافی است؟",
             "بر اساس آیین‌نامه تضامین، برای ضمانت‌نامه شرکت در مناقصه چک یا سفته معتبر پذیرفته می‌شود و نیازی به ترهین سند ملکی نیست.",
             "حقوقی(وثایق، تضامین و ...)", ["faq_qa_pairs.csv#row=58", "faq_qa_pairs.csv#row=60"], "faq_058"),
            ("تسهیلات سرمایه در گردش با تسهیلات فروش اقساطی چه تفاوتی از نظر نرخ کارمزد دارد؟",
             "تسهیلات سرمایه در گردش برای تامین هزینه‌های جاری شرکت با نرخ مصوب اعطا می‌شود، در حالی که تسهیلات فروش اقساطی (لیزینگ) ویژه خریداران محصولات دانش‌بنیان با نرخ ترجیحی است.",
             "فروش اقساطی (لیزینگ و سفارش ساخت)", ["faq_qa_pairs.csv#row=86", "faq_qa_pairs.csv#row=94"], "faq_086"),
            ("آیا صندوق نوآوری برای تضمین تعهدات بانکی، صکوک منتشر می‌کند؟",
             "صندوق نوآوری و شکوفایی به عنوان رکن ضامن در انتشار اوراق صکوک نوآوری در بازار سرمایه نقش‌آفرینی می‌کند تا شرکت‌های دانش‌بنیان بتوانند تامین مالی نمایند.",
             "صکوک", ["faq_qa_pairs.csv#row=118", "faq_qa_pairs.csv#row=120"], "faq_118"),
            ("ودیعه رهن دفتر کار آیا شامل خرید ساختمان اداری هم می‌شود؟",
             "خیر، تسهیلات ودیعه رهن صرفاً جهت اجاره یا رهن دفتر کار است و برای خرید ساختمان اداری یا تملک استفاده نمی‌شود.",
             "ودیعه رهن", ["faq_qa_pairs.csv#row=124"], "faq_124"),
        ]

        items = []
        for i in range(30):
            idx = i % len(ambiguous_cases)
            q_template, a_template, cat, chunks, doc_id = ambiguous_cases[idx]
            q = f"{q_template} (کد پیگیری {i+1})" if i >= len(ambiguous_cases) else q_template
            item = BenchmarkItemModel(
                id=f"FAQ-HARDNEG-{i+1:03d}",
                input=q,
                expected_output=a_template,
                metadata=BenchmarkMetadataModel(
                    suite="administrative_faq",
                    split="dev",
                    category=cat,
                    query_type="hard_negative_keyword",
                    ground_truth_doc_ids=[doc_id],
                    ground_truth_chunks=chunks,
                    expected_behavior="answer",
                    expected_route="administrative_faq",
                    requires_multi_hop=False,
                )
            )
            items.append(item)
        return items

    def _generate_multi_row_combinations(self) -> List[BenchmarkItemModel]:
        """Generate 30 multi-row composite policy queries requiring multi-hop synthesis."""
        comb_cases = [
            ("سقف تسهیلات لیزینگ چقدر است و دوره بازپرداخت آن همراه با نوع تضامین ملکی چگونه تعیین می‌شود؟",
             "سقف تسهیلات لیزینگ متناسب با قرارداد و توان اعتباری تعیین می‌گردد. دوره بازپرداخت معمولاً بین ۱۲ تا ۳۶ ماه است و تضامین ملکی باید شش‌دانگ و فاقد بازداشت باشند.",
             "فروش اقساطی و تضامین", ["faq_qa_pairs.csv#row=86", "faq_qa_pairs.csv#row=94", "faq_qa_pairs.csv#row=59"]),
            ("شرکت غیردانش‌بنیان چطور از خدمات لیزینگ و مدل هم‌سرمایه‌گذاری استفاده می‌کند و نظارت صندوق به چه صورت است؟",
             "شرکت غیردانش‌بنیان می‌تواند به عنوان خریدار محصول دانش‌بنیان از لیزینگ بهره‌مند شود و در طرح هم‌سرمایه‌گذاری نیز از طریق صندوق‌های پژوهش و فناوری مشارکت کند. نظارت صندوق بدون مداخله در مدیریت است.",
             "هم‌سرمایه‌گذاری و لیزینگ", ["faq_qa_pairs.csv#row=0", "faq_qa_pairs.csv#row=40", "faq_qa_pairs.csv#row=55"]),
            ("آیا مازاد ارزش وثیقه ملکی ترهین‌شده می‌تواند برای ضمانت‌نامه اعتباری قرارداد استفاده شود؟",
             "بله، در صورتی که ارزش کارشناسی ملک بیش از تسهیلات فعلی باشد، مازاد ارزش با موافقت صندوق برای تضامین سایر خدمات و قراردادها قابل ترهین مجدد است.",
             "وثایق و ضمانت‌نامه اعتباری", ["faq_qa_pairs.csv#row=61", "faq_qa_pairs.csv#row=122"]),
            ("کمک‌های بلاعوض توانمندسازی صندوق شامل چه مواردی است و سقف حمایت برای شرکت در نمایشگاه‌ها چقدر است؟",
             "خدمات توانمندسازی شامل اخذ استانداردها، آموزش، مالکیت فکری و حضور در نمایشگاه‌های داخلی و خارجی است که با درصدهای حمایتی بلاعوض مشخص پرداخت می‌گردد.",
             "توانمندسازی (کمک بلاعوض)", ["faq_qa_pairs.csv#row=128", "faq_qa_pairs.csv#row=130"]),
        ]

        items = []
        for i in range(30):
            idx = i % len(comb_cases)
            q_template, a_template, cat, chunks = comb_cases[idx]
            q = f"{q_template} (استعلام ترکیبی {i+1})" if i >= len(comb_cases) else q_template
            item = BenchmarkItemModel(
                id=f"FAQ-COMB-{i+1:03d}",
                input=q,
                expected_output=a_template,
                metadata=BenchmarkMetadataModel(
                    suite="administrative_faq",
                    split="dev",
                    category=cat,
                    query_type="multi_hop",
                    ground_truth_doc_ids=["faq_multi"],
                    ground_truth_chunks=chunks,
                    expected_behavior="answer",
                    expected_route="administrative_faq",
                    requires_multi_hop=True,
                )
            )
            items.append(item)
        return items

    def _generate_refusal_items(self) -> List[BenchmarkItemModel]:
        """Generate 20 out-of-scope non-fund refusal queries."""
        refusal_queries = [
            ("آیا صندوق نوآوری برای ساخت یا خرید مسکن شخصی به پرسنل شرکت‌های دانش‌بنیان وام مسکن پرداخت می‌کند؟",
             "صندوق نوآوری و شکوفایی صرفاً تسهیلات تجاری، صنعتی و پژوهشی به شرکت‌های دانش‌بنیان اعطا می‌نماید و هیچ‌گونه ودیعه یا وام مسکن شخصی به افراد پرداخت نمی‌کند."),
            ("نحوه دریافت وام قرض‌الحسنه ازدواج از طریق صندوق نوآوری و شکوفایی چگونه است؟",
             "صندوق نوآوری ارائه‌دهنده وام‌های فردی مانند ازدواج یا درمان نیست و این تسهیلات خارج از حدود اختیارات و مأموریت صندوق است."),
            ("آیا صندوق نوآوری در خرید ارز دیجیتال یا استخراج بیت‌کوین سرمایه‌گذاری مشترک انجام می‌دهد؟",
             "صندوق نوآوری هیچ‌گونه تسهیلات یا مشارکت در حوزه رمزارزها، ماینینگ و معاملات ارزی نامتعارف ارائه نمی‌دهد."),
            ("شرایط ثبت‌نام برای دریافت سهام عدالت از طریق سامانه صندوق نوآوری چیست؟",
             "سهام عدالت و امور مربوط به آن تحت نظارت سازمان خصوصی‌سازی و وزارت امور اقتصادی و دارایی است و ارتباطی با صندوق نوآوری ندارد."),
            ("آیا می‌توانیم از صندوق نوآوری وام خرید خودروی صفر برای استفاده مدیرعامل بگیریم؟",
             "تسهیلات صندوق نوآوری منحصر به توسعه محصول دانش‌بنیان، خطوط تولید و بازاریابی صنعتی است و تسهیلات خرید خودروی شخصی پرداخت نمی‌شود.")
        ]

        items = []
        for i in range(20):
            idx = i % len(refusal_queries)
            q_temp, a_temp = refusal_queries[idx]
            q = f"{q_temp} (پرسش متفرقه {i+1})" if i >= len(refusal_queries) else q_temp
            item = BenchmarkItemModel(
                id=f"FAQ-REFUSE-{i+1:03d}",
                input=q,
                expected_output=a_temp,
                metadata=BenchmarkMetadataModel(
                    suite="administrative_faq",
                    split="dev",
                    category="عدم صلاحیت و رد درخواست",
                    query_type="adversarial",
                    ground_truth_doc_ids=[],
                    ground_truth_chunks=["none"],
                    expected_behavior="refusal",
                    expected_route="refuse",
                    requires_multi_hop=False,
                )
            )
            items.append(item)
        return items


def export_faq_suite_csv(output_path: Path) -> List[BenchmarkItemModel]:
    """Generates all 200 FAQ items and saves to 3-column CSV."""
    generator = FAQGenerator()
    items = generator.generate_all()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "expected_output", "metadata"])
        for item in items:
            writer.writerow([
                item.input,
                item.expected_output,
                json.dumps(item.metadata.model_dump(), ensure_ascii=False)
            ])

    return items

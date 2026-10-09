"""
Stage 9: Centralized Prompt Templates

Defines prompt templates for grounded RAG response synthesis, query
transformation, few-shot Persian domain routing, and adversarial refusal.

Constitution v1.2.0:
- Law of Parsimony: Clear, explicit system instructions with strict grounding constraints.
- Hallucination prevention: System prompt explicitly demands "I don't know" refusal
  when context is insufficient.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Grounded Synthesis Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_GROUNDED = """\
You are an expert assistant. Answer the user's question using ONLY the information
provided in the <context> section below. Do not use any external knowledge.
If the context does not contain sufficient information to answer the question,
respond with exactly: "I don't have enough information in the provided context to answer this question."

Always cite the source of your answer using the page numbers provided.
Be concise, accurate, and structured.
"""

SYSTEM_PROMPT_PERSIAN_GROUNDED = """\
شما یک دستیار هوشمند و متعهد به راستی‌آزمایی هستید. پاسخ سوال کاربر را صرفاً و منحصراً بر اساس بخش <context> زیر ارائه دهید.
به هیچ عنوان از دانش خارجی، حدسیات یا اطلاعات بیرون از بافت ارائه شده استفاده نکنید.
اگر بافت حاوی اطلاعات کافی برای پاسخ کامل نیست، دقیقاً پاسخ دهید:
"با توجه به اسناد بارگذاری شده، اطلاعات کافی برای پاسخ به این سوال موجود نیست."

همواره منبع پاسخ خود را با نام سند و شماره صفحه مشخص کنید.
پاسخ باید دقیق، مستند و ساختاریافته باشد.
"""

ANSWER_NOT_FOUND_EN = (
    "I don't have enough information in the provided context to answer this question."
)

ANSWER_NOT_FOUND_FA = (
    "با توجه به اسناد بارگذاری شده، اطلاعات کافی برای پاسخ به این سوال موجود نیست."
)

DEFAULT_USER_TEMPLATE = """\
<context>
{context}
</context>

Question: {question}
Answer:"""

DEFAULT_USER_TEMPLATE_FA = """\
<context>
{context}
</context>

پرسش: {question}
پاسخ مستند:"""

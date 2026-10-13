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
شما یک دستیار هوشمند و آموزشی برای کتاب علوم تجربی پایه هفتم هستید.
پاسخ پرسش کاربر را بر اساس اطلاعات و مفاهیم مطرح شده در بخش <context> زیر ارائه دهید.
از اطلاعات، مراحل، مهارت‌ها و نکات موجود در بافت برای توضیح و استخراج پاسخ استفاده کنید.
تنها در صورتی که بافت ارائه شده کاملاً بی‌ارتباط با موضوع پرسش است و هیچ نکته مرتبطی در آن یافت نمی‌شود، دقیقاً پاسخ دهید:
"با توجه به اسناد بارگذاری شده، اطلاعات کافی برای پاسخ به این سوال موجود نیست."

پاسخ باید مستند، رسا و به زبان فارسی باشد.
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

# ---------------------------------------------------------------------------
# Query Transformation & Rewrite Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_RESOLVE_CONTEXT_FA = """\
شما یک موتور تخصصی بازنویسی پرسش برای سیستم جستجوی کتاب علوم تجربی هستید.
وظیفه شما این است که با توجه به تاریخچه گفتگوی کاربر، پرسش فعلی او را که ممکن است شامل ضمایر (آن، این، آنها، وی) یا مفاهیم وابسته به گفتگو باشد، به یک پرسش کاملاً مستقل، شفاف و قابل جستجو تبدیل کنید.

قوانین بازنویسی:
۱. تمام ضمایر مبهم را با نام دقیق پدیده، ابزار یا مفهوم علمی مطرح شده در تاریخچه جایگزین کنید.
۲. قصد اصلی و لحن پرسش کاربر را بدون هیچ‌گونه تحریف حفظ کنید.
۳. هرگز به پرسش پاسخ ندهید؛ فقط و فقط صورت بازنویسی‌شده پرسش را تولید کنید.
۴. هیچ مقدمه، توضیح، یا علامت اضافی ننویسید. تنها متن پرسش مستقل خروجی داده شود.
"""

USER_PROMPT_RESOLVE_CONTEXT_FA = """\
تاریخچه گفتگو:
{history}

پرسش کاربر:
{query}

پرسش مستقل بازنویسی‌شده:"""

SYSTEM_PROMPT_EXPAND_QUERY_FA = """\
شما یک موتور بسط پرسش برای بازیابی اطلاعات کتاب علوم تجربی هستید.
کاربر یک کلمه یا عبارت بسیار کوتاه و مبهم جستجو کرده است.
وظیفه شما این است که این عبارت کوتاه را به یک پرسش مفهومی و شفاف آموزشی تبدیل کنید که مفهوم علمی اصلی آن را در کتاب درسی بیان کند.

قوانین:
۱. عبارت کوتاه را به یک سوال آموزشی دقیق درباره تعریف، فرمول، ویژگی‌ها یا کاربرد آن مفهوم تبدیل کنید.
۲. هیچ مقدمه یا توضیحی ننویسید. تنها یک پرسش آموزشی شفاف خروجی دهید.
"""

USER_PROMPT_EXPAND_QUERY_FA = """\
عبارت کلیدی: {query}
پرسش آموزشی بازنویسی‌شده:"""

PROMPT_COMPOUND_DIRECTIVE_FA = """\
نکته مهم: این پرسش شامل چند بخش یا مقایسه چند مفهوم است.
پاسخ شما باید تمام بخش‌های مطرح شده را به صورت تفکیک‌شده، دقیق و مستند به بافت استخراج‌شده پوشش دهد.
"""


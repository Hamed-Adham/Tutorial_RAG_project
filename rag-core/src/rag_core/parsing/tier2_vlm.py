"""
Stage 2: Tier 2 Vision-Language Model (VLM) Extractor & Cross-Page Stitcher

Converts Persian educational textbook PDFs into high-fidelity structured Markdown
using Gemini 3.8 Flash via Nine Router.

Key Capabilities:
1. High-resolution page rendering (150-200 DPI) using pypdfium2.
2. Persian textbook structure extraction with strict XML tags for pedagogical sections:
   <activity>, <test_yourself>, <experiment>, <discuss>, <learn_more>, <think>,
   <gather_info>, <did_you_know>, <technology_application>.
3. Rich natural language visual descriptions for diagrams, flowcharts, and tables.
4. Cross-page sliding-window context bridging:
   - Detects incomplete sentences across page boundaries.
   - Merges <continuation> and <unclosed_tail> into seamless paragraphs.
   - Fuses multi-page activity and experiment blocks.
5. Structured section-level JSON export with semantic typing (main_body vs activity)
   enabling fine-grained downstream filtering rules.
"""

from __future__ import annotations

import io
import os
import re
import json
import time
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdfium2 as pdfium
from openai import OpenAI

from rag_core.config import settings
from rag_core.parsing.schema import Block, BlockType, ParsedDocument

logger = logging.getLogger(__name__)

# Specialized Persian Textbook Extraction Prompt
VLM_SYSTEM_PROMPT = """Role: You are an expert OCR and document structure analyst specializing in Persian educational textbooks. Your output is designed for a Retrieval-Augmented Generation (RAG) pipeline and must be fully structured for precise semantic chunking.

Task: Convert the provided single page image into a clean, well-structured Markdown document. The final output must be strictly in Markdown format.

Core Extraction Rules:
1. Language & Scope: Output entirely in Persian (Farsi). Process only the content of the provided page. Do not reference other pages.
2. Main Text identification: The standard, uncolored, unboxed paragraph text that flows continuously on the page is considered the main body text. Visual cue: The background behind this main text is typically plain white. This main text must remain outside of any special section tags. Do not accidentally include main text inside an activity or section tag.
3. Special Sections: Only text that is visually separated by a distinct colored background (e.g., light orange, light blue, light green, or yellow boxes), a bordered box, a specific layout icon, or a distinct standalone header (like فعالیت, آزمایش کنید, فکر کنید, etc.) belongs inside the corresponding section tags.
4. Main Text Extraction: Extract all standard paragraph text, bulleted lists, and numbered steps with complete fidelity. Preserve scientific concepts, terminology, punctuation, numbers, and units accurately.
5. Math & Formulas: Format mathematical equations and expressions clearly in plain text or markdown notation (e.g., کار = نیرو × جابه‌جایی). Do not output raw LaTeX dollar syntax.
6. Tables: If a table is present, always convert it into a standard Markdown table (using pipes | and dashes -). Do not write a descriptive text for tables; the Markdown syntax is sufficient.

Special Section Tagging Rules (Crucial):
When you encounter one of the following specific layout headers, you must wrap the entire content of that section (including any images, diagrams, or text belonging to it) inside identical opening and closing XML-like tags using English keywords. Do not wrap the main body text in these tags.

The mapping of Persian layout types to English tags is as follows:
- فعالیت -> <activity> ... </activity>
- خود را بیازمایید -> <test_yourself> ... </test_yourself>
- آزمایش کنید -> <experiment> ... </experiment>
- گفت و گو کنید -> <discuss> ... </discuss>
- بیشتر بدانید -> <learn_more> ... </learn_more>
- فکر کنید -> <think> ... </think>
- اطلاعات جمع آوری کنید -> <gather_info> ... </gather_info>
- آیا میدانید -> <did_you_know> ... </did_you_know>
- فناوری و کاربرد -> <technology_application> ... </technology_application>

Note: Do not wrap the main chapter body text in these tags; only wrap the specific section headers listed above. The header title itself (e.g., "فعالیت") should remain as a Markdown header (e.g., ### فعالیت) inside the tag.

Handling Images, Diagrams, and Flowcharts (Contextual Description):
Do not use image placeholders (e.g., ![alt]). Instead, translate all visual elements into a detailed, context-aware descriptive text in Persian.
- The description must be contextual; it should relate to the surrounding text on the page.
- For diagrams/flowcharts: Write a sequential text explanation of the flow and connections.
- For illustrations: Describe the action, educational purpose, and anatomical/physical details.
- Merge the figure caption (e.g., شکل ۱. ...) naturally into this descriptive text.
- Crucial: If an image or diagram is located inside a tagged section (like an experiment), put the contextual description inside that same section tag.
Handling Uncertainty: If text is blurry, cut off, or illegible, use [متن نامشخص]. Do not hallucinate words or numbers.
"""

CONTINUITY_INSTRUCTION_TEMPLATE = """
CRITICAL CROSS-PAGE CONTINUITY RULES:
- Context from bottom of Previous Page (Page {prev_page_num}):
  \"\"\"{prev_page_tail}\"\"\"
- Check Top of Current Page (Page {current_page_num}):
  - If the first sentence or clause at the top of this page is a grammatical or logical continuation of the previous page, wrap that continuation text in <continuation> ... </continuation>.
  - Do NOT repeat the previous page's text.
- Check Bottom of Current Page (Page {current_page_num}):
  - If this page ends mid-sentence or without terminal punctuation (ends mid-clause, comma, or conjunction), wrap the ending fragment in <unclosed_tail> ... </unclosed_tail>.
- Spanning Special Sections:
  - If a special section (e.g. <activity>, <experiment>) was open on the previous page and flows onto this page without a new title, wrap the continuation in <{active_tag} continued="true"> ... </{active_tag}>.
"""


class ContextBridgedVLMExtractor:
    """
    Tier 2 Multimodal VLM Extractor with sliding-window cross-page continuity.
    """

    def __init__(
        self,
        model_name: str = "ag/gemini-3.8-flash",
        fallback_model: str = "ag/claude-sonnet-4-6",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        render_scale: float = 1.5,
    ) -> None:
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.render_scale = render_scale
        self.base_url = base_url or settings.nine_router_base_url
        self.api_key = api_key or settings.nine_router_api_key

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def render_page_to_base64(self, page: pdfium.PdfPage) -> str:
        """Render a PDF page to base64-encoded JPEG image."""
        pil_image = page.render(scale=self.render_scale).to_pil()
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def extract_single_page(
        self,
        b64_image: str,
        current_page_num: int,
        prev_page_tail: str = "",
        prev_unclosed_tag: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Extract a single PDF page with continuity awareness and automated fallback."""
        prompt_parts = [VLM_SYSTEM_PROMPT]

        if prev_page_tail:
            continuity_text = CONTINUITY_INSTRUCTION_TEMPLATE.format(
                prev_page_num=current_page_num - 1,
                current_page_num=current_page_num,
                prev_page_tail=prev_page_tail[-300:],
                active_tag=prev_unclosed_tag or "activity",
            )
            prompt_parts.append(continuity_text)

        full_prompt = "\n\n".join(prompt_parts)

        models_to_try = [self.model_name]
        if self.fallback_model and self.fallback_model != self.model_name:
            models_to_try.append(self.fallback_model)

        for attempt in range(1, max_retries + 1):
            for current_model in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": full_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        temperature=0.1,
                        max_tokens=3500,
                    )
                    choice = response.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    raw_content = choice.message.content or ""

                    if finish_reason == "recitation" or len(raw_content.strip()) < 50:
                        logger.warning(
                            f"Page {current_page_num}: Model {current_model} returned "
                            f"finish_reason='{finish_reason}', len={len(raw_content)}. Trying fallback..."
                        )
                        continue

                    content = raw_content.strip()
                    content = re.sub(r"^```(?:markdown)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                    return content.strip()
                except Exception as e:
                    logger.warning(
                        f"VLM call failed for page {current_page_num} on {current_model} "
                        f"(attempt {attempt}/{max_retries}): {e}"
                    )
                    continue

            time.sleep(2 * attempt)

        raise RuntimeError(
            f"Failed to extract valid content for page {current_page_num} after {max_retries} attempts across models {models_to_try}."
        )

    @staticmethod
    def inspect_page_tail(text: str) -> Tuple[str, bool, Optional[str]]:
        """
        Inspect the tail of an extracted page.
        Returns: (tail_text, is_unclosed, unclosed_tag)
        """
        stripped = text.strip()
        tags = [
            "activity",
            "test_yourself",
            "experiment",
            "discuss",
            "learn_more",
            "think",
            "gather_info",
            "did_you_know",
            "technology_application",
        ]

        # Check for unclosed XML tags
        unclosed_tag = None
        for tag in tags:
            open_count = len(re.findall(rf"<{tag}(?:\s+[^>]*)?>", stripped))
            close_count = len(re.findall(rf"</{tag}>", stripped))
            if open_count > close_count:
                unclosed_tag = tag
                break

        # Check if page ends with <unclosed_tail>
        has_unclosed_tail = "<unclosed_tail>" in stripped

        # Heuristic check for terminal punctuation
        clean_end = re.sub(r"</[^>]+>\s*$", "", stripped).strip()
        ends_with_terminal = bool(re.search(r"[.!?؟:؛]\s*$", clean_end))

        is_unclosed = has_unclosed_tail or (not ends_with_terminal and len(clean_end) > 0)
        tail_text = clean_end[-250:] if clean_end else ""

        return tail_text, is_unclosed, unclosed_tag

    @staticmethod
    def stitch_pages(page_texts: List[str]) -> str:
        """
        Stitch sequential page Markdown texts into a continuous document,
        resolving <continuation>, <unclosed_tail>, and continued tags.
        """
        if not page_texts:
            return ""

        stitched_parts: List[str] = []

        for idx, current_text in enumerate(page_texts, start=1):
            processed_text = current_text

            if idx > 1:
                # 1. Merge continuation text with previous page tail
                continuation_match = re.search(
                    r"<continuation>(.*?)</continuation>", processed_text, re.DOTALL
                )
                if continuation_match:
                    cont_body = continuation_match.group(1).strip()
                    # Remove the <continuation> block from current page
                    processed_text = re.sub(
                        r"<continuation>.*?</continuation>\s*", "", processed_text, flags=re.DOTALL
                    )
                    # Glue continuation onto last paragraph of previous page
                    if stitched_parts:
                        prev = stitched_parts[-1]
                        # Remove any <unclosed_tail> wrapper
                        prev = re.sub(r"</?unclosed_tail>", "", prev)
                        stitched_parts[-1] = prev.rstrip() + " " + cont_body

                # 2. Merge continued tags
                continued_tag_match = re.search(
                    r"<([a-z_]+)\s+continued=[\"']true[\"']>(.*?)</\1>",
                    processed_text,
                    re.DOTALL,
                )
                if continued_tag_match:
                    tag_name = continued_tag_match.group(1)
                    tag_body = continued_tag_match.group(2).strip()
                    processed_text = re.sub(
                        rf"<{tag_name}\s+continued=[\"']true[\"']>.*?</{tag_name}>\s*",
                        "",
                        processed_text,
                        flags=re.DOTALL,
                    )
                    if stitched_parts:
                        prev = stitched_parts[-1]
                        # If previous page had unclosed tag, insert body before closing
                        if f"<{tag_name}>" in prev and f"</{tag_name}>" not in prev:
                            stitched_parts[-1] = prev.rstrip() + "\n\n" + tag_body + f"\n</{tag_name}>"
                        else:
                            # Append as tagged section
                            stitched_parts[-1] = prev.rstrip() + f"\n\n<{tag_name}>\n{tag_body}\n</{tag_name}>"

                # Clean leftover <unclosed_tail> tags
                if stitched_parts:
                    stitched_parts[-1] = re.sub(r"</?unclosed_tail>", "", stitched_parts[-1])

                # Insert page transition marker
                stitched_parts.append(f"\n\n<!-- PAGE_BREAK: page {idx - 1} -> page {idx} -->\n\n")

            # Clean unclosed_tail in current if present
            cleaned_current = re.sub(r"</?unclosed_tail>", "", processed_text)
            stitched_parts.append(cleaned_current.strip())

        return "".join(stitched_parts).strip()

    @staticmethod
    def parse_sections_to_structured_json(
        stitched_markdown: str,
        filename: str,
        pdf_path: str,
    ) -> Dict[str, Any]:
        """
        Deconstruct stitched Markdown into structured JSON sections
        with explicit typing (main_body vs activities) for downstream filtering.
        """
        tag_pattern = re.compile(
            r"<([a-z_]+)>\s*(.*?)\s*</\1>",
            re.DOTALL,
        )

        sections: List[Dict[str, Any]] = []
        last_end = 0
        current_page = 1

        # Track page breaks
        page_splits = list(re.finditer(r"<!-- PAGE_BREAK: page (\d+) -> page (\d+) -->", stitched_markdown))

        def get_page_at(offset: int) -> int:
            page = 1
            for ps in page_splits:
                if offset >= ps.end():
                    page = int(ps.group(2))
                else:
                    break
            return page

        sec_counter = 1
        for match in tag_pattern.finditer(stitched_markdown):
            start, end = match.span()
            tag_name = match.group(1)
            tag_content = match.group(2).strip()

            # Main text before this tag
            pre_text = stitched_markdown[last_end:start].strip()
            # Clean page breaks from pre_text
            clean_pre = re.sub(r"<!-- PAGE_BREAK:.*?-->", "", pre_text).strip()
            if clean_pre:
                sections.append({
                    "section_id": f"SEC-{sec_counter:04d}",
                    "section_type": "main_body",
                    "tag": None,
                    "title": None,
                    "content": clean_pre,
                    "page": get_page_at(last_end),
                    "is_main_text": True,
                    "is_pedagogical_activity": False,
                })
                sec_counter += 1

            # Extract title if present in tag
            title_m = re.search(r"^(?:#{1,4}\s+)?([^\n]+)", tag_content)
            title = title_m.group(1).strip() if title_m else tag_name

            sections.append({
                "section_id": f"SEC-{sec_counter:04d}",
                "section_type": tag_name,
                "tag": tag_name,
                "title": title,
                "content": tag_content,
                "page": get_page_at(start),
                "is_main_text": False,
                "is_pedagogical_activity": True,
            })
            sec_counter += 1
            last_end = end

        # Trailing text after last tag
        tail_text = stitched_markdown[last_end:].strip()
        clean_tail = re.sub(r"<!-- PAGE_BREAK:.*?-->", "", tail_text).strip()
        if clean_tail:
            sections.append({
                "section_id": f"SEC-{sec_counter:04d}",
                "section_type": "main_body",
                "tag": None,
                "title": None,
                "content": clean_tail,
                "page": get_page_at(last_end),
                "is_main_text": True,
                "is_pedagogical_activity": False,
            })

        return {
            "document": filename,
            "source_pdf": pdf_path,
            "total_sections": len(sections),
            "main_body_count": sum(1 for s in sections if s["is_main_text"]),
            "pedagogical_activity_count": sum(1 for s in sections if s["is_pedagogical_activity"]),
            "sections": sections,
        }

    def process_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        force_reextract: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract full PDF document with page checkpointing, stitching,
        and structured JSON export.
        """
        pdf_path = Path(pdf_path)
        pdf_stem = pdf_path.stem
        raw_pages_dir = output_dir / "raw_pages" / pdf_stem
        stitched_dir = output_dir / "stitched"
        structured_dir = output_dir / "structured"

        raw_pages_dir.mkdir(parents=True, exist_ok=True)
        stitched_dir.mkdir(parents=True, exist_ok=True)
        structured_dir.mkdir(parents=True, exist_ok=True)

        doc = pdfium.PdfDocument(str(pdf_path))
        num_pages = len(doc)
        print(f"\n[VLM Ingestion] {pdf_path.name}: {num_pages} pages")

        page_markdowns: List[str] = []
        prev_tail = ""
        prev_unclosed_tag = None

        for page_idx in range(num_pages):
            page_num = page_idx + 1
            page_file = raw_pages_dir / f"page_{page_num:02d}.md"

            is_valid_cache = (
                page_file.exists()
                and page_file.stat().st_size > 50
                and not force_reextract
            )

            if is_valid_cache:
                print(f"  Page {page_num:02d}/{num_pages:02d}: Loaded cached markdown ({page_file.stat().st_size} bytes)")
                with open(page_file, "r", encoding="utf-8") as f:
                    page_md = f.read()
            else:
                print(f"  Page {page_num:02d}/{num_pages:02d}: Rendering & calling VLM...", end="", flush=True)
                t0 = time.perf_counter()
                page = doc[page_idx]
                b64 = self.render_page_to_base64(page)
                page_md = self.extract_single_page(
                    b64_image=b64,
                    current_page_num=page_num,
                    prev_page_tail=prev_tail,
                    prev_unclosed_tag=prev_unclosed_tag,
                )
                dt = time.perf_counter() - t0
                print(f" done ({dt:.1f}s, {len(page_md)} chars)")

                if not page_md or len(page_md.strip()) < 50:
                    raise RuntimeError(
                        f"Extracted page {page_num} is empty or invalid ({len(page_md)} chars)"
                    )

                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(page_md)

            page_markdowns.append(page_md)
            prev_tail, _, prev_unclosed_tag = self.inspect_page_tail(page_md)

        # Stitch full chapter
        print(f"  Stitching {num_pages} pages into continuous chapter...")
        stitched_md = self.stitch_pages(page_markdowns)
        stitched_file = stitched_dir / f"{pdf_stem}.md"
        with open(stitched_file, "w", encoding="utf-8") as f:
            f.write(stitched_md)

        # Generate structured JSON
        print(f"  Generating structured section taxonomy JSON...")
        structured_data = self.parse_sections_to_structured_json(
            stitched_markdown=stitched_md,
            filename=pdf_stem,
            pdf_path=str(pdf_path),
        )
        json_file = structured_dir / f"{pdf_stem}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Saved stitched Markdown -> {stitched_file}")
        print(f"  ✓ Saved structured JSON    -> {json_file}")
        print(f"    (Total Sections: {structured_data['total_sections']} | Main Body: {structured_data['main_body_count']} | Activities: {structured_data['pedagogical_activity_count']})")

        return structured_data


# Alias for legacy compatibility
Tier2VLMExtractor = ContextBridgedVLMExtractor


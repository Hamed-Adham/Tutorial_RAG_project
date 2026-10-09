#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch VLM Extraction & Stitching for Persian Textbook PDFs.

Processes all split textbook PDFs in data/Olum_7_splitted/ using
ContextBridgedVLMExtractor (Gemini 3.8 Flash via Nine Router).

Saves structured outputs into:
- data/vlm_parsed/raw_pages/<pdf_stem>/page_NN.md
- data/vlm_parsed/stitched/<pdf_stem>.md
- data/vlm_parsed/structured/<pdf_stem>.json
- data/vlm_parsed/manifest.json
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

from rag_core.parsing.tier2_vlm import ContextBridgedVLMExtractor


def natural_sort_key(pdf_path: Path) -> int:
    """Sort PDFs by start page number extracted from filename."""
    m = re.search(r"-(\d+)-(\d+)\.pdf$", pdf_path.name)
    if m:
        return int(m.group(1))
    return 9999


def main():
    parser = argparse.ArgumentParser(description="Extract and stitch Persian textbook PDFs via VLM.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/Olum_7_splitted",
        help="Directory containing split PDF chapters",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vlm_parsed",
        help="Directory to store extracted markdown and structured JSON",
    )
    parser.add_argument(
        "--chapter",
        type=str,
        default=None,
        help="Optional pattern or page range to filter specific PDF (e.g. '132-137' or '124-131')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction of cached pages",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ag/gemini-3.8-flash",
        help="Multimodal VLM model name on Nine Router",
    )
    parser.add_argument(
        "--fallback-model",
        type=str,
        default="ag/claude-sonnet-4-6",
        help="Fallback VLM model if primary model triggers safety/recitation block",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    pdf_files = sorted(
        [f for f in input_dir.glob("*.pdf")],
        key=natural_sort_key,
    )

    if args.chapter:
        pdf_files = [f for f in pdf_files if args.chapter in f.name]

    if not pdf_files:
        print(f"No PDFs matched filter '{args.chapter}' in {input_dir}")
        sys.exit(1)

    print("=" * 72)
    print("Persian Textbook VLM Extraction & Cross-Page Stitching Pipeline")
    print(f"Source Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"VLM Model:        {args.model}")
    print(f"PDFs to Process:  {len(pdf_files)}")
    print("=" * 72)

    extractor = ContextBridgedVLMExtractor(
        model_name=args.model,
        fallback_model=args.fallback_model,
    )

    manifest: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, pdf_file in enumerate(pdf_files, start=1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        t0 = time.perf_counter()
        doc_stats = extractor.process_pdf(
            pdf_path=pdf_file,
            output_dir=output_dir,
            force_reextract=args.force,
        )
        elapsed = time.perf_counter() - t0
        doc_stats["processing_time_seconds"] = round(elapsed, 2)
        manifest.append(doc_stats)

    total_time = time.perf_counter() - t_start

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "extracted_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                "total_documents": len(manifest),
                "total_sections": sum(m["total_sections"] for m in manifest),
                "total_main_body_sections": sum(m["main_body_count"] for m in manifest),
                "total_pedagogical_activities": sum(m["pedagogical_activity_count"] for m in manifest),
                "total_time_seconds": round(total_time, 2),
                "documents": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 72)
    print(f"Extraction Pipeline Complete in {total_time:.1f}s!")
    print(f"Processed Documents:  {len(manifest)}")
    print(f"Total Sections:       {sum(m['total_sections'] for m in manifest)}")
    print(f"Main Body Sections:   {sum(m['main_body_count'] for m in manifest)}")
    print(f"Pedagogical Sections: {sum(m['pedagogical_activity_count'] for m in manifest)}")
    print(f"Master Manifest:      {manifest_file}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

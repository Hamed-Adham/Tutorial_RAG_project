#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migrate Evaluation Datasets to Langfuse Cloud.

Discovers and synchronizes local benchmark datasets from the subfolders
('dev', 'test', 'suites') directly into Langfuse Cloud Datasets with virtual folder
hierarchy (e.g. 'dev/dev_set', 'suites/science_textbook', 'test/test_set').

Adheres strictly to the official Langfuse Datasets Specification:
https://langfuse.com/docs/evaluation/experiments/datasets
- Preserves subfolder hierarchy via slash naming (<subfolder>/<dataset_name>).
- Idempotent upserts using explicit item IDs (e.g. 'GT-SCI-001').
- Unpacks JSON metadata into native structured dictionaries.
- Graceful offline fallback and detailed audit progress reporting.
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag_core.config import settings
from rag_core.observability.langfuse import LangfuseTracker

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None

# Default path to the datasets directory containing dev/, test/, and suites/
DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "datasets" / "langfuse"
DEFAULT_SUBFOLDERS = ["dev", "test", "suites"]


def load_dataset_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a Langfuse-formatted 3-column CSV file (input, expected_output, metadata).
    Unpacks JSON metadata string into a native Python dictionary.
    """
    items = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader, start=2):
            raw_input = row.get("input", "")
            raw_output = row.get("expected_output", "")
            raw_meta = row.get("metadata", "")

            # Unpack JSON metadata string if serialized
            meta_dict = {}
            if raw_meta:
                try:
                    meta_dict = json.loads(raw_meta)
                except Exception:
                    meta_dict = {"raw_metadata": raw_meta}

            # Extract persistent item ID if available (enables idempotent upserts)
            item_id = meta_dict.get("id")

            items.append({
                "id": item_id,
                "input": raw_input,
                "expected_output": raw_output,
                "metadata": meta_dict
            })
    return items


def migrate_dataset_file(
    client: Any,
    csv_path: Path,
    dataset_name: str,
    subfolder_name: str,
    dry_run: bool = False
) -> int:
    """
    Upload or idempotent-upsert a single CSV file into a Langfuse Cloud Dataset.
    Dataset name follows the virtual folder pattern: '<subfolder>/<filename_stem>'.
    """
    items = load_dataset_csv(csv_path)
    if not items:
        print(f"  [Empty] Skipping {csv_path.name} (0 items found)")
        return 0

    if dry_run:
        print(f"  [Dry-Run] Would migrate {len(items)} items -> '{dataset_name}'")
        return len(items)

    # 1. Create or get dataset with metadata
    description = f"Unified RAG Benchmark: {subfolder_name.upper()} suite ({csv_path.stem})"
    try:
        client.create_dataset(
            name=dataset_name,
            description=description,
            metadata={
                "subfolder": subfolder_name,
                "source_file": csv_path.name,
                "item_count": len(items),
                "migrated_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
            }
        )
    except Exception:
        # Dataset may already exist in project; proceed to upsert items
        pass

    # 2. Check for existing items to prune obsolete ones no longer in benchmark
    try:
        existing_ds = client.get_dataset(dataset_name)
        existing_item_map = {item.id: item for item in (existing_ds.items or [])}
    except Exception:
        existing_item_map = {}

    expected_scoped_ids = {
        f"{dataset_name}:{it['id']}" for it in items if it.get("id")
    }

    deleted_count = 0
    for old_id in list(existing_item_map.keys()):
        if old_id not in expected_scoped_ids:
            try:
                client.api.dataset_items.delete(id=old_id)
                deleted_count += 1
            except Exception as e:
                pass
    if deleted_count > 0:
        print(f"    Cleaned up {deleted_count} obsolete items from '{dataset_name}'")

    # 3. Upload items with explicit ID for idempotency
    synced_count = 0
    for it in items:
        try:
            # Langfuse item IDs are globally unique per project across all datasets.
            # Scope item ID with dataset_name so the same benchmark question can exist
            # across dev, test, and suite datasets while enabling idempotent upserts.
            scoped_id = f"{dataset_name}:{it['id']}" if it.get("id") else None

            client.create_dataset_item(
                dataset_name=dataset_name,
                id=scoped_id,
                input=it["input"],
                expected_output=it["expected_output"],
                metadata=it["metadata"]
            )
            synced_count += 1
        except Exception as e:
            print(f"    Notice: Failed to insert item {it.get('id')}: {e}")

    return synced_count


def migrate_all_subfolders(
    base_dir: Path = DEFAULT_BASE_DIR,
    subfolders: Optional[List[str]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Iterate over the specified subfolders ('dev', 'test', 'suites') inside the base directory,
    migrating each CSV dataset into Langfuse Cloud under virtual folder paths.
    """
    target_subfolders = subfolders or DEFAULT_SUBFOLDERS
    base_dir = Path(base_dir).resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Base dataset directory does not exist: {base_dir}")

    print("=" * 72)
    print("Langfuse Evaluation Dataset Cloud Migration")
    print(f"Source Directory: {base_dir}")
    print(f"Target Subfolders: {target_subfolders}")
    print(f"Target Host: {settings.langfuse_host}")
    print(f"Dry-Run Mode: {dry_run}")
    print("=" * 72)

    # Initialize Langfuse client
    if not dry_run:
        if not settings.langfuse_enabled or Langfuse is None:
            print("ERROR: Langfuse is not enabled or SDK is not installed.")
            sys.exit(1)

        tracker = LangfuseTracker()
        if not tracker.is_active:
            print("ERROR: LangfuseTracker could not connect to Langfuse.")
            sys.exit(1)

        client = tracker.client
        auth_ok = client.auth_check()
        if not auth_ok:
            print("ERROR: Langfuse authentication check failed. Verify API credentials.")
            sys.exit(1)
        print("✓ Langfuse Cloud Authentication Successful!\n")
    else:
        client = None

    stats: Dict[str, Any] = {
        "total_datasets": 0,
        "total_items": 0,
        "details": {}
    }

    start_time = time.perf_counter()

    for folder_name in target_subfolders:
        folder_path = base_dir / folder_name
        if not folder_path.is_dir():
            print(f"Notice: Subfolder '{folder_name}' not found at {folder_path}. Skipping.")
            continue

        csv_files = sorted(folder_path.glob("*.csv"))
        if not csv_files:
            print(f"Notice: No CSV files found in subfolder '{folder_name}'.")
            continue

        print(f"📁 Processing folder '{folder_name}' ({len(csv_files)} datasets)...")

        for csv_file in csv_files:
            dataset_name = f"{folder_name}/{csv_file.stem}"
            print(f"  -> Migrating '{csv_file.name}' into Langfuse Dataset '{dataset_name}'...")
            
            count = migrate_dataset_file(
                client=client,
                csv_path=csv_file,
                dataset_name=dataset_name,
                subfolder_name=folder_name,
                dry_run=dry_run
            )

            stats["total_datasets"] += 1
            stats["total_items"] += count
            stats["details"][dataset_name] = count
            print(f"     ✓ Synchronized {count} items.")

        print()

    # Flush all queued events
    if client and not dry_run:
        print("Flushing telemetry batches to Langfuse Cloud...")
        client.flush()

    elapsed = time.perf_counter() - start_time
    print("=" * 72)
    print(f"Migration Complete in {elapsed:.2f}s!")
    print(f"Total Datasets: {stats['total_datasets']}")
    print(f"Total Items Synchronized: {stats['total_items']}")
    print("=" * 72)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate local evaluation datasets (dev, test, suites) to Langfuse Cloud."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Path to the directory containing dataset subfolders."
    )
    parser.add_argument(
        "--subfolders",
        nargs="+",
        default=DEFAULT_SUBFOLDERS,
        help="Subfolders to migrate (default: dev test suites)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect datasets and item counts without sending network calls."
    )
    args = parser.parse_args()

    migrate_all_subfolders(
        base_dir=args.dir,
        subfolders=args.subfolders,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

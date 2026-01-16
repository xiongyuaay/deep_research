#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Hugging Face dataset "walledai/StrongREJECT" into current directory.

Default behavior:
- Uses `datasets.load_dataset` to download/cache the dataset, then exports the chosen split
  to a local file under the output directory (default: ./StrongREJECT_local/).

Optional behavior:
- Use --snapshot to mirror the dataset repository files via huggingface_hub.snapshot_download.

Usage examples:
  python download_strongreject.py
  python download_strongreject.py --split train --format jsonl --out .
  python download_strongreject.py --format parquet
  python download_strongreject.py --snapshot --out ./StrongREJECT_repo_mirror
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional


DATASET_ID = "walledai/StrongREJECT"


def export_with_datasets(
    dataset_id: str,
    split: str,
    out_dir: Path,
    fmt: str,
    num_proc: Optional[int] = None,
) -> Path:
    """
    Download dataset via `datasets` and export split to file in out_dir.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: datasets\n"
            "Install with: pip install -U datasets\n"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_id, split=split)  # downloads to HF cache automatically

    # Choose output file name
    safe_split = split.replace("/", "_")
    if fmt == "jsonl":
        out_path = out_dir / f"{dataset_id.split('/')[-1]}_{safe_split}.jsonl"
        # write json lines
        with out_path.open("w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return out_path

    if fmt == "parquet":
        out_path = out_dir / f"{dataset_id.split('/')[-1]}_{safe_split}.parquet"
        ds.to_parquet(str(out_path))
        return out_path

    if fmt == "csv":
        out_path = out_dir / f"{dataset_id.split('/')[-1]}_{safe_split}.csv"
        ds.to_csv(str(out_path), index=False)
        return out_path

    raise ValueError(f"Unsupported format: {fmt}. Choose from: jsonl, parquet, csv.")


def snapshot_repo(dataset_id: str, out_dir: Path) -> Path:
    """
    Mirror the dataset repository files locally (raw files), using huggingface_hub.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install with: pip install -U huggingface_hub\n"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    # This will download repo files to out_dir (no symlinks), useful if you want the raw dataset scripts/files.
    local_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def main():
    parser = argparse.ArgumentParser(description=f"Download HF dataset: {DATASET_ID}")
    parser.add_argument("--split", default="train", help="Dataset split to export (default: train)")
    parser.add_argument(
        "--format",
        default="jsonl",
        choices=["jsonl", "parquet", "csv"],
        help="Export file format (default: jsonl)",
    )
    parser.add_argument(
        "--out",
        default=str(Path.cwd() / "StrongREJECT"),
        help="Output directory (default: ./StrongREJECT_local)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="If set, mirror dataset repository files (raw) instead of exporting a split.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out).resolve()

    # If user wants to download gated/private data, they can set HF_TOKEN in env beforehand.
    # huggingface_hub and datasets will respect it automatically.
    if args.snapshot:
        path = snapshot_repo(DATASET_ID, out_dir)
        print(f"[OK] Dataset repository mirrored to: {path}")
        return

    out_file = export_with_datasets(
        dataset_id=DATASET_ID,
        split=args.split,
        out_dir=out_dir,
        fmt=args.format,
    )
    print(f"[OK] Exported split '{args.split}' to: {out_file}")
    print("[Note] HF cached files are stored in your Hugging Face cache directory by default.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Data sanity-check utility for KrishiRakshak datasets.

Usage (from project root):

    python scripts/sanity_check.py \
        --csv data/processed_data/train.csv \
        --csv data/processed_data/val.csv \
        --csv data/processed_data/test.csv \
        --out reports/data_quality_report.json

The script evaluates:
1. Schema compliance (columns present, correct dtypes)
2. Missing values
3. Duplicate rows / duplicate image paths
4. Basic class-distribution stats
5. Existence of image files referenced by each split
6. Spot-check for corrupted images (sample of up to 200 files per split)
7. Consistency of class labels across splits

Outputs a single JSON report that can be version-controlled.
"""
from __future__ import annotations

import argparse
import json
import imghdr
import pathlib
from collections import Counter
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm

EXPECTED_COLS = ["image_path", "label"]


def analyse_csv(csv_path: pathlib.Path, sample_size: int = 200) -> Dict[str, Any]:
    """Run sanity checks on one CSV split.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the CSV file.
    sample_size : int, optional
        Number of images to sample for corruption check. Defaults to 200.

    Returns
    -------
    Dict[str, Any]
        Dictionary with results for this split.
    """
    df = pd.read_csv(csv_path)

    # Basic properties
    result: Dict[str, Any] = {
        "split": csv_path.stem,
        "path": str(csv_path),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_expected_cols": [col for col in EXPECTED_COLS if col not in df.columns],
    }

    # Missing values
    result["null_values"] = df.isnull().sum().to_dict()

    # Duplicate analysis
    result["duplicate_rows"] = int(df.duplicated().sum())
    if "image_path" in df.columns:
        result["duplicate_image_path"] = int(df["image_path"].duplicated().sum())
    else:
        result["duplicate_image_path"] = None

    # Label stats
    if "label" in df.columns:
        label_series = df["label"].astype(str)
        result["num_classes"] = int(label_series.nunique())
        result["class_distribution"] = label_series.value_counts().to_dict()
    else:
        result["num_classes"] = None
        result["class_distribution"] = {}

    # File existence check
    missing_files: List[str] = []
    if "image_path" in df.columns:
        for img_path in df["image_path"]:
            if not pathlib.Path(img_path).exists():
                missing_files.append(img_path)
    result["missing_files_count"] = len(missing_files)
    if missing_files:
        result["missing_files_sample"] = missing_files[:10]

    # Corrupted image spot-check
    corrupted: List[str] = []
    if "image_path" in df.columns:
        sample_paths = df["image_path"].sample(min(sample_size, len(df)), random_state=42)
        for img_path in tqdm(sample_paths, desc=f"Scanning {csv_path.name} for corruption", unit="img"):
            img_path = pathlib.Path(img_path)
            if img_path.exists():
                try:
                    if imghdr.what(img_path) is None:
                        corrupted.append(str(img_path))
                except Exception:
                    corrupted.append(str(img_path))
    result["corrupted_files_spotcheck"] = corrupted[:5]
    result["corrupted_files_count_spotcheck"] = len(corrupted)

    return result


def compare_label_sets(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare class labels across multiple split reports."""
    label_sets = {
        r["split"]: set(r["class_distribution"].keys()) for r in reports if r["class_distribution"]
    }
    all_labels = set().union(*label_sets.values()) if label_sets else set()
    inconsistencies = {}
    for split, labels in label_sets.items():
        missing = all_labels - labels
        extra = labels - all_labels
        if missing or extra:
            inconsistencies[split] = {
                "missing_labels": sorted(missing),
                "extra_labels": sorted(extra),
            }
    return inconsistencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sanity checks on KrishiRakshak CSV splits.")
    parser.add_argument("--csv", type=pathlib.Path, nargs="+", required=True, help="CSV file paths")
    parser.add_argument("--out", type=pathlib.Path, default="reports/data_quality_report.json", help="Output JSON path")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for csv_file in args.csv:
        reports.append(analyse_csv(csv_file))

    summary = {
        "reports": reports,
        "label_set_inconsistencies": compare_label_sets(reports),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Data quality report saved to {args.out}")


if __name__ == "__main__":
    main()

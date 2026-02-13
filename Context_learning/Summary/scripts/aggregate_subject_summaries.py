#!/usr/bin/env python3

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd

SHOT_MAP = {"1": "zero-shot", "2": "three-shot", "3": "six-shot"}
PROC_MAP = {"1": "regex", "2": "nonpro", "3": "RAG"}


def extract_setting(filename: str) -> str:
    match = re.search(r"([ABC])([123])([123])", filename)
    if not match:
        return "unknown"
    _, shot, proc = match.groups()
    return f"{SHOT_MAP.get(shot, 'unknown')}-{PROC_MAP.get(proc, 'unknown')}"


def extract_model(filename: str) -> str:
    lower = filename.lower()
    if "gemma" in lower:
        return "gemma"
    if "llama2" in lower:
        return "llama2"
    if "llama3" in lower:
        return "llama3"
    if filename.startswith("SUM_A"):
        return "gemma"
    if filename.startswith("SUM_B"):
        return "llama2"
    if filename.startswith("SUM_C"):
        return "llama3"
    return "unknown"


def extract_window(filename: str) -> str:
    match = re.search(r"_(10days|15days|20days|subject)", filename)
    if match:
        window = match.group(1)
        return {"10days": "20days", "15days": "30days", "20days": "40days"}.get(window, window)
    if "_subject" in filename:
        return "subject"
    return "unknown_window"


def aggregate_summaries(base_dir: Path) -> None:
    input_dir = base_dir / "data" / "sum_subject"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for sensitivity in ["system_sensitivity", "model_sensitivity"]:
        results_subdir = input_dir / sensitivity
        rows = []

        for path in glob.glob(str(results_subdir / "SUM_*_subject.csv")):
            filename = os.path.basename(path)
            if not filename.startswith(("SUM_A", "SUM_B", "SUM_C")):
                continue
            df = pd.read_csv(path)
            if df.empty or "N_c" not in df.columns:
                continue
            row = df.iloc[0].to_dict()
            row.update(
                {
                    "filename": filename,
                    "setting": extract_setting(filename),
                    "model": extract_model(filename),
                    "window": extract_window(filename),
                }
            )
            rows.append(row)

        if not rows:
            print(f"No SUM_*_subject.csv files found for {sensitivity}.")
            continue

        consolidated = pd.DataFrame(rows)
        ordered_cols = ["filename", "window", "setting", "model", "P_c", "P_i", "P_l", "N_c", "N_i", "N_u"]
        consolidated = consolidated[[c for c in ordered_cols if c in consolidated.columns] + [c for c in consolidated.columns if c not in ordered_cols]]

        output_path = output_dir / f"consolidated_summary_subject_{sensitivity}.csv"
        consolidated.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate subject-level SUM CSVs into consolidated summaries.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Pipeline base directory (defaults to Context_learning/Subject_summary_pipeline).",
    )
    args = parser.parse_args()
    aggregate_summaries(Path(args.base_dir))

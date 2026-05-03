#!/usr/bin/env python3
"""Convert narratives.jsonl to the CSV format expected by Step 1 of the pipeline.

Samples N documents from the JSONL and writes a CSV with columns:
  doc_id, narrative_text, depression
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


BLEED_MARKERS = [
    "here is your rewritten",
    "here is the rewritten",
    "here is a rewritten",
    "the noise variables",
    "as requested",
    "let me know if you need",
    "no further action is needed",
    "the best answer is",
    "the response has already",
    "for a patient with the profile",
    "this response captures",
    "this narrative captures",
    "pressing distress without",
    "note:",
    "i have rewritten",
    "the provided text",
]


def clean_narrative(text: str) -> str:
    # Strip anything after a ``` marker
    text = text.split("```")[0]

    # Strip anything after known model commentary markers (case-insensitive)
    lower = text.lower()
    cutoff = len(text)
    for marker in BLEED_MARKERS:
        idx = lower.find(marker)
        if idx != -1:
            cutoff = min(cutoff, idx)
    text = text[:cutoff]

    return text.strip()


def is_clean(text: str, min_chars: int = 200) -> bool:
    return len(text) >= min_chars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--narratives-jsonl", required=True, type=Path,
                        help="Path to reliable_narratives.jsonl (output of audit_and_filter_narratives.py)")
    parser.add_argument("--output-csv",       required=True, type=Path)
    parser.add_argument("--n-sample",         type=int, default=250)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    records = []
    with args.narratives_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} narratives")

    # Clean all records first, mark which are usable
    rng = random.Random(args.seed)
    for r in records:
        r["_clean"] = clean_narrative(r["narrative"])
        r["_ok"]    = is_clean(r["_clean"])

    clean_records = [r for r in records if r["_ok"]]
    dirty_count   = len(records) - len(clean_records)
    print(f"Clean: {len(clean_records)}  |  Removed (garbled): {dirty_count}")

    sample = rng.sample(clean_records, k=min(args.n_sample, len(clean_records)))

    n_pos = sum(r["response_value"] for r in sample)
    n_neg = len(sample) - n_pos
    print(f"Sampled {len(sample)}  →  Y=1: {n_pos}  Y=0: {n_neg}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "narrative_text", "depression"])
        writer.writeheader()
        for r in sample:
            writer.writerow({
                "doc_id":         r["doc_id"],
                "narrative_text": r["_clean"],
                "depression":     r["response_value"],
            })

    print(f"Saved → {args.output_csv}")


if __name__ == "__main__":
    main()

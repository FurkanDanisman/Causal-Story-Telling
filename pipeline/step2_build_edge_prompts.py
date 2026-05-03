#!/usr/bin/env python3
"""Step 2: Construct directed edge tasks/prompts for every ordered pair (Ci, Cj)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


PROMPT_TEMPLATE = """Document:
{document}

Candidate variables:
{candidate_variables}

Target directed edge:
{source} -> {target}

Other candidate variables that could lie between {source} and {target}:
{other_candidates}

Task:
Decide whether the document supports a direct causal edge from {source} to {target}.

Definition of direct edge:
A direct edge {source} -> {target} is supported only if the document suggests that {source} affects, changes, produces, worsens, improves, triggers, or contributes to {target} without requiring another listed candidate variable as an intermediate step.

Do not answer Yes if the document only supports an indirect pathway such as:
{source} -> Ck -> {target}

Do not answer Yes only because {source} happens before {target} or is associated with {target}.

Question:
Does the document support a direct causal edge {source} -> {target}?

Answer with one word only: Yes or No.

Answer:
"""


def read_docs(input_csv: Path, doc_id_col: str, text_col: str) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {doc_id_col, text_col}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Missing columns in input CSV: {missing}")
        for row in reader:
            docs[str(row[doc_id_col])] = str(row[text_col])
    return docs


def read_candidates(candidates_json: Path) -> List[dict]:
    with candidates_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Candidates JSON must be a list of records.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Build edge prompts for ordered variable pairs.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--candidates-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--prompt-template-out", type=Path, default=None)
    parser.add_argument("--doc-id-col", default="doc_id")
    parser.add_argument("--text-col", default="narrative_text")
    args = parser.parse_args()

    doc_text = read_docs(args.input_csv, args.doc_id_col, args.text_col)
    records = read_candidates(args.candidates_json)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for rec in records:
            doc_id = str(rec["doc_id"])
            response_value = int(rec.get("response_value", 0))
            candidates = list(rec["candidates"])
            if doc_id not in doc_text:
                raise ValueError(f"doc_id={doc_id} present in candidates JSON but missing in input CSV.")

            candidate_block = "[" + ", ".join(candidates) + "]"
            document = doc_text[doc_id]

            for source in candidates:
                for target in candidates:
                    if source == target:
                        continue
                    others = [c for c in candidates if c not in {source, target}]
                    other_block = "[" + ", ".join(others) + "]"
                    prompt = PROMPT_TEMPLATE.format(
                        document=document,
                        candidate_variables=candidate_block,
                        source=source,
                        target=target,
                        other_candidates=other_block,
                    )
                    row = {
                        "doc_id": doc_id,
                        "response_value": response_value,
                        "document": document,
                        "candidates": candidates,
                        "source": source,
                        "target": target,
                        "other_candidates": others,
                        "prompt": prompt,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.prompt_template_out is not None:
        args.prompt_template_out.parent.mkdir(parents=True, exist_ok=True)
        with args.prompt_template_out.open("w", encoding="utf-8") as f:
            f.write(PROMPT_TEMPLATE)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Step 1 (KB): Knowledge-based variable extraction guided by the Beck & Bredemeier (2016) taxonomy.

Approach: the taxonomy (clinical definitions + intensity criteria) is furnished to an LLM
as a structured prompt. The LLM applies the theoretical framework to the narrative text.
This is a knowledge-based method — the knowledge (what to look for, how to classify) comes
from the clinical taxonomy; the LLM is the execution engine that handles linguistic variation.

This differs from the free-form step1 extraction, which lets the LLM invent variable names.
Here, the variable set and classification criteria are externally defined and fixed.

Usage:
  python step1_kb_extract.py \\
      --input-csv      out/pipeline_input.csv \\
      --output-json    out/kb_extractions.json \\
      --output-csv     out/kb_extractions.csv \\
      --hf-model       /path/to/Llama-3.3-70B-Instruct

Input CSV columns: doc_id, narrative_text (or --text-col), depression (or --response-col)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Load .env from project root if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            import os as _os; _os.environ.setdefault(_k.strip(), _v.strip())

from taxonomy_kb import TAXONOMY, TAXONOMY_VAR_NAMES, build_taxonomy_block


# ── Prompt construction ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical psychologist applying the Beck & Bredemeier (2016) unified model of \
depression to analyze therapy session narratives. Your task is to extract and quantify \
the presence of theoretical constructs from the text, strictly following the provided \
taxonomy definitions and intensity criteria.

Be conservative: only assign intensity > 0 when there is explicit textual evidence. \
Do not infer what is not expressed. Quote the exact span of text that grounds each \
non-zero assignment.\
"""

EXTRACTION_PROMPT = """\
{system}

==============================================================
CLINICAL TAXONOMY (Beck & Bredemeier, 2016)
==============================================================
{taxonomy_block}

==============================================================
NARRATIVE TO ANALYZE
==============================================================
{narrative}

==============================================================
TASK
==============================================================
For each of the {n_vars} variables listed below, assign:
  - intensity: integer 0 to 3 (following the level definitions above)
  - evidence: exact quote from the narrative that grounds your rating (empty string if intensity=0)

Variables to assess (in order):
{var_list}

==============================================================
OUTPUT FORMAT
==============================================================
Respond ONLY with a valid JSON array. No preamble, no explanation outside the array.
Each element must follow this exact schema:

[
  {{
    "variable": "<variable_name>",
    "intensity": <0|1|2|3>,
    "evidence": "<exact quote or empty string>"
  }},
  ...
]

The array must contain exactly {n_vars} elements, one per variable, in the order listed.
"""


def build_prompt(narrative: str) -> str:
    taxonomy_block = build_taxonomy_block()
    var_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(TAXONOMY_VAR_NAMES))
    return EXTRACTION_PROMPT.format(
        system=SYSTEM_PROMPT,
        taxonomy_block=taxonomy_block,
        narrative=narrative,
        n_vars=len(TAXONOMY_VAR_NAMES),
        var_list=var_list,
    )


# ── Output parsing ─────────────────────────────────────────────────────────────

def binary_encode(level: int, n_levels: int = 3) -> List[int]:
    enc = [0] * (n_levels + 1)
    enc[max(0, min(level, n_levels))] = 1
    return enc


def parse_llm_output(raw: str) -> Optional[List[dict]]:
    """
    Extract and validate the JSON array from the LLM output.
    Returns None if parsing fails.
    """
    raw = raw.strip()

    # Try direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fall back to extracting first [...] block
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


def validate_and_enrich(raw_items: List[dict]) -> List[dict]:
    """
    Validate each item, clamp intensity to [0,3], add metadata from taxonomy.
    If a variable is missing from LLM output, insert a default level-0 entry.
    """
    by_name: Dict[str, dict] = {}
    for item in raw_items:
        name = str(item.get("variable", "")).strip()
        if name in TAXONOMY_VAR_NAMES:
            by_name[name] = item

    result = []
    for var in TAXONOMY:
        item = by_name.get(var.name, {})
        intensity = int(item.get("intensity", 0))
        intensity = max(0, min(intensity, 3))
        evidence  = str(item.get("evidence", "")).strip()
        if intensity == 0:
            evidence = ""

        result.append({
            "category":       var.category,
            "subcategory":    var.subcategory,
            "variable":       var.name,
            "intensity":      intensity,
            "binary_encoding": binary_encode(intensity),
            "evidence":       evidence,
        })
    return result


# ── HuggingFace backend ────────────────────────────────────────────────────────

class HFGenerator:
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Requires torch + transformers.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        torch = self.torch
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ── I/O ────────────────────────────────────────────────────────────────────────

def parse_binary_col(value: str) -> int:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return 1
    if v in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(v) > 0.5)
    except ValueError:
        return 0


def write_flat_csv(records: List[dict], path: Path) -> None:
    fieldnames = [
        "doc_id", "response_value",
        "category", "subcategory", "variable",
        "intensity", "L0", "L1", "L2", "L3",
        "evidence",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            for var in rec["variables"]:
                enc = var["binary_encoding"]
                w.writerow({
                    "doc_id":         rec["doc_id"],
                    "response_value": rec["response_value"],
                    "category":       var["category"],
                    "subcategory":    var["subcategory"],
                    "variable":       var["variable"],
                    "intensity":      var["intensity"],
                    "L0": enc[0], "L1": enc[1], "L2": enc[2], "L3": enc[3],
                    "evidence":       var["evidence"],
                })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1 (KB): LLM-guided taxonomy extraction (Beck & Bredemeier 2016)."
    )
    parser.add_argument("--input-csv",      required=True,  type=Path)
    parser.add_argument("--output-json",    required=True,  type=Path)
    parser.add_argument("--output-csv",     default=None,   type=Path)
    parser.add_argument("--hf-model",       required=True)
    parser.add_argument("--doc-id-col",     default="doc_id")
    parser.add_argument("--text-col",       default="narrative_text")
    parser.add_argument("--response-col",   default="depression")
    parser.add_argument("--max-new-tokens", type=int,   default=1200)
    parser.add_argument("--temperature",    type=float, default=0.0)
    args = parser.parse_args()

    generator = HFGenerator(model_name=args.hf_model)

    records: List[dict] = []

    with args.input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id         = str(row[args.doc_id_col])
            text           = str(row[args.text_col])
            response_value = parse_binary_col(row.get(args.response_col, "0"))

            prompt  = build_prompt(text)
            raw_out = generator.generate(prompt, args.max_new_tokens, args.temperature)

            raw_items = parse_llm_output(raw_out)
            if raw_items is None:
                print(f"[{doc_id}] WARNING: could not parse LLM output — defaulting all to 0")
                variables = validate_and_enrich([])
            else:
                variables = validate_and_enrich(raw_items)

            detected = sum(1 for v in variables if v["intensity"] > 0)
            print(f"[{doc_id}]  Y={response_value}  {detected}/{len(variables)} variables detected")

            records.append({
                "doc_id":         doc_id,
                "response_value": response_value,
                "variables":      variables,
                "raw_llm_output": raw_out,
            })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved -> {args.output_json}")

    if args.output_csv:
        write_flat_csv(records, args.output_csv)
        print(f"CSV  saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
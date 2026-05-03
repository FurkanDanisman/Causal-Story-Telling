#!/usr/bin/env python3
"""Step 1b: Normalize candidate variable names across all documents.

Collects every unique variable name produced by Step 1, asks the LLM to
group synonyms and assign one canonical snake_case name per concept, then
rewrites the candidates JSON with those canonical names.

This corrects for the case where different documents surface the same
underlying concept under slightly different phrasings (e.g. "work_stress",
"stress_at_work", "job_related_stress" all become "work_stress"), so that
Step 4's union variable set C^all does not fragment one concept into many
spuriously distinct variables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


NORMALIZATION_PROMPT_TEMPLATE = """You are normalizing candidate variables extracted from multiple narrative documents for causal analysis.

Below is the complete list of unique candidate variables extracted across all documents:
{variable_list}

Task:
Identify variables that refer to the same underlying concept and assign each group one canonical snake_case name.
Variables with minor phrasing differences (e.g. "work_stress" and "stress_at_work") should share one canonical name.
Variables that are genuinely distinct should keep separate canonical names.
Every original variable name must appear as a key in your output.

Output only a JSON object where each key is an original variable name and each value is its canonical snake_case name.

Example:
{{"work_stress": "work_stress", "stress_at_work": "work_stress", "sleep_disturbance": "sleep_disturbance", "disrupted_sleep": "sleep_disturbance"}}

Answer:
"""


def to_snake_case(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t)
    return t.strip("_")


def extract_json_object(raw: str) -> Dict[str, str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


class HFGenerator:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("step1b requires torch + transformers.") from exc
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
        toks = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][toks.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def build_normalization_map(
    all_vars: List[str],
    generator: HFGenerator,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, str]:
    """
    Call the LLM in batches and merge the returned mappings.
    Falls back to identity (variable maps to itself) for any variable
    the model fails to include in its response.
    """
    mapping: Dict[str, str] = {}

    for start in range(0, len(all_vars), batch_size):
        batch = all_vars[start : start + batch_size]
        var_list_str = "\n".join(f"- {v}" for v in batch)
        prompt = NORMALIZATION_PROMPT_TEMPLATE.format(variable_list=var_list_str)
        raw = generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        batch_map = extract_json_object(raw)

        for var in batch:
            canonical = batch_map.get(var)
            if canonical and isinstance(canonical, str) and canonical.strip():
                mapping[var] = to_snake_case(canonical)
            else:
                # LLM missed this variable — identity fallback
                mapping[var] = var

    return mapping


def apply_mapping(records: List[dict], mapping: Dict[str, str]) -> List[dict]:
    """Rewrite every document's candidate list using the canonical names."""
    out = []
    for rec in records:
        new_candidates = sorted(set(mapping.get(v, v) for v in rec["candidates"]))
        out.append({**rec, "candidates": new_candidates, "normalization_map": mapping})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1b: Normalize candidate variable names across documents."
    )
    parser.add_argument("--candidates-json", required=True, type=Path,
                        help="Output JSON from Step 1.")
    parser.add_argument("--output-json", required=True, type=Path,
                        help="Normalized candidates JSON (same format as Step 1 output).")
    parser.add_argument("--normalization-map-json", type=Path, default=None,
                        help="Optional path to save the raw variable->canonical mapping.")
    parser.add_argument("--hf-model", required=True,
                        help="HuggingFace model name or path for normalization.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=60,
                        help="Max variables per LLM call (to stay within context).")
    args = parser.parse_args()

    with args.candidates_json.open("r", encoding="utf-8") as f:
        records: List[dict] = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Candidates JSON must be a list of records.")

    # Collect all unique variable names across all documents
    all_vars: List[str] = sorted({v for rec in records for v in rec["candidates"]})
    if not all_vars:
        raise ValueError("No candidate variables found in input JSON.")

    generator = HFGenerator(model_name=args.hf_model, device=args.device)

    mapping = build_normalization_map(
        all_vars=all_vars,
        generator=generator,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    normalized_records = apply_mapping(records, mapping)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(normalized_records, f, indent=2)

    if args.normalization_map_json is not None:
        args.normalization_map_json.parent.mkdir(parents=True, exist_ok=True)
        with args.normalization_map_json.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)


if __name__ == "__main__":
    main()

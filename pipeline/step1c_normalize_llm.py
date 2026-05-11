#!/usr/bin/env python3
"""Step 1c (LLM variant): Normalize candidate names using an LLM.

Collects all unique candidate names across documents, asks an LLM to group
synonyms under canonical labels, and remaps every document's candidate list.
Names not included in any group are assumed to map to themselves.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


NORMALIZATION_PROMPT = """You are normalizing candidate variable names extracted from therapy session narratives for causal analysis.

Below is a list of {n_names} unique variable names. Group names that refer to the same underlying concept under a single canonical snake_case label.

Rules:
- Only merge names that are genuine synonyms or minor surface variations of the same concept.
- Do NOT merge opposites (e.g. social_withdrawal vs social_connection are distinct states).
- Do NOT merge concepts that are related but distinct (e.g. stress and rumination are different).
- The canonical name should be the clearest, most general label for the group.
- Output ONLY groups with 2 or more members — omit names that have no synonyms.
- Output one group per line in this exact format (no JSON, no extra text):
  canonical_name: member1, member2, member3

Variable names:
{names}

Output:"""


def parse_line_format(raw: str) -> dict:
    """Parse compact line format: 'canonical_name: member1, member2, member3'"""
    groups = {}
    for line in raw.splitlines():
        line = line.strip().strip("-").strip()
        if ":" not in line:
            continue
        canonical, _, members_str = line.partition(":")
        canonical = canonical.strip()
        members = [m.strip() for m in members_str.split(",") if m.strip()]
        if canonical and len(members) >= 1:
            groups[canonical] = members
    return groups


class HFGenerator:
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("Requires torch + transformers.") from exc
        self.torch = torch
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.device = self.model.device
        print("Model ready.\n")

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        torch = self.torch
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1c (LLM): Normalize candidate names.")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--cluster-map-json", required=True, type=Path)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    name_counter: Counter = Counter()
    for rec in records:
        name_counter.update(rec["candidates"])

    all_names = sorted(name_counter.keys())
    print(f"Unique candidate names: {len(all_names)}")

    names_str = "\n".join(f"- {n}" for n in all_names)
    prompt = NORMALIZATION_PROMPT.format(n_names=len(all_names), names=names_str)

    generator = HFGenerator(model_name=args.hf_model)
    print("Sending prompt to LLM...")
    raw = generator.generate(prompt=prompt, max_new_tokens=args.max_new_tokens)
    print(f"Raw output length: {len(raw)} chars")
    print(f"Raw output (first 500 chars):\n{raw[:500]}\n")

    # Save raw output for debugging
    raw_path = args.output_json.parent / "step1c_raw_output.txt"
    raw_path.write_text(raw, encoding="utf-8")
    print(f"Full raw output saved → {raw_path}\n")

    groups = parse_line_format(raw)
    canonical_map: dict[str, str] = {}

    if groups:
        for canonical, members in groups.items():
            for m in members:
                canonical_map[str(m).strip()] = str(canonical).strip()
        print(f"Parsed {len(groups)} groups covering {len(canonical_map)} names")
    else:
        print("WARNING: Could not parse LLM output. Falling back to identity mapping.")

    # Names not in any group map to themselves
    unmapped = 0
    for n in all_names:
        if n not in canonical_map:
            canonical_map[n] = n
            unmapped += 1
    if unmapped:
        print(f"{unmapped} names had no synonyms → identity mapped")

    n_canonical = len(set(canonical_map.values()))
    print(f"\nFinal: {len(all_names)} names → {n_canonical} canonical concepts")

    # Remap documents
    out_records = []
    for rec in records:
        normalized = sorted(set(canonical_map.get(c, c) for c in rec["candidates"]))
        out_rec = {k: v for k, v in rec.items()}
        out_rec["candidates_collapsed"] = rec["candidates"]
        out_rec["candidates"] = normalized
        out_records.append(out_rec)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    args.cluster_map_json.parent.mkdir(parents=True, exist_ok=True)
    with args.cluster_map_json.open("w", encoding="utf-8") as f:
        json.dump(canonical_map, f, indent=2, sort_keys=True)

    print(f"Saved → {args.output_json}")
    print(f"Cluster map → {args.cluster_map_json}")


if __name__ == "__main__":
    main()

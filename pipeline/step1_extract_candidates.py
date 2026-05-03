#!/usr/bin/env python3
"""Step 1: Extract candidate variables using an open-source model prompt."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List


EXTRACTION_PROMPT_TEMPLATE = """You are extracting candidate variables from a narrative for causal analysis.

Definition:
A candidate variable is any text-grounded concept that describes a state, event, behavior, condition, attribute, or action that could vary across units, people, or contexts.

Rule:
Extract a text-grounded concept c as a candidate variable if there exists at least one plausible comparison case in which c is absent, different in kind, or different in intensity.

Task:
From the document below, return all candidate variables that satisfy the definition and rule.
Use short snake_case names.
Only include variables grounded in the document.
Output only a JSON array of strings.

Document:
{document}

Answer:
"""


def parse_binary(value: str) -> int:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return 1
    if v in {"0", "false", "no", "n"}:
        return 0
    return int(float(v) > 0.5)


def to_snake_case(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t)
    return t.strip("_")


def extract_json_array(raw: str) -> List[str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [to_snake_case(str(x)) for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [to_snake_case(str(x)) for x in data if str(x).strip()]
        except json.JSONDecodeError:
            return []
    return []


class HFGenerator:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("step1 requires torch + transformers.") from exc
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Prompt-based candidate extraction.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--doc-id-col", default="doc_id")
    parser.add_argument("--text-col", default="narrative_text")
    parser.add_argument("--response-col", default="depression")
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    generator = HFGenerator(model_name=args.hf_model, device=args.device)

    records = []
    with args.input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {args.doc_id_col, args.text_col, args.response_col}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Missing columns in input CSV: {missing}")

        for row in reader:
            doc_id = str(row[args.doc_id_col])
            text = str(row[args.text_col])
            response_value = parse_binary(str(row[args.response_col]))

            prompt = EXTRACTION_PROMPT_TEMPLATE.format(document=text)
            raw_output = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            candidates = sorted(set(extract_json_array(raw_output)))

            records.append(
                {
                    "doc_id": doc_id,
                    "response_value": response_value,
                    "candidates": candidates,
                    "prompt": prompt,
                    "raw_model_output": raw_output,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()

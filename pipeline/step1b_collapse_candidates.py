#!/usr/bin/env python3
"""Step 1b: Collapse granular extracted candidates into construct-level variables.

For each document, takes the original text and the granular candidate list from
Step 1, then asks an LLM to merge sub-behaviors and surface expressions into
the single underlying construct they represent.

Input:  JSON from step1 (list of {doc_id, candidates, response_value, ...})
Output: same structure with 'candidates' replaced by collapsed list,
        original granular list preserved as 'candidates_raw'
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List


COLLAPSE_PROMPT_TEMPLATE = """You are collapsing a list of candidate variables into construct-level concepts \
for causal analysis.

Task:
Some candidates are specific surface behaviors or symptoms that are all expressions of the \
same underlying state or construct. Group those together and replace them with a single \
representative concept name. Keep candidates that are already at the right level of \
abstraction as-is.

Rules:
- Merge candidates only when they clearly refer to the same underlying state.
- The merged concept name should be the most abstract, general label that covers the group.
- Do not merge candidates that represent genuinely distinct states, even if they co-occur.
- Output only a JSON array of unique snake_case concept names, nothing else.

Example:
Candidates: ["snapping_over_small_things", "feeling_shaky", "feeling_overwhelmed", "emotion_dysregulation", "social_withdrawal"]

"snapping_over_small_things", "feeling_shaky", "feeling_overwhelmed" are all surface \
expressions of "emotion_dysregulation". Collapse them into it.
"social_withdrawal" is a distinct construct — keep it.

Output: ["emotion_dysregulation", "social_withdrawal"]

Now collapse the following:

Candidates:
{candidates}

Output:
"""


def extract_json_array(raw: str) -> List[str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            return []
    return []


class HFGenerator:
    def __init__(self, model_name: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1b: Collapse candidates to construct level.")
    parser.add_argument("--input-json", required=True, type=Path,
                        help="JSON output from step1.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    generator = HFGenerator(model_name=args.hf_model)

    out_records = []
    for rec in records:
        doc_id = rec["doc_id"]
        candidates_raw = rec["candidates"]

        candidates_str = json.dumps(candidates_raw, ensure_ascii=False)
        prompt = COLLAPSE_PROMPT_TEMPLATE.format(candidates=candidates_str)

        raw_output = generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        collapsed = extract_json_array(raw_output)

        out_rec = {k: v for k, v in rec.items()}
        out_rec["candidates_raw"] = candidates_raw
        out_rec["candidates"] = collapsed if collapsed else candidates_raw
        out_rec["collapse_prompt"] = prompt
        out_rec["collapse_raw_output"] = raw_output

        out_records.append(out_rec)
        print(f"[{doc_id}]  {len(candidates_raw)} → {len(out_rec['candidates'])} candidates")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    print(f"\nDone. Saved → {args.output_json}")


if __name__ == "__main__":
    main()

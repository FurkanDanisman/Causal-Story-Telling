#!/usr/bin/env python3
"""Step 3: Compute P_ij(D) via MC Yes/No logits from an open-source model.

Supports checkpoint/resume: already-scored (doc_id, source, target) triples
are skipped, so the job can be resubmitted after hitting a time limit and
will continue from where it left off.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Set, Tuple


PROMPT_VARIANTS = [
    "",
    "Keep the decision conservative and avoid over-claiming causality.\n\n",
    "Use only evidence grounded in the document text, not background assumptions.\n\n",
]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(x) for x in parts)
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) % 2_147_483_647
    return h


def softmax_binary(yes_score: float, no_score: float) -> float:
    m = max(yes_score, no_score)
    e_yes = math.exp(yes_score - m)
    e_no  = math.exp(no_score - m)
    return e_yes / (e_yes + e_no)


class HFYesNoScorer:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Backend 'hf_logits' requires torch + transformers."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def yes_prob(self, prompt: str) -> float:
        """Single forward pass — read Yes/No logits at the final position."""
        torch = self.torch
        formatted = self._format_prompt(prompt)
        tok = self.tokenizer

        yes_id = tok(" Yes", add_special_tokens=False).input_ids[-1]
        no_id  = tok(" No",  add_special_tokens=False).input_ids[-1]

        input_ids = tok(formatted, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        last_logits = logits[0, -1, :]
        return softmax_binary(float(last_logits[yes_id]), float(last_logits[no_id]))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done(output_csv: Path) -> Set[Tuple[str, str, str]]:
    """Return set of (doc_id, source, target) already written to the output CSV."""
    done: Set[Tuple[str, str, str]] = set()
    if not output_csv.exists():
        return done
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["doc_id"], row["source"], row["target"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Score directed edges with MC Yes/No logits.")
    parser.add_argument("--edge-prompts-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv",         required=True, type=Path)
    parser.add_argument("--mc-samples",  type=int,   default=4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--hf-model",    required=True)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    if args.mc_samples <= 0:
        raise ValueError("--mc-samples must be >= 1")

    # ── Checkpoint: skip already-scored edges ────────────────────────────────
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output_csv)
    print(f"Already scored: {len(done)} edges — skipping these.")

    # Count remaining work
    total = sum(1 for _ in read_jsonl(args.edge_prompts_jsonl))
    remaining = total - len(done)
    print(f"Total edges: {total}  |  Remaining: {remaining}")

    if remaining == 0:
        print("All edges already scored. Nothing to do.")
        return

    # ── Load model only if there is work to do ───────────────────────────────
    hf_scorer = HFYesNoScorer(model_name=args.hf_model, device=args.device)

    # Append mode so partial runs accumulate safely
    write_header = not args.output_csv.exists() or args.output_csv.stat().st_size == 0
    scored = 0
    with args.output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "source", "target", "p_edge"])
        if write_header:
            writer.writeheader()

        for task in read_jsonl(args.edge_prompts_jsonl):
            key = (task["doc_id"], task["source"], task["target"])
            if key in done:
                continue

            samples: List[float] = []
            for b in range(args.mc_samples):
                s = stable_seed(args.seed, task["doc_id"], task["source"], task["target"], b)
                rng = random.Random(s)
                prefix = PROMPT_VARIANTS[rng.randrange(len(PROMPT_VARIANTS))]
                p_yes = hf_scorer.yes_prob(prefix + task["prompt"])
                samples.append(max(0.0, min(1.0, p_yes)))

            p_edge = sum(samples) / len(samples)
            writer.writerow({
                "doc_id": task["doc_id"],
                "source": task["source"],
                "target": task["target"],
                "p_edge": f"{p_edge:.6f}",
            })
            f.flush()  # ensure each row is on disk before the job is killed
            done.add(key)
            scored += 1

            if scored % 100 == 0:
                print(f"  Scored {scored}/{remaining} remaining edges ({len(done)}/{total} total)")

    print(f"Done. Scored {scored} new edges. Total in file: {len(done)}/{total}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Synthetic data generation: Stage 3+4 — Prompt construction and narrative generation.

Reads the Stage 1+2 samples JSON, fills the prompt template for each document,
calls the LLM to generate a therapy session transcript, and writes results to JSONL.

Designed for cluster use:
  --start-idx / --end-idx   process a slice of the samples (for SLURM array jobs)
  --checkpoint-every        flush to disk every N documents so partial runs are saved
  --output-jsonl            append-safe line-by-line output

Example (single job):
  python3 synth_stage3_generate_narratives.py \
      --samples-json out/samples.json \
      --output-jsonl out/narratives.jsonl \
      --hf-model /path/to/model \
      --device cuda

Example (SLURM array, 10 jobs × 100 docs each):
  python3 synth_stage3_generate_narratives.py \
      --samples-json out/samples.json \
      --output-jsonl out/narratives_${SLURM_ARRAY_TASK_ID}.jsonl \
      --start-idx $((SLURM_ARRAY_TASK_ID * 100)) \
      --end-idx $(((SLURM_ARRAY_TASK_ID + 1) * 100)) \
      --hf-model /path/to/model \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


# ── Prompt construction ────────────────────────────────────────────────────────

VAR_DESCRIPTIONS = {
    "early_adversity":       "early_adversity (difficult or traumatic childhood experiences)",
    "chronic_stress":        "chronic_stress (persistent stress from work, finances, or relationships)",
    "emotion_dysregulation": "emotion_dysregulation (difficulty controlling or regulating emotions)",
    "social_withdrawal":     "social_withdrawal (pulling away from friends and social life)",
    "rumination":            "rumination (repetitively dwelling on the same negative thoughts)",
}

PROMPT_TEMPLATE = """\
Write a first-person therapy session monologue for the following patient. \
Output ONLY the monologue between <transcript> and </transcript> tags — nothing else.

Patient profile:
- Active experiences: {active_dag_variables}
- Depressed: {depression_label}
{noise_line}
Requirements:
- 150-200 words, patient speaking directly to their therapist
- Every active experience must be clearly named or unmistakably described in the narrative \
(e.g. "rumination" → the patient says "I keep ruminating" or "I can't stop ruminating")
- If Depressed=YES: patient must explicitly say they feel depressed or have been feeling depressed
- If Depressed=NO: patient must not mention feeling depressed — stress and difficulties are fine
- Noise variables appear briefly and feel unrelated to the patient's core struggles
- Do not state explicit causal links ("X caused Y" or "because of X, I feel Y")
- Output nothing outside the tags

<transcript>"""


def build_prompt(record: dict) -> str:
    active = record["active_dag_variables"]
    noise  = record["noise_variables"]
    y      = record["response_value"]

    active_desc = ", ".join(
        VAR_DESCRIPTIONS.get(v, v) for v in active
    ) if active else "none"

    noise_line = (
        f"- Also mentions: {', '.join(noise)}\n" if noise else ""
    )

    return PROMPT_TEMPLATE.format(
        active_dag_variables=active_desc,
        depression_label="YES" if y == 1 else "NO",
        noise_line=noise_line,
    )


# ── Transcript extraction ─────────────────────────────────────────────────────

def _extract_transcript(raw: str) -> str:
    """Pull content between <transcript> and </transcript>. If closing tag is
    missing (model stopped early), return everything after the opening tag."""
    start = raw.find("<transcript>")
    if start != -1:
        raw = raw[start + len("<transcript>"):]
    end = raw.find("</transcript>")
    if end != -1:
        raw = raw[:end]
    return raw.strip()


# ── HuggingFace generator ──────────────────────────────────────────────────────

class HFGenerator:
    def __init__(self, model_name: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("Requires torch + transformers.") from exc

        self.torch = torch
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Left-pad so batched generation works correctly
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {model_name}  →  device_map=auto (multi-GPU)")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        # With device_map=auto the model spans GPUs; inputs go to the first device
        self.device = self.model.device
        print("Model ready.\n")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        torch = self.torch
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return _extract_transcript(raw)


# ── Checkpointing ──────────────────────────────────────────────────────────────

def already_done(output_jsonl: Path) -> set:
    """Return set of doc_ids already written to the output file."""
    done = set()
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["doc_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3+4: Generate therapy narratives from SCM samples."
    )
    parser.add_argument("--samples-json", required=True, type=Path,
                        help="Output JSON from Stage 1+2.")
    parser.add_argument("--output-jsonl", required=True, type=Path,
                        help="Output JSONL — one narrative record per line.")
    parser.add_argument("--hf-model", required=True,
                        help="HuggingFace model name or local path.")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda, cpu, or cuda:N.")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First record index to process (for cluster array jobs).")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last record index (exclusive). Default: all remaining.")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Flush output to disk every N documents.")
    args = parser.parse_args()

    # Load samples
    with args.samples_json.open("r", encoding="utf-8") as f:
        all_records: List[dict] = json.load(f)

    end = args.end_idx if args.end_idx is not None else len(all_records)
    records = all_records[args.start_idx:end]
    print(f"Processing records {args.start_idx} to {end}  ({len(records)} documents)")

    # Skip already-generated documents (safe resume after cluster preemption)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.output_jsonl)
    if done:
        print(f"Resuming — {len(done)} documents already done, skipping.")

    pending = [r for r in records if r["doc_id"] not in done]
    print(f"Remaining: {len(pending)} documents\n")

    if not pending:
        print("Nothing to do.")
        return

    generator = HFGenerator(model_name=args.hf_model, device=args.device)

    buffer = []
    with args.output_jsonl.open("a", encoding="utf-8") as out_f:
        for i, record in enumerate(pending):
            prompt   = build_prompt(record)
            narrative = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            out_record = {
                "doc_id":               record["doc_id"],
                "response_value":       record["response_value"],
                "active_dag_variables": record["active_dag_variables"],
                "noise_variables":      record["noise_variables"],
                "binary_vector":        record["binary_vector"],
                "prompt":               prompt,
                "narrative":            narrative,
            }
            buffer.append(out_record)

            # Checkpoint
            if len(buffer) >= args.checkpoint_every:
                for r in buffer:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()
                buffer = []

            done_count = i + 1 + len(done)
            total      = len(records) + len(done)
            print(f"[{done_count}/{total}]  {record['doc_id']}  Y={record['response_value']}")

        # Flush remainder
        for r in buffer:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved → {args.output_jsonl}")


if __name__ == "__main__":
    main()

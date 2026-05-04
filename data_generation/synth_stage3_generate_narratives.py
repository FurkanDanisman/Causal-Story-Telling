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

# All directed edges in the ground truth DAG
GROUND_TRUTH_EDGES = [
    ("early_adversity",       "chronic_stress"),
    ("early_adversity",       "emotion_dysregulation"),
    ("early_adversity",       "depression"),
    ("chronic_stress",        "emotion_dysregulation"),
    ("chronic_stress",        "social_withdrawal"),
    ("emotion_dysregulation", "rumination"),
    ("social_withdrawal",     "rumination"),
    ("rumination",            "depression"),
]

PROMPT_TEMPLATE = """\
You are writing synthetic therapy session transcripts for a causal inference research study.

TASK
Write a first-person monologue (150-200 words) as a patient speaking to their therapist.

PATIENT PROFILE
- Core experiences: {active_dag_variables}
{noise_line}- Depressed: {depression_label}

CAUSAL STRUCTURE TO ENCODE
The narrative must imply the following causal relationships through the flow of the story:
{active_edges}
The patient should speak in a way that naturally suggests these directions — for example \
using phrases like "when X happens, I find myself Y-ing", "X tends to make me", \
"ever since X, I notice Y", "X leaves me with Y". The causal direction must be \
recoverable from the text, but must not be stated as an explicit fact.

WHAT THE NARRATIVE MUST CONTAIN
- Every core experience must appear by its exact name or an unmistakable close variant:
    rumination            → "I keep ruminating" / "the rumination" / "I can't stop ruminating"
    chronic_stress        → "the chronic stress" / "chronic stress at work"
    social_withdrawal     → "I've been withdrawing socially" / "social withdrawal"
    emotion_dysregulation → "I can't regulate my emotions" / "emotion dysregulation"
    early_adversity       → "early adversity" / "childhood adversity" / "adversity growing up"
- If Depressed=YES: patient must explicitly say "I feel depressed" / "I've been depressed" / "I am depressed".
- If Depressed=NO: patient must not say they feel depressed or persistently low.
- Noise variables appear in one sentence only and feel unrelated to the patient's core struggles.

DON'T
- Don't replace core experience names with vague metaphors only — use the actual terms.
- Don't say "X caused Y" or "because of X" — imply direction through narrative flow, not explicit statements.
- Don't mention depression in any form if Depressed=NO.
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't describe noise variables as emotionally significant or causally connected to anything.

EXAMPLE
Profile: chronic_stress → social_withdrawal → rumination → depression | Depressed=YES | Noise: travel
<transcript>
Work has been relentless — the chronic stress just does not let up, the deadlines, the \
financial pressure, all of it piling on. When the stress gets this bad I find myself \
withdrawing socially, cancelling plans, not picking up calls, just going quiet. I know I \
should reach out but I genuinely can't make myself do it, and being alone just feeds the \
rumination. I keep ruminating on every conversation, every mistake, the same loops over \
and over, and I cannot shut it off. The rumination leaves me feeling completely hollowed \
out. I feel depressed. Not just tired — actually depressed, and it has been like this for \
weeks. I did book a trip last month which was a nice distraction, but it hasn't changed anything.
</transcript>

Now write the transcript for this patient. Output ONLY between the tags.

<transcript>"""


def _active_edges(active_vars: list, y: int) -> str:
    """Return the subset of ground truth edges where both endpoints are active."""
    active_set = set(active_vars)
    if y == 1:
        active_set.add("depression")
    lines = []
    for src, tgt in GROUND_TRUTH_EDGES:
        if src in active_set and tgt in active_set:
            lines.append(f"  {src} → {tgt}")
    return "\n".join(lines) if lines else "  (none — variables are independent)"


def build_prompt(record: dict) -> str:
    active = record["active_dag_variables"]
    noise  = record["noise_variables"]
    y      = record["response_value"]

    active_desc = ", ".join(
        VAR_DESCRIPTIONS.get(v, v) for v in active
    ) if active else "none"

    noise_line = (
        f"- Noise (mention briefly, unrelated to struggles): {', '.join(noise)}\n" if noise else ""
    )

    return PROMPT_TEMPLATE.format(
        active_dag_variables=active_desc,
        depression_label="YES" if y == 1 else "NO",
        noise_line=noise_line,
        active_edges=_active_edges(active, y),
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

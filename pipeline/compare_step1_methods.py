#!/usr/bin/env python3
"""
Compare step1 (LLM free-form) vs step1_kb (LLM-guided taxonomy) on 3 mappable variables.

Ground truth  : narratives_gemma.csv  — binary columns per DAG variable
LLM method    : step1c_normalized.json — normalized candidate lists (keyword mapping)
KB method     : taxonomy_kb.py prompt furnished to an LLM (Anthropic or HuggingFace)

Evaluated variables (clean mapping to ground truth DAG):
  early_adversity   <- LLM: keyword match | KB: early_life_trauma   (intensity > 0)
  social_withdrawal <- LLM: keyword match | KB: social_withdrawal    (intensity > 0)
  rumination        <- LLM: keyword match | KB: rumination           (intensity > 0)

Usage (Anthropic backend, sample of 100 docs):
  python compare_step1_methods.py \\
      --narratives-csv data_generation/out/narratives_gemma.csv \\
      --step1c-json    data_generation/out/step1c_normalized.json \\
      --backend        anthropic \\
      --model          claude-haiku-4-5-20251001 \\
      --n-docs         100 \\
      --output-csv     data_generation/out/step1_comparison_v2.csv

Usage (HuggingFace backend, full dataset):
  python compare_step1_methods.py \\
      --narratives-csv data_generation/out/narratives_gemma.csv \\
      --step1c-json    data_generation/out/step1c_normalized.json \\
      --backend        hf \\
      --model          /path/to/Llama-3.3-70B-Instruct \\
      --output-csv     data_generation/out/step1_comparison_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

# Load .env from project root if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            import os as _os; _os.environ.setdefault(_k.strip(), _v.strip())

from taxonomy_kb import TAXONOMY_VAR_NAMES
from step1_kb_extract import build_prompt, parse_llm_output, validate_and_enrich


# ── Keyword sets for LLM step1c candidate mapping ─────────────────────────────
# Conservative anchors calibrated on the step1c vocabulary.

LLM_KEYWORDS: Dict[str, List[str]] = {
    "early_adversity":   ["adverse", "childhood", "trauma", "abuse", "neglect",
                          "early_life", "upbring"],
    "social_withdrawal": ["isolat", "withdraw"],
    "rumination":        ["rumin"],
}

# KB taxonomy variable -> ground truth DAG variable
KB_VAR_MAP: Dict[str, str] = {
    "early_life_trauma":  "early_adversity",
    "social_withdrawal":  "social_withdrawal",
    "rumination":         "rumination",
}

EVAL_VARS = ["early_adversity", "social_withdrawal", "rumination"]


# ── Step1c LLM prediction (keyword mapping) ────────────────────────────────────

def llm_predict(candidates: List[str], var: str) -> int:
    keywords = LLM_KEYWORDS[var]
    return int(any(kw in cand for cand in candidates for kw in keywords))


# ── KB prediction backends ─────────────────────────────────────────────────────

def _parse_kb_output(raw: str) -> Dict[str, int]:
    """Parse LLM JSON output into {gt_var: binary_prediction} for the 3 eval vars."""
    raw_items = parse_llm_output(raw)
    if raw_items is None:
        return {v: 0 for v in EVAL_VARS}
    enriched = validate_and_enrich(raw_items)
    kb_by_name = {r["variable"]: r["intensity"] for r in enriched}
    return {
        gt_var: int(kb_by_name.get(kb_var, 0) > 0)
        for kb_var, gt_var in KB_VAR_MAP.items()
    }


class AnthropicBackend:
    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic") from e
        self.client = anthropic.Anthropic()
        self.model  = model

    def kb_predict(self, narrative: str) -> Dict[str, int]:
        prompt = build_prompt(narrative)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return _parse_kb_output(raw)


class OpenRouterBackend:
    """OpenAI-compatible backend pointing at OpenRouter.
    Reads the API key from ANTHROPIC_API_KEY or OPENROUTER_API_KEY env var."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str) -> None:
        import os
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai") from e
        api_key = (os.environ.get("OPENROUTER_API_KEY")
                   or os.environ.get("ANTHROPIC_API_KEY"))
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env")
        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key)
        self.model  = model

    def kb_predict(self, narrative: str) -> Dict[str, int]:
        prompt = build_prompt(narrative)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        return _parse_kb_output(raw)


class HFBackend:
    def __init__(self, model: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("pip install torch transformers") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto"
        )
        self.model_obj.eval()

    def kb_predict(self, narrative: str) -> Dict[str, int]:
        import torch
        prompt = build_prompt(narrative)
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model_obj.device)
        with torch.no_grad():
            out = self.model_obj.generate(
                **inputs, max_new_tokens=1200, do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return _parse_kb_output(raw)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": tp + fn}


def print_results(all_metrics: Dict) -> None:
    sep = "-" * 82
    for var in EVAL_VARS:
        print(f"\n{sep}")
        print(f"  Variable: {var}  (support = {all_metrics[var]['llm']['support']} positives / {all_metrics[var]['llm']['tp']+all_metrics[var]['llm']['fp']+all_metrics[var]['llm']['fn']+all_metrics[var]['llm']['tn']} docs)")
        print(sep)
        print(f"  {'Metric':<12} {'FreeForm-Gemma':>14} {'KB-LLM':>14} {'Delta KB-LLM':>14}")
        print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
        for metric in ["precision", "recall", "f1", "accuracy"]:
            lv = all_metrics[var]["llm"][metric]
            kv = all_metrics[var]["kb"][metric]
            d  = kv - lv
            sign = "+" if d >= 0 else ""
            print(f"  {metric:<12} {lv:>14.3f} {kv:>14.3f} {sign}{d:>13.3f}")
        m_llm = all_metrics[var]["llm"]
        m_kb  = all_metrics[var]["kb"]
        print(f"\n  FreeForm-Gemma: TP={m_llm['tp']} FP={m_llm['fp']} FN={m_llm['fn']} TN={m_llm['tn']}")
        print(f"  KB-LLM:         TP={m_kb['tp']}  FP={m_kb['fp']}  FN={m_kb['fn']}  TN={m_kb['tn']}")

    print(f"\n{sep}")
    print(f"  MACRO AVERAGE across {len(EVAL_VARS)} variables")
    print(sep)
    print(f"  {'Metric':<12} {'FreeForm-Gemma':>14} {'KB-LLM':>14} {'Delta KB-LLM':>14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    for metric in ["precision", "recall", "f1", "accuracy"]:
        lv = sum(all_metrics[v]["llm"][metric] for v in EVAL_VARS) / len(EVAL_VARS)
        kv = sum(all_metrics[v]["kb"][metric]  for v in EVAL_VARS) / len(EVAL_VARS)
        d  = kv - lv
        sign = "+" if d >= 0 else ""
        print(f"  {metric:<12} {lv:>14.3f} {kv:>14.3f} {sign}{d:>13.3f}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--narratives-csv", required=True, type=Path)
    parser.add_argument("--step1c-json",    required=True, type=Path)
    parser.add_argument("--backend",        required=True, choices=["anthropic", "openrouter", "hf"])
    parser.add_argument("--model",          required=True,
                        help="Model name/path: Anthropic model ID or HF model path.")
    parser.add_argument("--n-docs",         type=int, default=None,
                        help="Number of documents to evaluate (random sample). Default: all.")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--output-csv",     type=Path, default=None)
    parser.add_argument("--text-col",       default="narrative")
    parser.add_argument("--doc-id-col",     default="doc_id")
    args = parser.parse_args()

    # Load ground truth
    gt: Dict[str, Dict] = {}
    with args.narratives_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gt[row[args.doc_id_col]] = {
                "text":              row[args.text_col],
                "early_adversity":   int(row["early_adversity"]),
                "social_withdrawal": int(row["social_withdrawal"]),
                "rumination":        int(row["rumination"]),
            }

    # Load step1c candidates
    with args.step1c_json.open("r", encoding="utf-8") as f:
        llm_cands: Dict[str, List[str]] = {
            r["doc_id"]: r["candidates"] for r in json.load(f)
        }

    # Select documents
    all_ids = sorted(set(gt) & set(llm_cands))
    if args.n_docs is not None and args.n_docs < len(all_ids):
        rng = random.Random(args.seed)
        doc_ids = rng.sample(all_ids, args.n_docs)
    else:
        doc_ids = all_ids
    print(f"Documents to evaluate: {len(doc_ids)}")

    # Instantiate KB backend
    if args.backend == "anthropic":
        backend = AnthropicBackend(model=args.model)
    elif args.backend == "openrouter":
        backend = OpenRouterBackend(model=args.model)
    else:
        backend = HFBackend(model=args.model)

    # Collect predictions
    y_true     = {v: [] for v in EVAL_VARS}
    y_pred_llm = {v: [] for v in EVAL_VARS}
    y_pred_kb  = {v: [] for v in EVAL_VARS}
    per_doc_rows = []

    for i, doc_id in enumerate(doc_ids):
        text       = gt[doc_id]["text"]
        candidates = llm_cands[doc_id]
        kb_preds   = backend.kb_predict(text)

        row = {"doc_id": doc_id}
        for var in EVAL_VARS:
            truth    = gt[doc_id][var]
            llm_pred = llm_predict(candidates, var)
            kb_pred  = kb_preds[var]
            y_true[var].append(truth)
            y_pred_llm[var].append(llm_pred)
            y_pred_kb[var].append(kb_pred)
            row[f"{var}_gt"]  = truth
            row[f"{var}_llm"] = llm_pred
            row[f"{var}_kb"]  = kb_pred
        per_doc_rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(doc_ids):
            print(f"  [{i+1}/{len(doc_ids)}]  {doc_id}")

    # Compute and display metrics
    all_metrics = {
        var: {
            "llm": compute_metrics(y_true[var], y_pred_llm[var]),
            "kb":  compute_metrics(y_true[var], y_pred_kb[var]),
        }
        for var in EVAL_VARS
    }
    print_results(all_metrics)

    # Optional CSV
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_doc_rows[0].keys()))
            w.writeheader()
            w.writerows(per_doc_rows)
        print(f"\nPer-document CSV -> {args.output_csv}")


if __name__ == "__main__":
    main()
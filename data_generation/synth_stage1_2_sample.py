#!/usr/bin/env python3
"""
Synthetic data generation: Stage 1 (SCM sampling) and Stage 2 (noise sampling).

Stage 1: Sample binary variable vectors from the structural causal model (SCM).
         Sampling respects the ground truth DAG's topological order and the
         conditional probability table agreed upon in the design session.

Stage 2: For each sampled person, independently draw 1-3 noise variables that
         will appear in their narrative but have no causal relationship to
         depression.

Output: a JSON list of records, one per document, ready to feed into
        Stage 3 (prompt construction) and Stage 4 (LLM narrative generation).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


# ── Ground truth DAG ───────────────────────────────────────────────────────────

# Sampling must follow topological order (parents before children).
SAMPLING_ORDER: List[str] = [
    "early_adversity",
    "chronic_stress",
    "emotion_dysregulation",
    "social_withdrawal",
    "rumination",
    "depression",
]

RESPONSE_VARIABLE = "depression"

# Parent sets for each node.
PARENTS: Dict[str, List[str]] = {
    "early_adversity":      [],
    "chronic_stress":       ["early_adversity"],
    "emotion_dysregulation":["early_adversity", "chronic_stress"],
    "social_withdrawal":    ["chronic_stress"],
    "rumination":           ["emotion_dysregulation", "social_withdrawal"],
    "depression":           ["early_adversity", "rumination"],
}

# Conditional probability table: P(node = 1 | parent values).
# Keys are sorted tuples of (parent_name, parent_value) pairs.
CPT: Dict[str, Dict[Tuple, float]] = {
    "early_adversity": {
        (): 0.30,
    },
    "chronic_stress": {
        (("early_adversity", 0),): 0.25,
        (("early_adversity", 1),): 0.65,
    },
    "emotion_dysregulation": {
        (("chronic_stress", 0), ("early_adversity", 0)): 0.10,
        (("chronic_stress", 0), ("early_adversity", 1)): 0.35,
        (("chronic_stress", 1), ("early_adversity", 0)): 0.30,
        (("chronic_stress", 1), ("early_adversity", 1)): 0.65,
    },
    "social_withdrawal": {
        (("chronic_stress", 0),): 0.15,
        (("chronic_stress", 1),): 0.55,
    },
    "rumination": {
        (("emotion_dysregulation", 0), ("social_withdrawal", 0)): 0.10,
        (("emotion_dysregulation", 0), ("social_withdrawal", 1)): 0.35,
        (("emotion_dysregulation", 1), ("social_withdrawal", 0)): 0.40,
        (("emotion_dysregulation", 1), ("social_withdrawal", 1)): 0.72,
    },
    "depression": {
        (("early_adversity", 0), ("rumination", 0)): 0.05,
        (("early_adversity", 0), ("rumination", 1)): 0.45,
        (("early_adversity", 1), ("rumination", 0)): 0.15,
        (("early_adversity", 1), ("rumination", 1)): 0.70,
    },
}

# ── Noise variable pool ────────────────────────────────────────────────────────

# Sampled independently of the DAG — no causal link to depression.
NOISE_POOL: List[str] = [
    "diet_change",
    "housing_change",
    "work_promotion",
    "new_hobby",
    "travel"
]


# ── Stage 1: SCM sampling ──────────────────────────────────────────────────────

def sample_node(node: str, state: Dict[str, int], rng: random.Random) -> int:
    key = tuple(sorted((p, state[p]) for p in PARENTS[node]))
    prob = CPT[node][key]
    return 1 if rng.random() < prob else 0


def sample_person(rng: random.Random) -> Dict[str, int]:
    state: Dict[str, int] = {}
    for node in SAMPLING_ORDER:
        state[node] = sample_node(node, state, rng)
    return state


# ── Stage 2: Noise sampling ────────────────────────────────────────────────────

def sample_noise(rng: random.Random, max_noise: int) -> List[str]:
    n = rng.randint(0, max_noise)  # 0 is equally likely as any other count
    if n == 0:
        return []
    return rng.sample(NOISE_POOL, k=min(n, len(NOISE_POOL)))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1+2: Sample SCM binary vectors and noise variables."
    )
    parser.add_argument("--n-documents", type=int, default=100,
                        help="Number of synthetic documents to generate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--max-noise", type=int, default=3,
                        help="Maximum number of noise variables per document (0 is always equally likely).")
    parser.add_argument("--output-json", required=True, type=Path,
                        help="Path to write the output JSON.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = []

    for i in range(args.n_documents):
        state = sample_person(rng)
        noise = sample_noise(rng, args.max_noise)

        # Active DAG variables are those sampled as 1, excluding the response.
        active = [v for v in SAMPLING_ORDER if v != RESPONSE_VARIABLE and state[v] == 1]

        records.append({
            "doc_id":              f"doc_{i:04d}",
            "binary_vector":       {v: state[v] for v in SAMPLING_ORDER},
            "active_dag_variables": active,
            "noise_variables":     noise,
            "response_value":      state[RESPONSE_VARIABLE],
        })

    # ── Summary statistics ─────────────────────────────────────────────────────
    n = args.n_documents
    n_pos = sum(r["response_value"] for r in records)
    n_neg = n - n_pos
    print(f"Generated {n} documents")
    print(f"  Y=1 (depression):    {n_pos:4d}  ({100 * n_pos / n:.1f}%)")
    print(f"  Y=0 (no depression): {n_neg:4d}  ({100 * n_neg / n:.1f}%)")
    print()

    # Per-variable activation rates
    print("DAG variable activation rates:")
    for v in SAMPLING_ORDER:
        rate = sum(r["binary_vector"][v] for r in records) / n
        print(f"  {v:<25} {100 * rate:.1f}%")
    print()

    # Noise variable frequencies
    from collections import Counter
    noise_count_dist = Counter(len(r["noise_variables"]) for r in records)
    noise_var_counts = Counter(v for r in records for v in r["noise_variables"])

    print("Noise count distribution:")
    for k in range(args.max_noise + 1):
        count = noise_count_dist.get(k, 0)
        print(f"  {k} noise variable(s): {count:4d}  ({100 * count / n:.1f}%)")
    print()

    print("Noise variable frequency:")
    for v in NOISE_POOL:
        count = noise_var_counts.get(v, 0)
        print(f"  {v:<25} {count:4d}  ({100 * count / n:.1f}%)")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()

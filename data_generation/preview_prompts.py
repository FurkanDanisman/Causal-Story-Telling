#!/usr/bin/env python3
"""Preview N generated prompts without running any LLM."""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from synth_stage1_2_sample import sample_person, sample_noise
from synth_stage3_generate_narratives import build_prompt

SAMPLING_ORDER = [
    "early_adversity", "chronic_stress", "emotion_dysregulation",
    "social_withdrawal", "rumination", "depression",
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Preview generated prompts.")
    parser.add_argument("--n", type=int, default=5, help="Number of prompts to preview.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-noise", type=int, default=3)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    for i in range(args.n):
        state  = sample_person(rng)
        noise  = sample_noise(rng, args.max_noise)
        active = [v for v in SAMPLING_ORDER if v != "depression" and state[v] == 1]
        record = {
            "doc_id":               f"doc_{i:04d}",
            "active_dag_variables": active,
            "noise_variables":      noise,
            "response_value":       state["depression"],
            "binary_vector":        {v: state[v] for v in SAMPLING_ORDER},
        }

        print(f"\n{'='*70}")
        print(f"doc_{i:04d}  |  active={active}  |  noise={noise}  |  Y={state['depression']}")
        print(f"{'='*70}\n")
        print(build_prompt(record))

if __name__ == "__main__":
    main()

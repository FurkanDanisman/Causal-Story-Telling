#!/bin/bash
#SBATCH --job-name=causal_narratives
#SBATCH --partition=gpubase_h100_b1   # H100 80GB x2 = 160GB VRAM, fits 70B in float16
#SBATCH --gres=gpu:2                  # 2 H100s required for Llama-3.3-70B (~140GB VRAM)
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR=~/projects/aip-rgrosse/furkanbd/causal-stories
MODEL=/model-weights/Llama-3.3-70B-Instruct

# ── Environment ────────────────────────────────────────────────────────────────
source $PROJECT_DIR/../.venv/bin/activate

# ── Run ────────────────────────────────────────────────────────────────────────
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/out

python3 "$PROJECT_DIR/Data generation/synth_stage3_generate_narratives.py" \
    --samples-json  "$PROJECT_DIR/out/samples.json" \
    --output-jsonl  "$PROJECT_DIR/out/narratives.jsonl" \
    --hf-model      $MODEL \
    --max-new-tokens 350 \
    --temperature   0.7 \
    --checkpoint-every 10

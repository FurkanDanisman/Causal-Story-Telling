#!/bin/bash
#SBATCH --job-name=causal_extract_collapse
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR=~/projects/aip-rgrosse/furkanbd/causal-stories
MODEL=/model-weights/Meta-Llama-3.1-70B-Instruct
PIPELINE=$PROJECT_DIR/pipeline
OUT=$PROJECT_DIR/out/pipeline

source $PROJECT_DIR/../.venv/bin/activate
mkdir -p $OUT $PROJECT_DIR/logs

echo "=== Step 1: Extract candidate variables ==="
python3 $PIPELINE/step1_extract_candidates.py \
    --input-csv      $PROJECT_DIR/out/pipeline_input.csv \
    --output-json    $OUT/step1_candidates.json \
    --hf-model       $MODEL \
    --max-new-tokens 180 \
    --temperature    0.0

echo "=== Step 1b: Collapse candidates within document ==="
python3 $PIPELINE/step1b_collapse_candidates.py \
    --input-json     $OUT/step1_candidates.json \
    --output-json    $OUT/step1b_candidates_collapsed.json \
    --hf-model       $MODEL \
    --max-new-tokens 120 \
    --temperature    0.0

echo "=== Done. Run step1c normalization next, then step2 onwards ==="

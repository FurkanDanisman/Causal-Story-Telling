#!/bin/bash
#SBATCH --job-name=causal_gemma_test
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR=~/projects/aip-rgrosse/furkanbd/causal-stories
MODEL=~/projects/aip-rgrosse/furkanbd/model-weights/gemma-4-31B-it

source $PROJECT_DIR/../.venv/bin/activate
mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/out

python3 $PROJECT_DIR/data_generation/synth_stage3_generate_narratives.py \
    --samples-json    $PROJECT_DIR/out/samples_gemma.json \
    --output-jsonl    $PROJECT_DIR/out/narratives_gemma.jsonl \
    --hf-model        $MODEL \
    --max-new-tokens  350 \
    --temperature     0.7 \
    --checkpoint-every 10

echo "=== Gemma test done ==="

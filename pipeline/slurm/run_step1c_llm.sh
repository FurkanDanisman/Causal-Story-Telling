#!/bin/bash
#SBATCH --job-name=causal_step1c_llm
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
mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/out/pipeline

echo "=== Step 1c (LLM): Normalize candidates ==="
python3 $PROJECT_DIR/pipeline/step1c_normalize_llm.py \
    --input-json       $PROJECT_DIR/out/pipeline/step1b_collapsed.json \
    --output-json      $PROJECT_DIR/out/pipeline/step1c_normalized.json \
    --cluster-map-json $PROJECT_DIR/out/pipeline/step1c_cluster_map.json \
    --hf-model         $MODEL \
    --max-new-tokens   4000

echo "=== Step 1c done ==="

#!/bin/bash
#SBATCH --job-name=causal_steps1to2
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR=~/projects/aip-rgrosse/furkanbd/causal-stories
MODEL=/model-weights/Llama-3.3-70B-Instruct
PIPELINE=$PROJECT_DIR/pipeline
OUT=$PROJECT_DIR/out/pipeline

source $PROJECT_DIR/../.venv/bin/activate
mkdir -p $OUT $PROJECT_DIR/logs

echo "=== Step 1: Extract candidate variables ==="
python3 $PIPELINE/step1_extract_candidates.py \
    --input-csv     $PROJECT_DIR/out/pipeline_input.csv \
    --output-json   $OUT/step1_candidates.json \
    --hf-model      $MODEL \
    --device        cuda \
    --max-new-tokens 180 \
    --temperature   0.0

echo "=== Step 1b: Normalize candidate variable names ==="
python3 $PIPELINE/step1b_normalize_candidates.py \
    --candidates-json $OUT/step1_candidates.json \
    --output-json     $OUT/step1b_normalized.json \
    --hf-model        $MODEL \
    --device          cuda \
    --batch-size      60

echo "=== Step 2: Build directed edge prompts ==="
python3 $PIPELINE/step2_build_edge_prompts.py \
    --input-csv       $PROJECT_DIR/out/pipeline_input.csv \
    --candidates-json $OUT/step1b_normalized.json \
    --output-jsonl    $OUT/step2_edge_prompts.jsonl

echo "=== Steps 1-2 complete. Submit run_pipeline_step3.sh next ==="
echo "Edge prompts written to: $OUT/step2_edge_prompts.jsonl"
wc -l $OUT/step2_edge_prompts.jsonl

#!/bin/bash
#SBATCH --job-name=causal_step3
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

echo "=== Step 3: Score edges via MC Yes/No logits (checkpoint/resume) ==="
echo "Total edge prompts: $(wc -l < $OUT/step2_edge_prompts.jsonl)"

python3 $PIPELINE/step3_score_edges_mc.py \
    --edge-prompts-jsonl $OUT/step2_edge_prompts.jsonl \
    --output-csv         $OUT/step3_edge_scores.csv \
    --hf-model           $MODEL \
    --device             cuda \
    --mc-samples         4 \
    --seed               42

echo "Edges scored so far: $(tail -n +2 $OUT/step3_edge_scores.csv | wc -l) / $(wc -l < $OUT/step2_edge_prompts.jsonl)"

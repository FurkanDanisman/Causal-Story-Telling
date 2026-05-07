#!/bin/bash
#SBATCH --job-name=causal_narratives
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e

PROJECT_DIR=~/projects/aip-rgrosse/furkanbd/causal-stories
MODEL=/model-weights/Meta-Llama-3.1-70B-Instruct

source $PROJECT_DIR/../.venv/bin/activate
mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/out

CHUNK=250
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: records ${START} to ${END} ==="

python3 $PROJECT_DIR/data_generation/synth_stage3_generate_narratives.py \
    --samples-json      $PROJECT_DIR/out/samples.json \
    --output-jsonl      $PROJECT_DIR/out/narratives_${SLURM_ARRAY_TASK_ID}.jsonl \
    --hf-model          $MODEL \
    --max-new-tokens    350 \
    --temperature       0.7 \
    --checkpoint-every  10 \
    --start-idx         $START \
    --end-idx           $END

echo "=== Task ${SLURM_ARRAY_TASK_ID} done ==="

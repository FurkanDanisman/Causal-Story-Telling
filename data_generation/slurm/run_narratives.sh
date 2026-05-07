#!/bin/bash
#SBATCH --job-name=causal_sample
#SBATCH --partition=default
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR=~/projects/aip-rgrosse/furkanbd/causal-stories

source $PROJECT_DIR/../.venv/bin/activate
mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/out

echo "=== Stage 1+2: Sample SCM ==="
python3 $PROJECT_DIR/data_generation/synth_stage1_2_sample.py \
    --n-documents   1000 \
    --seed          42 \
    --output-json   $PROJECT_DIR/out/samples.json

echo "=== Done. Submit run_narratives_array.sh next ==="

#!/bin/bash
#SBATCH --job-name=causal_steps4to6
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR=~/projects/aip-rgrosse/furkanbd/causal-stories
PIPELINE=$PROJECT_DIR/pipeline
OUT=$PROJECT_DIR/out/pipeline

source $PROJECT_DIR/../.venv/bin/activate
mkdir -p $PROJECT_DIR/logs

# Verify step 3 is complete before running
TOTAL=$(wc -l < $OUT/step2_edge_prompts.jsonl)
SCORED=$(tail -n +2 $OUT/step3_edge_scores.csv | wc -l)
if [ "$SCORED" -lt "$TOTAL" ]; then
    echo "ERROR: Step 3 incomplete. Scored $SCORED / $TOTAL edges. Resubmit run_pipeline_step3.sh."
    exit 1
fi
echo "Step 3 complete: $SCORED / $TOTAL edges scored."

echo "=== Step 4: Aggregate to dataset-level edges ==="
python3 $PIPELINE/step4_aggregate_dataset_edges.py \
    --doc-candidates-json       $OUT/step1b_normalized.json \
    --doc-edge-csv              $OUT/step3_edge_scores.csv \
    --dataset-edge-out-csv      $OUT/step4_dataset_edges.csv \
    --variable-set-out-json     $OUT/step4_variable_set.json \
    --expanded-doc-edge-out-csv $OUT/step4_expanded_doc_edges.csv \
    --use-ipw

echo "=== Step 5: Threshold pairwise relations ==="
python3 $PIPELINE/step5_threshold_relations.py \
    --dataset-edge-csv           $OUT/step4_dataset_edges.csv \
    --directed-classes-out-csv   $OUT/step5_directed_classes.csv \
    --pairwise-relations-out-csv $OUT/step5_pairwise_relations.csv \
    --tau-low  0.3 \
    --tau-high 0.6

echo "=== Step 6: Extract response subgraph ==="
python3 $PIPELINE/step6_extract_response_subgraph.py \
    --dataset-edge-csv   $OUT/step4_dataset_edges.csv \
    --response-variable  depression \
    --tau-high           0.6 \
    --output-json        $OUT/step6_response_subgraph.json

echo "=== Pipeline complete. Results in $OUT ==="
cat $OUT/step6_response_subgraph.json

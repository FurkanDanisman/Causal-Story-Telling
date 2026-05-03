# Causal Story Telling

Recovering causal DAGs from therapy session narratives using open-source LLMs.

The project tests whether an LLM can read synthetic therapy transcripts and reconstruct the ground truth causal structure underlying a patient's depression — without any access to numeric data, only natural language.

---

## Ground Truth DAG

The simulation is built around a **6-variable depression DAG** grounded in the ACE literature, Response Styles Theory, and the diathesis-stress model.

```
early_adversity ──┬──► chronic_stress ──┬──► emotion_dysregulation ──► rumination ──► depression
                  │                     │                                    ▲
                  ├──► emotion_dysregulation                                 │
                  │                     └──► social_withdrawal ──────────────┘
                  └──────────────────────────────────────────────────────────► depression
```

| Variable | Role |
|---|---|
| `early_adversity` | Confounder — causes chronic_stress, emotion_dysregulation, and depression directly |
| `chronic_stress` | Exposure |
| `emotion_dysregulation` | Mediator |
| `social_withdrawal` | Mediator |
| `rumination` | Proximal cause |
| `depression` | Outcome Y |

See [ground_truth/ground_truth_dag.md](ground_truth/ground_truth_dag.md) for the full edge-by-edge literature justification.

---

## Project Structure

```
.
├── data_generation/          # Synthetic narrative generation
│   ├── synth_stage1_2_sample.py        # Sample SCM parameters and patient profiles
│   ├── synth_stage3_generate_narratives.py  # Generate therapy transcripts via LLM
│   ├── audit_and_filter_narratives.py  # Filter out bleed-through artifacts
│   ├── prepare_pipeline_input.py       # Sample 250 records for pipeline
│   ├── prompt_template.md
│   └── slurm/
│       └── run_narratives.sh
│
├── pipeline/                 # Causal inference pipeline (Steps 1–6)
│   ├── step1_extract_candidates.py     # Extract candidate causal variables per document
│   ├── step1b_normalize_candidates.py  # Normalize variable names across documents
│   ├── step2_build_edge_prompts.py     # Build directed-edge prompts for all variable pairs
│   ├── step3_score_edges_mc.py         # Score edges via MC Yes/No logits (checkpoint/resume)
│   ├── step4_aggregate_dataset_edges.py # Aggregate per-document scores to dataset level (IPW)
│   ├── step5_threshold_relations.py    # Threshold pairwise relations (τ_low, τ_high)
│   ├── step6_extract_response_subgraph.py # Extract subgraph of ancestors of Y
│   ├── PIPELINE_SPEC.md
│   └── slurm/
│       ├── run_pipeline_steps1to2.sh
│       ├── run_pipeline_step3.sh       # Resubmit until all edges are scored
│       └── run_pipeline_steps4to6.sh
│
└── ground_truth/             # Ground truth DAG definition and visualization
    ├── ground_truth_dag.md
    ├── ground_truth_dag.png
    └── visualize_dag.py
```

---

## Pipeline Overview

The pipeline takes a CSV of therapy narratives and outputs an estimated causal DAG.

### Step 1 — Extract Candidate Variables
For each document, an LLM reads the narrative and names the causal constructs it detects (e.g., "childhood trauma", "social isolation"). These are free-text candidates.

### Step 1b — Normalize Candidates
A second LLM pass maps free-text candidates to a canonical variable set, so "childhood trauma" and "early adversity" become the same node across documents.

### Step 2 — Build Edge Prompts
For every ordered pair of normalized candidates `(Ci, Cj)` within a document, a structured prompt is constructed asking whether the document supports a **direct** causal edge `Ci → Cj`.

### Step 3 — Score Edges via MC Yes/No Logits
The LLM scores each prompt by reading the Yes/No token logits at the final position (single forward pass). This is averaged over `B=4` Monte Carlo samples with different prompt prefixes to reduce variance. Output: per-document edge probability `P_ij(D)`.

This step supports **checkpoint/resume**: already-scored edges are skipped on resubmission.

### Step 4 — Aggregate to Dataset Level
Per-document edge scores are aggregated to a single dataset-level score `P_ij` using **Inverse Probability Weighting (IPW)** to correct for the imbalanced outcome distribution (Y=1 narratives are upweighted).

### Step 5 — Threshold Pairwise Relations
Each pair `(i, j)` is classified using two thresholds `τ_low < τ_high`:
- `P_ij ≥ τ_high` → `i → j` (directed forward)
- `P_ji ≥ τ_high` → `j → i` (directed backward)
- Both scores in `(τ_low, τ_high)` → undirected association
- Both scores `< τ_low` → no relation

### Step 6 — Extract Response Subgraph
Find all variables that have a directed path into `depression` in the thresholded graph.

---

## Synthetic Data Generation

Narratives are generated in three stages:

1. **Sample SCM parameters** — sample structural equation model coefficients and patient profiles from the ground truth DAG using logistic noise.
2. **Generate narratives** — prompt an LLM to write a first-person therapy session monologue for each patient profile. Label `Y=1` if the patient meets the depression threshold.
3. **Filter artifacts** — strip model meta-commentary (bleed-through), check label consistency, and write `reliable_narratives.jsonl`.

Final dataset: **779 reliable narratives** (Y=1: ~19%, Y=0: ~81%). Step 3 samples 250 for the pipeline run (seed=42).

---

## Running on a SLURM Cluster

The cluster scripts target `killarney.alliancecan.ca` (Alliance Canada HPC) with H100 GPUs. Adjust `PROJECT_DIR` and `MODEL` paths in each script.

```bash
# Steps 1–2: extract candidates and build edge prompts (~1–2 hours)
sbatch pipeline/slurm/run_pipeline_steps1to2.sh

# Step 3: score edges — resubmit until complete (~2 hours per submission)
sbatch pipeline/slurm/run_pipeline_step3.sh
# check progress:
tail -n +2 out/pipeline/step3_edge_scores.csv | wc -l

# Steps 4–6: aggregate, threshold, extract subgraph (CPU only, ~30 min)
sbatch pipeline/slurm/run_pipeline_steps4to6.sh
```

Step 3 uses checkpoint/resume — it appends to `step3_edge_scores.csv` and skips already-scored edges on each resubmission.

---

## Requirements

```
torch>=2.1.0
transformers>=4.45.0
accelerate>=0.27.0
networkx>=3.0
matplotlib>=3.7
```

Install with:
```bash
pip install -r requirements.txt
```

The pipeline is tested with **Llama-3.3-70B-Instruct** in fp16 across 2×H100 GPUs. Any instruction-tuned causal LM with a HuggingFace chat template should work.

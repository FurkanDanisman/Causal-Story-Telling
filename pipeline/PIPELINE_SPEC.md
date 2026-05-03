# Causal Stories Pipeline Spec

## Step Scripts
- `step1_extract_candidates.py`
- `step2_build_edge_prompts.py`
- `step3_score_edges_mc.py`
- `step4_aggregate_dataset_edges.py`
- `step5_threshold_relations.py`
- `step6_extract_response_subgraph.py`

## Implemented stages
1. Candidate-variable extraction from each narrative text.
2. Directed edge prompt construction for each ordered pair `(Ci, Cj)`.
3. Monte Carlo estimation:
   - `P_ij(D) = E[P(Yes | D, claim=Ci->Cj, C\\{Ci,Cj}, U)]`
   - Approximated with `B` samples:
   - `P_ij(D) ~= (1/B) * sum_b P_ij,b(D)`
4. Common variable set across documents and zero-fill for missing variables.
5. Dataset-level averaging:
   - `P_ij(dataset) = average_n P_ij(D^n)`
6. Thresholding with `tau_low < tau_high` and pairwise relation interpretation.
7. Response-specific subgraph extraction:
   - Find all variables with a directed path into `Y`.

## Prompt structure used in code
```text
Document:
[D]

Candidate variables:
[C1, C2, ..., CK]

Target directed edge:
Ci -> Cj

Other candidate variables that could lie between Ci and Cj:
[C \ {Ci, Cj}]

Task:
Decide whether the document supports a direct causal edge from Ci to Cj.

Definition of direct edge:
A direct edge Ci -> Cj is supported only if the document suggests that Ci affects,
changes, produces, worsens, improves, triggers, or contributes to Cj without
requiring another listed candidate variable as an intermediate step.

Do not answer Yes if the document only supports an indirect pathway such as:
Ci -> Ck -> Cj

Do not answer Yes only because Ci happens before Cj or is associated with Cj.

Question:
Does the document support a direct causal edge Ci -> Cj?

Answer with one word only: Yes or No.

Answer:
```

## Model Policy
- Candidate extraction is prompt-based and uses an open-source model.
- Edge scoring is prompt-based and uses an open-source model with Yes/No continuation logits.
- No lexicon defaults, no rule-based extraction, no heuristic scoring backend.

## Outputs written by pipeline
- `doc_candidates.json`
- `doc_edge_probabilities.csv`
- `dataset_edge_probabilities.csv`
- `dataset_pairwise_relations.csv`
- `response_subgraph.json`
- `prompt_template.txt`
- `run_metadata.json`

# Step 1 Extraction Methods — Comparison Results

**Date:** 2026-05-15  
**Sample:** 100 documents (random seed 42, drawn from `narratives_gemma.csv`)  
**Raw predictions:** `step1_comparison_100.csv`

---

## Methods

| Method | Model | Variable naming | Normalization |
|--------|-------|----------------|---------------|
| **FreeForm-Gemma** | Gemma-4-31B-it | Free-form (LLM-invented) | Cross-document embedding clustering (SentenceTransformer + AgglomerativeClustering) → canonical label |
| **KB-LLM** | Claude Haiku 4.5 (via OpenRouter) | Fixed — Beck & Bredemeier (2016) taxonomy (15 variables) | None — taxonomy enforces a closed variable set |

---

## Evaluated Variables

Three variables with unambiguous ground-truth mappings:

| Ground truth (DAG) | FreeForm-Gemma mapping | KB-LLM mapping |
|--------------------|----------------------|----------------|
| `early_adversity` | keyword match on normalized candidates (`adverse`, `childhood`, `trauma`, `abuse`, `neglect`, `early_life`, `upbring`) | taxonomy variable `early_life_trauma` (intensity > 0) |
| `social_withdrawal` | keyword match (`isolat`, `withdraw`) | taxonomy variable `social_withdrawal` (intensity > 0) |
| `rumination` | keyword match (`rumin`) | taxonomy variable `rumination` (intensity > 0) |

---

## Results per Variable

### early_adversity  (53 positives / 100 docs)

| Metric | FreeForm-Gemma | KB-LLM | Delta (KB − FF) |
|--------|---------------|--------|----------------|
| Precision | 1.000 | 1.000 | +0.000 |
| Recall | 0.717 | **1.000** | +0.283 |
| F1 | 0.835 | **1.000** | +0.165 |
| Accuracy | 0.850 | **1.000** | +0.150 |

Confusion matrix:

|  | FreeForm-Gemma | KB-LLM |
|--|---------------|--------|
| TP | 38 | 53 |
| FP | 0 | 0 |
| FN | 15 | 0 |
| TN | 47 | 47 |

FreeForm-Gemma misses 15 true positives because the Gemma-generated candidates do not always surface a substring matching the keyword anchors, even when early adversity is present in the narrative.

---

### social_withdrawal  (39 positives / 100 docs)

| Metric | FreeForm-Gemma | KB-LLM | Delta (KB − FF) |
|--------|---------------|--------|----------------|
| Precision | 1.000 | 1.000 | +0.000 |
| Recall | 1.000 | 1.000 | +0.000 |
| F1 | **1.000** | **1.000** | +0.000 |
| Accuracy | 1.000 | 1.000 | +0.000 |

Both methods achieve perfect scores. `social_withdrawal` is the most lexically stable variable — its surface forms are consistent enough for both the keyword anchor and the clinical taxonomy to reliably detect it.

---

### rumination  (42 positives / 100 docs)

| Metric | FreeForm-Gemma | KB-LLM | Delta (KB − FF) |
|--------|---------------|--------|----------------|
| Precision | **1.000** | 0.483 | -0.517 |
| Recall | 0.405 | **1.000** | +0.595 |
| F1 | 0.576 | 0.651 | +0.075 |
| Accuracy | **0.750** | 0.550 | -0.200 |

Confusion matrix:

|  | FreeForm-Gemma | KB-LLM |
|--|---------------|--------|
| TP | 17 | 42 |
| FP | 0 | 45 |
| FN | 25 | 0 |
| TN | 58 | 13 |

KB-LLM is over-sensitive: it flags 45 false positives out of 58 true negatives. Haiku appears to apply the clinical definition of rumination broadly, classifying chronic stress or emotion dysregulation patterns as rumination even when the ground truth is negative. FreeForm-Gemma's tight substring anchor (`rumin`) yields perfect precision but misses 25 true positives.

---

## Macro Average (3 variables)

| Metric | FreeForm-Gemma | KB-LLM | Delta (KB − FF) |
|--------|---------------|--------|----------------|
| Precision | **1.000** | 0.828 | -0.172 |
| Recall | 0.707 | **1.000** | +0.293 |
| F1 | 0.804 | **0.884** | **+0.080** |
| Accuracy | **0.867** | 0.850 | -0.017 |

---

## Discussion

KB-LLM achieves a higher macro F1 (+0.08) driven by its near-perfect detection of `early_adversity`. Its main weakness is an over-broad interpretation of `rumination`, where it trades high recall (1.000) for low precision (0.483).

FreeForm-Gemma is uniformly high-precision (1.000 across all variables) because its keyword anchors are conservative. However, this conservatism causes significant false negatives on `early_adversity` (recall 0.717) and especially `rumination` (recall 0.405), where the Gemma model often does not produce a normalized candidate containing the substring `rumin`.

A key confound remains: the two methods use **different underlying models** (Gemma vs. Claude Haiku), so observed differences reflect both the extraction strategy and the model capability. A controlled comparison would require running both strategies on the same base model.
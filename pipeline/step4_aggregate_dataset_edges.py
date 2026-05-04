#!/usr/bin/env python3
"""Step 4: Aggregate document-level edge probabilities to dataset level.

Supports two aggregation modes:

  Simple average (default):
    P_ij(dataset) = (1/N) * sum_n P_ij(D^n)

  Inverse-probability-weighted average (--use-ipw):
    Each document is up-weighted in inverse proportion to how common its
    response class is. If Y=1 appears in only 20% of documents and Y=0 in
    80%, documents with Y=1 receive weight 1/0.20 and those with Y=0 receive
    weight 1/0.80, then all weights are normalised to sum to 1.

    This corrects for disproportionate response-variable distributions so
    that the dataset-level graph is not dominated by the majority class.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_candidates(path: Path) -> Dict[str, dict]:
    """Returns {doc_id: record} where record has 'candidates' and 'response_value'."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, dict] = {}
    for rec in data:
        out[str(rec["doc_id"])] = rec
    return out


def read_doc_edges(path: Path) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"doc_id", "source", "target", "p_edge"}
        if not required.issubset(set(r.fieldnames or [])):
            missing = sorted(required - set(r.fieldnames or []))
            raise ValueError(f"Missing columns in doc-edge CSV: {missing}")
        for row in r:
            key = (str(row["doc_id"]), str(row["source"]), str(row["target"]))
            out[key] = float(row["p_edge"])
    return out


def compute_ipw_weights(doc_ids: List[str], doc_records: Dict[str, dict]) -> Dict[str, float]:
    """
    Compute normalised inverse-probability weights based on response_value class frequency.

    Weight for doc n = 1 / P(response_class of doc n)
    where P(class) = count(docs in that class) / total docs.
    Weights are then normalised to sum to 1.

    If all documents share the same response class, IPW reduces to uniform weights.
    """
    response_values = {doc_id: int(doc_records[doc_id].get("response_value", 0))
                       for doc_id in doc_ids}
    class_counts = Counter(response_values.values())
    n = len(doc_ids)

    raw: Dict[str, float] = {}
    for doc_id in doc_ids:
        cls = response_values[doc_id]
        # 1 / P(class) = n / count(class)
        raw[doc_id] = n / class_counts[cls]

    total = sum(raw.values())
    return {doc_id: w / total for doc_id, w in raw.items()}


def compute_uniform_weights(doc_ids: List[str]) -> Dict[str, float]:
    n = len(doc_ids)
    return {doc_id: 1.0 / n for doc_id in doc_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Build dataset-level edge probabilities.")
    parser.add_argument("--doc-candidates-json", required=True, type=Path)
    parser.add_argument("--doc-edge-csv", required=True, type=Path)
    parser.add_argument("--dataset-edge-out-csv", required=True, type=Path)
    parser.add_argument("--variable-set-out-json", required=True, type=Path)
    parser.add_argument("--expanded-doc-edge-out-csv", type=Path, default=None)
    parser.add_argument(
        "--use-ipw",
        action="store_true",
        default=False,
        help=(
            "Weight each document by the inverse of its response-class frequency "
            "to correct for imbalanced Y distributions."
        ),
    )
    args = parser.parse_args()

    doc_records = read_candidates(args.doc_candidates_json)
    doc_edges = read_doc_edges(args.doc_edge_csv)

    doc_ids = sorted(doc_records.keys())
    if not doc_ids:
        raise ValueError("No documents found in candidates JSON.")

    if args.use_ipw:
        weights = compute_ipw_weights(doc_ids, doc_records)
    else:
        weights = compute_uniform_weights(doc_ids)

    variable_set: Set[str] = set()
    for rec in doc_records.values():
        variable_set.update(rec["candidates"])
    vars_sorted = sorted(variable_set)

    write_expanded = args.expanded_doc_edge_out_csv is not None
    expanded_rows = []
    dataset_rows = []

    for source in vars_sorted:
        for target in vars_sorted:
            if source == target:
                continue
            p_dataset = 0.0
            for doc_id in doc_ids:
                p = doc_edges.get((doc_id, source, target), 0.0)
                p_dataset += weights[doc_id] * p
                if write_expanded:
                    expanded_rows.append(
                        {
                            "doc_id": doc_id,
                            "source": source,
                            "target": target,
                            "p_edge": f"{p:.6f}",
                            "weight": f"{weights[doc_id]:.6f}",
                        }
                    )
            dataset_rows.append(
                {
                    "source": source,
                    "target": target,
                    "p_edge": f"{p_dataset:.6f}",
                }
            )

    args.dataset_edge_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset_edge_out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "p_edge"])
        w.writeheader()
        for row in dataset_rows:
            w.writerow(row)

    args.variable_set_out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.variable_set_out_json.open("w", encoding="utf-8") as f:
        json.dump(vars_sorted, f, indent=2)

    if args.expanded_doc_edge_out_csv is not None:
        args.expanded_doc_edge_out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.expanded_doc_edge_out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["doc_id", "source", "target", "p_edge", "weight"]
            )
            w.writeheader()
            for row in expanded_rows:
                w.writerow(row)


if __name__ == "__main__":
    main()

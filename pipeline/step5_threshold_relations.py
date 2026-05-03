#!/usr/bin/env python3
"""Step 5: Threshold dataset-level edge probabilities and classify relations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def classify(p: float, tau_low: float, tau_high: float) -> str:
    if p >= tau_high:
        return "high"
    if p <= tau_low:
        return "low"
    return "uncertain"


def read_dataset_edges(path: Path) -> Tuple[List[str], Dict[Tuple[str, str], float]]:
    edge_map: Dict[Tuple[str, str], float] = {}
    variable_set = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"source", "target", "p_edge"}
        if not required.issubset(set(r.fieldnames or [])):
            missing = sorted(required - set(r.fieldnames or []))
            raise ValueError(f"Missing columns in dataset-edge CSV: {missing}")
        for row in r:
            s = str(row["source"])
            t = str(row["target"])
            p = float(row["p_edge"])
            edge_map[(s, t)] = p
            variable_set.add(s)
            variable_set.add(t)
    return sorted(variable_set), edge_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: Apply thresholds and relation rules.")
    parser.add_argument("--dataset-edge-csv", required=True, type=Path)
    parser.add_argument("--directed-classes-out-csv", required=True, type=Path)
    parser.add_argument("--pairwise-relations-out-csv", required=True, type=Path)
    parser.add_argument("--tau-low", type=float, required=True)
    parser.add_argument("--tau-high", type=float, required=True)
    args = parser.parse_args()

    if not (0.0 <= args.tau_low < args.tau_high <= 1.0):
        raise ValueError("Need 0 <= tau_low < tau_high <= 1.")

    variables, edge_map = read_dataset_edges(args.dataset_edge_csv)

    directed_rows = []
    for s in variables:
        for t in variables:
            if s == t:
                continue
            p = edge_map.get((s, t), 0.0)
            directed_rows.append(
                {
                    "source": s,
                    "target": t,
                    "p_edge": f"{p:.6f}",
                    "edge_class": classify(p, args.tau_low, args.tau_high),
                }
            )

    pair_rows = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            vi = variables[i]
            vj = variables[j]
            pij = edge_map.get((vi, vj), 0.0)
            pji = edge_map.get((vj, vi), 0.0)
            cij = classify(pij, args.tau_low, args.tau_high)
            cji = classify(pji, args.tau_low, args.tau_high)

            if cij == "high" and cji == "low":
                relation = f"{vi} -> {vj}"
            elif cij == "low" and cji == "high":
                relation = f"{vj} -> {vi}"
            elif cij == "low" and cji == "low":
                relation = "no_supported_connection"
            elif cij == "high" and cji == "high":
                relation = "bidirectional_or_mutually_reinforcing"
            else:
                relation = "uncertain_relation"

            pair_rows.append(
                {
                    "var_i": vi,
                    "var_j": vj,
                    "p_i_to_j": f"{pij:.6f}",
                    "p_j_to_i": f"{pji:.6f}",
                    "class_i_to_j": cij,
                    "class_j_to_i": cji,
                    "pair_relation": relation,
                }
            )

    args.directed_classes_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.directed_classes_out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "p_edge", "edge_class"])
        w.writeheader()
        for row in directed_rows:
            w.writerow(row)

    args.pairwise_relations_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.pairwise_relations_out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "var_i",
                "var_j",
                "p_i_to_j",
                "p_j_to_i",
                "class_i_to_j",
                "class_j_to_i",
                "pair_relation",
            ],
        )
        w.writeheader()
        for row in pair_rows:
            w.writerow(row)


if __name__ == "__main__":
    main()


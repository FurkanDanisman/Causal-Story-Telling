#!/usr/bin/env python3
"""Step 6: Extract all variables with a directed path into response variable Y."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_dataset_edges(path: Path) -> Tuple[List[str], Dict[Tuple[str, str], float]]:
    edge_map: Dict[Tuple[str, str], float] = {}
    nodes: Set[str] = set()
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
            nodes.add(s)
            nodes.add(t)
    return sorted(nodes), edge_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Extract response-specific ancestor subgraph.")
    parser.add_argument("--dataset-edge-csv", required=True, type=Path)
    parser.add_argument("--response-variable", required=True, type=str)
    parser.add_argument("--tau-high", required=True, type=float)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    if not (0.0 <= args.tau_high <= 1.0):
        raise ValueError("--tau-high must be within [0, 1].")

    nodes, edge_map = read_dataset_edges(args.dataset_edge_csv)
    if args.response_variable not in nodes:
        nodes.append(args.response_variable)
        nodes = sorted(set(nodes))

    adjacency: Dict[str, List[str]] = {n: [] for n in nodes}
    reverse_adjacency: Dict[str, List[str]] = {n: [] for n in nodes}

    for s in nodes:
        for t in nodes:
            if s == t:
                continue
            p = edge_map.get((s, t), 0.0)
            if p >= args.tau_high:
                adjacency[s].append(t)
                reverse_adjacency[t].append(s)

    ancestors: Set[str] = set()
    stack = list(reverse_adjacency.get(args.response_variable, []))
    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        for parent in reverse_adjacency.get(node, []):
            if parent not in ancestors:
                stack.append(parent)

    sub_nodes = sorted(ancestors | {args.response_variable})
    sub_edges = []
    for s in sub_nodes:
        for t in sub_nodes:
            if s == t:
                continue
            p = edge_map.get((s, t), 0.0)
            if p >= args.tau_high:
                sub_edges.append({"source": s, "target": t, "p_edge": round(p, 6)})

    out = {
        "response_variable": args.response_variable,
        "ancestors": sorted(ancestors),
        "nodes": sub_nodes,
        "edges": sub_edges,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()


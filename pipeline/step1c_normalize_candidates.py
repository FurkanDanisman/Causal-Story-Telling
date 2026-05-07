#!/usr/bin/env python3
"""Step 1c: Cross-document normalization via embedding clustering.

Embeds all unique candidate variable names across all documents, clusters
them by cosine similarity, picks the most frequent label per cluster as the
canonical name, and remaps every document's candidate list.

Input:  JSON from step1b (list of {doc_id, candidates, candidates_raw, ...})
Output: same structure with 'candidates' remapped to canonical names,
        cluster mapping saved separately for inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def embed_names(names: list[str], model_name: str) -> "np.ndarray":
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        raise RuntimeError("step1c requires sentence-transformers and numpy.") from e

    model = SentenceTransformer(model_name)
    return model.encode(names, normalize_embeddings=True, show_progress_bar=True)


def cluster_embeddings(embeddings: "np.ndarray", threshold: float) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    return clustering.fit_predict(embeddings).tolist()


def pick_canonical(names_in_cluster: list[str]) -> str:
    return Counter(names_in_cluster).most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1c: Embed + cluster candidate names.")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--cluster-map-json", required=True, type=Path,
                        help="Output JSON mapping each raw name to its canonical name.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name or local path.")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Cosine distance threshold for agglomerative clustering. "
                             "Lower = tighter clusters. Try 0.10-0.20.")
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    # Collect all unique names and their frequency across documents
    name_counter: Counter = Counter()
    for rec in records:
        name_counter.update(rec["candidates"])

    all_names = sorted(name_counter.keys())
    print(f"Unique candidate names across all documents: {len(all_names)}")

    # Embed
    embeddings = embed_names(all_names, args.embedding_model)

    # Cluster
    labels = cluster_embeddings(embeddings, args.threshold)
    n_clusters = len(set(labels))
    print(f"Clusters formed: {n_clusters}  (threshold={args.threshold})")

    # Group names by cluster, pick canonical label
    from collections import defaultdict
    clusters: dict[int, list[str]] = defaultdict(list)
    for name, label in zip(all_names, labels):
        clusters[label].append(name)

    canonical_map: dict[str, str] = {}
    for label, members in clusters.items():
        # Weight by document frequency when picking canonical name
        weighted = [name for name in members for _ in range(name_counter[name])]
        canonical = pick_canonical(weighted)
        for name in members:
            canonical_map[name] = canonical

    # Print clusters for inspection
    print("\nClusters (raw → canonical):")
    for label, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        canonical = canonical_map[members[0]]
        if len(members) > 1:
            print(f"  [{canonical}]  ←  {members}")

    # Remap documents
    out_records = []
    for rec in records:
        normalized = sorted(set(canonical_map.get(c, c) for c in rec["candidates"]))
        out_rec = {k: v for k, v in rec.items()}
        out_rec["candidates_collapsed"] = rec["candidates"]
        out_rec["candidates"] = normalized
        out_records.append(out_rec)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    args.cluster_map_json.parent.mkdir(parents=True, exist_ok=True)
    with args.cluster_map_json.open("w", encoding="utf-8") as f:
        json.dump(canonical_map, f, indent=2, sort_keys=True)

    print(f"\nDone. Saved → {args.output_json}")
    print(f"Cluster map → {args.cluster_map_json}")


if __name__ == "__main__":
    main()

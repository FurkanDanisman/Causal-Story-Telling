#!/usr/bin/env python3
"""Plot recovered DAG vs ground truth for multiple tau thresholds.

Only nodes that are ancestors of depression (or depression itself) in the
recovered graph are shown.
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
from pathlib import Path

STEP5_CSV = Path("/Users/furkandanisman/Desktop/Causal Story Telling/data_generation/out/step5_directed_classes.csv")
OUT_DIR   = Path("/Users/furkandanisman/Desktop/Causal Story Telling/data_generation/out")

TAU_LOW = 0.02
SCENARIOS = [0.06, 0.05, 0.04, 0.03]

# GT edges in pipeline canonical names
GT_EDGES = {
    ("adverse_childhood_experiences", "stress"),
    ("adverse_childhood_experiences", "emotional_instability"),
    ("adverse_childhood_experiences", "depression"),
    ("stress",                        "emotional_instability"),
    ("stress",                        "social_isolation"),
    ("emotional_instability",         "mental_rumination"),
    ("social_isolation",              "mental_rumination"),
    ("mental_rumination",             "depression"),
}

# Full position map for the 6 GT nodes
FULL_POS = {
    "adverse_childhood_experiences": (0.5,  1.0),
    "stress":                        (0.0,  0.58),
    "emotional_instability":         (0.72, 0.58),
    "social_isolation":              (0.0,  0.1),
    "mental_rumination":             (0.72, 0.1),
    "depression":                    (1.22, 0.38),
}

NODE_COLORS = {
    "adverse_childhood_experiences": "#E07B54",
    "stress":                        "#5B8DB8",
    "emotional_instability":         "#7DB87D",
    "social_isolation":              "#7DB87D",
    "mental_rumination":             "#B87DB8",
    "depression":                    "#D4A843",
}

LABELS = {
    "adverse_childhood_experiences": "Adverse\nChildhood\nExperiences",
    "stress":                        "Stress",
    "emotional_instability":         "Emotional\nInstability",
    "social_isolation":              "Social\nIsolation",
    "mental_rumination":             "Mental\nRumination",
    "depression":                    "Depression",
}

DEFAULT_COLOR = "#999999"


def load_all_edges(path: Path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((row["source"], row["target"], float(row["p_edge"])))
    return rows


def ancestors_of(node, edges):
    """Return all ancestors of node in the directed graph defined by edges."""
    G = nx.DiGraph()
    G.add_edges_from([(s, t) for s, t, *_ in edges])
    if node not in G:
        return set()
    return nx.ancestors(G, node)


def get_scenario_data(all_edges, tau_high):
    recovered = [(s, t) for s, t, p in all_edges if p >= tau_high]
    anc = ancestors_of("depression", [(s, t, p) for s, t, p in all_edges if p >= tau_high])
    keep_nodes = anc | {"depression"}

    rec_in_scope   = [(s, t) for s, t in recovered if s in keep_nodes and t in keep_nodes]
    rec_set_scoped = set(rec_in_scope)

    gt_nodes  = set(FULL_POS.keys())
    pos_nodes = [n for n in (gt_nodes | keep_nodes) if n in FULL_POS]
    pos       = {n: FULL_POS[n] for n in pos_nodes}

    missing_edges = [(s, t) for s, t in GT_EDGES if (s, t) not in rec_set_scoped]
    fp_edges      = [(s, t) for s, t in rec_in_scope if (s, t) not in GT_EDGES]

    n_rec = len(rec_set_scoped & GT_EDGES)
    n_fp  = len(fp_edges)

    rec_gt = [(s, t) for s, t in rec_in_scope if (s, t) in GT_EDGES and s in pos and t in pos]
    mis    = [(s, t) for s, t in missing_edges if s in pos and t in pos]
    fp     = [(s, t) for s, t in fp_edges if s in pos and t in pos]

    G = nx.DiGraph()
    G.add_nodes_from(pos_nodes)
    G.add_edges_from([(s, t) for s, t in rec_in_scope + missing_edges if s in pos and t in pos])

    return dict(G=G, pos=pos, rec_gt=rec_gt, mis=mis, fp=fp,
                n_rec=n_rec, n_fp=n_fp, n_tot=len(GT_EDGES))


def draw_on_ax(ax, data, tau_high):
    G, pos = data["G"], data["pos"]
    n_rec, n_fp, n_tot = data["n_rec"], data["n_fp"], data["n_tot"]

    ax.set_facecolor("#F7F7F7")
    node_colors = [NODE_COLORS.get(n, DEFAULT_COLOR) for n in G.nodes()]
    draw_kw = dict(arrows=True, arrowstyle="-|>", arrowsize=16, node_size=2200,
                   connectionstyle="arc3,rad=0.05")

    if data["mis"]:
        nx.draw_networkx_edges(G, pos, edgelist=data["mis"], ax=ax,
                               edge_color="#aaaaaa", width=1.5, style="dashed", alpha=0.6, **draw_kw)
    if data["rec_gt"]:
        nx.draw_networkx_edges(G, pos, edgelist=data["rec_gt"], ax=ax,
                               edge_color="#2ca02c", width=2.0, **draw_kw)
    if data["fp"]:
        nx.draw_networkx_edges(G, pos, edgelist=data["fp"], ax=ax,
                               edge_color="#d62728", width=1.8, **draw_kw)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=2200, linewidths=1.5, edgecolors="#333333")
    nx.draw_networkx_labels(G, pos, labels={n: LABELS.get(n, n) for n in G.nodes()},
                            ax=ax, font_size=7.5, font_weight="bold")

    legend_handles = [
        mlines.Line2D([], [], color="#2ca02c", linewidth=2.0, label=f"Recovered ({n_rec}/{n_tot})"),
        mlines.Line2D([], [], color="#aaaaaa", linewidth=1.5, linestyle="dashed",
                      label=f"Missing ({len(data['mis'])}/{n_tot})"),
    ]
    if data["fp"]:
        legend_handles.append(
            mlines.Line2D([], [], color="#d62728", linewidth=1.8, label=f"FP ({n_fp})")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.set_title(
        f"tau_high={tau_high}  |  {n_rec}/{n_tot} recovered · {n_fp} FP",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.axis("off")


def main():
    all_edges = load_all_edges(STEP5_CSV)

    # Also save individual plots
    for tau in SCENARIOS:
        data = get_scenario_data(all_edges, tau)
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor("#F7F7F7")
        draw_on_ax(ax, data, tau)
        plt.tight_layout()
        out = OUT_DIR / f"dag_vs_gt_tau{str(tau).replace('.', '')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out}")

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor("#F7F7F7")
    fig.suptitle(
        "Recovered DAG vs Ground Truth — Tau Sensitivity\n(tau_low=0.02)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    for ax, tau in zip(axes.flat, SCENARIOS):
        data = get_scenario_data(all_edges, tau)
        draw_on_ax(ax, data, tau)

    plt.tight_layout()
    out_grid = OUT_DIR / "dag_vs_gt_2x2.png"
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 2x2 → {out_grid}")

    # Update default
    data = get_scenario_data(all_edges, 0.06)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#F7F7F7")
    draw_on_ax(ax, data, 0.06)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dag_vs_gt.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

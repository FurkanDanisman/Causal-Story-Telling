#!/usr/bin/env python3
"""Visualize the ground truth depression DAG."""

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

NODES = {
    "early_adversity":       {"role": "confounder",     "label": "Early\nAdversity"},
    "chronic_stress":        {"role": "exposure",        "label": "Chronic\nStress"},
    "emotion_dysregulation": {"role": "mediator",        "label": "Emotion\nDysregulation"},
    "social_withdrawal":     {"role": "mediator",        "label": "Social\nWithdrawal"},
    "rumination":            {"role": "proximal_cause",  "label": "Rumination"},
    "depression":            {"role": "outcome",         "label": "Depression"},
}

EDGES = [
    ("early_adversity",       "chronic_stress"),
    ("early_adversity",       "emotion_dysregulation"),
    ("early_adversity",       "depression"),
    ("chronic_stress",        "emotion_dysregulation"),
    ("chronic_stress",        "social_withdrawal"),
    ("emotion_dysregulation", "rumination"),
    ("social_withdrawal",     "rumination"),
    ("rumination",            "depression"),
]

# Hand-tuned positions for a clear layout
POS = {
    "early_adversity":       (0.5,  1.0),
    "chronic_stress":        (0.0,  0.58),
    "emotion_dysregulation": (0.72, 0.58),
    "social_withdrawal":     (0.0,  0.1),
    "rumination":            (0.72, 0.1),
    "depression":            (1.22, 0.38),
}

ROLE_COLORS = {
    "confounder":    "#E07B54",   # orange-red
    "exposure":      "#5B8DB8",   # steel blue
    "mediator":      "#7DB87D",   # muted green
    "proximal_cause":"#B87DB8",   # muted purple
    "outcome":       "#D4A843",   # amber
}

ROLE_LABELS = {
    "confounder":    "Confounder",
    "exposure":      "Exposure",
    "mediator":      "Mediator",
    "proximal_cause":"Proximal Cause",
    "outcome":       "Outcome (Y)",
}


def main() -> None:
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    G.add_edges_from(EDGES)

    node_colors = [ROLE_COLORS[NODES[n]["role"]] for n in G.nodes()]
    labels = {n: NODES[n]["label"] for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("#F7F7F7")
    fig.patch.set_facecolor("#F7F7F7")

    # Draw edges — highlight the confounder's direct path to depression in red
    confounder_edges = [e for e in EDGES if e[0] == "early_adversity"]
    other_edges      = [e for e in EDGES if e[0] != "early_adversity"]

    nx.draw_networkx_edges(
        G, POS, edgelist=other_edges, ax=ax,
        edge_color="#555555", arrows=True,
        arrowstyle="-|>", arrowsize=22,
        width=2.0, node_size=3800,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_edges(
        G, POS, edgelist=confounder_edges, ax=ax,
        edge_color=ROLE_COLORS["confounder"], arrows=True,
        arrowstyle="-|>", arrowsize=22,
        width=2.2, node_size=3800, style="dashed",
        connectionstyle="arc3,rad=0.05",
    )

    nx.draw_networkx_nodes(
        G, POS, ax=ax,
        node_color=node_colors, node_size=3800, linewidths=1.8,
        edgecolors="#333333",
    )
    nx.draw_networkx_labels(G, POS, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=ROLE_LABELS[role])
        for role, color in ROLE_COLORS.items()
    ]
    legend_handles.append(
        mpatches.Patch(color="white", label="Dashed = confounder edges",
                       linestyle="dashed", linewidth=1.5,
                       edgecolor=ROLE_COLORS["confounder"])
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_title(
        "Ground Truth Causal DAG — Depression\n"
        "8 edges · 1 confounder · 6 variables",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("ground_truth_dag.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → ground_truth_dag.png")


if __name__ == "__main__":
    main()

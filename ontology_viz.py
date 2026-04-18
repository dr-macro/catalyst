"""
Visualise daily ontologies and their diffs.

Produces two charts (both saved under summaries/):
  1. ontology_diff_<date>.png   — side-by-side network graph (prev vs curr),
     change-coded by colour.
  2. ontology_themes_<date>.png — grouped bar chart of theme prominence,
     prev vs curr.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

CHARTS_DIR = Path("summaries")

# Node subtype → base colour (dark-background palette)
_SUBTYPE_COLOR = {
    "Institution": "#4A90D9",
    "Country":     "#7ED321",
    "Person":      "#F5A623",
    "Asset":       "#9B59B6",
    "Policy":      "#E74C3C",
    "Indicator":   "#1ABC9C",
    "Company":     "#3498DB",
}
_DEFAULT_COLOR = "#95A5A6"

# Diff-state → highlight colour
_CHANGE_COLOR = {
    "new":       "#27AE60",
    "removed":   "#C0392B",
    "increased": "#2ECC71",
    "decreased": "#E67E22",
    "unchanged": "#BDC3C7",
}


def _node_color(node: dict, state: str = "unchanged") -> str:
    if state != "unchanged":
        return _CHANGE_COLOR.get(state, _CHANGE_COLOR["unchanged"])
    return _SUBTYPE_COLOR.get(node.get("subtype", ""), _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Network graph
# ---------------------------------------------------------------------------

def build_ontology_diff_chart(prev: dict, curr: dict, diff: dict) -> Path | None:
    """
    Two-panel network graph: left = previous day, right = current day.
    Current-day panel highlights new/removed/changed nodes in diff colours.
    Returns path to saved PNG, or None on import error.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("  Warning: matplotlib/networkx missing — skipping ontology network chart.")
        return None

    CHARTS_DIR.mkdir(exist_ok=True)
    date_str = curr.get("date", datetime.today().strftime("%Y-%m-%d"))
    out_path = CHARTS_DIR / f"ontology_diff_{date_str}.png"

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#0D1117")

    new_ids = {n["id"] for n in diff.get("new_nodes", [])}
    removed_ids = {n["id"] for n in diff.get("removed_nodes", [])}
    changed_ids = {
        n["id"]: n["prominence_delta"] for n in diff.get("changed_nodes", [])
    }

    def _draw(ax: "plt.Axes", ontology: dict, title: str,
              mark_new: set | None = None,
              mark_removed: set | None = None,
              mark_changed: dict | None = None) -> None:
        ax.set_facecolor("#0D1117")
        mark_new = mark_new or set()
        mark_removed = mark_removed or set()
        mark_changed = mark_changed or {}

        G: "nx.DiGraph" = nx.DiGraph()
        nodes = ontology.get("nodes", [])
        edges = ontology.get("edges", [])
        node_ids = {n["id"] for n in nodes}

        for n in nodes:
            G.add_node(n["id"], **n)
        for e in edges:
            if e.get("from") in node_ids and e.get("to") in node_ids:
                G.add_edge(
                    e["from"], e["to"],
                    label=e.get("label", ""),
                    weight=e.get("weight", 0.5),
                )

        if not G.nodes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="white", fontsize=14, transform=ax.transAxes)
            ax.set_title(title, color="white", fontsize=12, pad=8)
            return

        pos = nx.spring_layout(G, k=2.2, seed=42, weight="weight")

        node_list = list(G.nodes)
        sizes, colors = [], []
        for nid in node_list:
            node = next((n for n in nodes if n["id"] == nid), {})
            sizes.append(300 + node.get("prominence", 0.5) * 1000)
            if nid in mark_new:
                colors.append(_CHANGE_COLOR["new"])
            elif nid in mark_removed:
                colors.append(_CHANGE_COLOR["removed"])
            elif nid in mark_changed:
                delta = mark_changed[nid]
                colors.append(_CHANGE_COLOR["increased"] if delta > 0
                              else _CHANGE_COLOR["decreased"])
            else:
                colors.append(_node_color(node))

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#4A4A6A", alpha=0.45, arrows=True,
            arrowsize=10, width=0.7,
            connectionstyle="arc3,rad=0.1",
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=node_list, ax=ax,
            node_size=sizes, node_color=colors, alpha=0.88,
        )

        labels = {
            nid: (
                (lbl := next((n.get("label", nid) for n in nodes if n["id"] == nid), nid))
                and (lbl[:18] + "…" if len(lbl) > 18 else lbl)
            )
            for nid in node_list
            if next((n for n in nodes if n["id"] == nid), {}).get("prominence", 0) >= 0.35
        }
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                font_size=7, font_color="white")
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")

    _draw(axes[0], prev, f"Ontology  {prev.get('date', '?')}")
    _draw(
        axes[1], curr,
        f"Ontology  {curr.get('date', '?')}  (changes highlighted)",
        mark_new=new_ids,
        mark_removed=removed_ids,
        mark_changed=changed_ids,
    )

    legend = [
        mpatches.Patch(color=_CHANGE_COLOR["new"],       label="New entity"),
        mpatches.Patch(color=_CHANGE_COLOR["removed"],   label="Dropped entity"),
        mpatches.Patch(color=_CHANGE_COLOR["increased"], label="Prominence ↑"),
        mpatches.Patch(color=_CHANGE_COLOR["decreased"], label="Prominence ↓"),
        mpatches.Patch(color=_SUBTYPE_COLOR["Institution"], label="Institution"),
        mpatches.Patch(color=_SUBTYPE_COLOR["Country"],     label="Country"),
        mpatches.Patch(color=_SUBTYPE_COLOR["Asset"],       label="Asset"),
        mpatches.Patch(color=_SUBTYPE_COLOR["Person"],      label="Person"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4,
        facecolor="#1C2128", edgecolor="#555", labelcolor="white",
        fontsize=9, framealpha=0.9,
    )
    fig.suptitle(
        f"Narrative Ontology Shift:  {prev.get('date', '?')}  →  {curr.get('date', '?')}",
        color="white", fontsize=14, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Ontology diff chart: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Theme drift bar chart
# ---------------------------------------------------------------------------

def build_theme_drift_chart(prev: dict, curr: dict, diff: dict) -> Path | None:
    """
    Grouped bar chart: each theme as a pair of bars (prev height, curr height).
    Bars colour-coded by change state.
    Returns path to saved PNG, or None on import error / no data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  Warning: matplotlib/numpy missing — skipping theme drift chart.")
        return None

    curr_themes = curr.get("themes", [])
    if not curr_themes:
        return None

    CHARTS_DIR.mkdir(exist_ok=True)
    date_str = curr.get("date", datetime.today().strftime("%Y-%m-%d"))
    out_path = CHARTS_DIR / f"ontology_themes_{date_str}.png"

    prev_prom = {t["label"]: t.get("prominence", 0) for t in prev.get("themes", [])}
    curr_prom = {t["label"]: t.get("prominence", 0) for t in curr_themes}

    new_labels    = {t["label"] for t in diff.get("new_themes", [])}
    removed_labels = {t["label"] for t in diff.get("removed_themes", [])}
    changed_labels = {t["label"] for t in diff.get("changed_themes", [])}

    all_labels = sorted(
        set(curr_prom) | set(prev_prom),
        key=lambda l: -max(curr_prom.get(l, 0), prev_prom.get(l, 0)),
    )[:15]

    x = np.arange(len(all_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    bar_colors = []
    for lbl in all_labels:
        if lbl in new_labels:
            bar_colors.append(_CHANGE_COLOR["new"])
        elif lbl in removed_labels:
            bar_colors.append(_CHANGE_COLOR["removed"])
        elif lbl in changed_labels:
            delta = curr_prom.get(lbl, 0) - prev_prom.get(lbl, 0)
            bar_colors.append(
                _CHANGE_COLOR["increased"] if delta > 0 else _CHANGE_COLOR["decreased"]
            )
        else:
            bar_colors.append("#4A90D9")

    ax.bar(x - w / 2,
           [prev_prom.get(l, 0) for l in all_labels],
           w, label=f"Prev ({prev.get('date', '?')})",
           color="#4A4A6A", alpha=0.75)
    ax.bar(x + w / 2,
           [curr_prom.get(l, 0) for l in all_labels],
           w, label=f"Curr ({curr.get('date', '?')})",
           color=bar_colors, alpha=0.90)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=38, ha="right",
                       fontsize=8.5, color="white")
    ax.set_ylabel("Prominence", color="white")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Narrative Theme Shift:  {prev.get('date', '?')}  →  {curr.get('date', '?')}",
        color="white", fontsize=13,
    )
    ax.legend(facecolor="#1C2128", labelcolor="white", edgecolor="#555")
    ax.tick_params(colors="white")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Theme drift chart: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Convenience: find today's pre-generated charts
# ---------------------------------------------------------------------------

def get_chart_paths(date_str: str | None = None) -> dict[str, Path | None]:
    """Return paths for existing diff/theme charts for a given date."""
    if date_str is None:
        date_str = datetime.today().strftime("%Y-%m-%d")
    diff_path  = CHARTS_DIR / f"ontology_diff_{date_str}.png"
    theme_path = CHARTS_DIR / f"ontology_themes_{date_str}.png"
    return {
        "diff":  diff_path  if diff_path.exists()  else None,
        "theme": theme_path if theme_path.exists() else None,
    }

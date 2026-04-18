"""
Compute the structural diff between two consecutive daily ontologies.

Matches nodes and themes by normalised label (case-insensitive) so that
"Federal Reserve" == "federal reserve" == "fed reserve" won't create spurious
new/removed pairs — the merge step in ontology_builder should already
handle fuzzy deduplication, but label normalisation here gives a second safety net.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _by_label(items: list[dict]) -> dict[str, dict]:
    return {item.get("label", "").lower().strip(): item for item in items}


def _edges_by_key(ontology: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for e in ontology.get("edges", []):
        key = f"{e.get('from')}::{e.get('relation')}::{e.get('to')}"
        result[key] = e
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

PROMINENCE_CHANGE_THRESHOLD = 0.10  # minimum delta to count as "changed"


def compute_diff(prev: dict, curr: dict) -> dict:
    """
    Compare two ontologies.  Returns a diff dict containing:
      new_nodes, removed_nodes, changed_nodes  (prominence delta ≥ threshold)
      new_edges, removed_edges
      new_themes, removed_themes, changed_themes
      stats  — summary counts
    """
    prev_nodes = _by_label(prev.get("nodes", []))
    curr_nodes = _by_label(curr.get("nodes", []))
    prev_edges = _edges_by_key(prev)
    curr_edges = _edges_by_key(curr)
    prev_themes = _by_label(prev.get("themes", []))
    curr_themes = _by_label(curr.get("themes", []))

    # --- nodes ---
    new_node_keys = set(curr_nodes) - set(prev_nodes)
    removed_node_keys = set(prev_nodes) - set(curr_nodes)
    shared_node_keys = set(prev_nodes) & set(curr_nodes)

    new_nodes = sorted(
        [curr_nodes[k] for k in new_node_keys],
        key=lambda n: -n.get("prominence", 0),
    )
    removed_nodes = sorted(
        [prev_nodes[k] for k in removed_node_keys],
        key=lambda n: -n.get("prominence", 0),
    )
    changed_nodes = []
    for k in shared_node_keys:
        delta = curr_nodes[k].get("prominence", 0) - prev_nodes[k].get("prominence", 0)
        if abs(delta) >= PROMINENCE_CHANGE_THRESHOLD:
            changed_nodes.append({
                **curr_nodes[k],
                "prominence_delta": round(delta, 3),
                "prev_prominence": round(prev_nodes[k].get("prominence", 0), 3),
            })
    changed_nodes.sort(key=lambda n: -abs(n["prominence_delta"]))

    # --- edges ---
    new_edges = [curr_edges[k] for k in sorted(set(curr_edges) - set(prev_edges))]
    removed_edges = [prev_edges[k] for k in sorted(set(prev_edges) - set(curr_edges))]

    # --- themes ---
    new_theme_keys = set(curr_themes) - set(prev_themes)
    removed_theme_keys = set(prev_themes) - set(curr_themes)
    shared_theme_keys = set(prev_themes) & set(curr_themes)

    new_themes = sorted(
        [curr_themes[k] for k in new_theme_keys],
        key=lambda t: -t.get("prominence", 0),
    )
    removed_themes = sorted(
        [prev_themes[k] for k in removed_theme_keys],
        key=lambda t: -t.get("prominence", 0),
    )
    changed_themes = []
    for k in shared_theme_keys:
        delta = curr_themes[k].get("prominence", 0) - prev_themes[k].get("prominence", 0)
        if abs(delta) >= PROMINENCE_CHANGE_THRESHOLD:
            changed_themes.append({
                **curr_themes[k],
                "prominence_delta": round(delta, 3),
                "prev_prominence": round(prev_themes[k].get("prominence", 0), 3),
            })
    changed_themes.sort(key=lambda t: -abs(t["prominence_delta"]))

    return {
        "prev_date": prev.get("date"),
        "curr_date": curr.get("date"),
        "new_nodes": new_nodes,
        "removed_nodes": removed_nodes,
        "changed_nodes": changed_nodes,
        "new_edges": new_edges,
        "removed_edges": removed_edges,
        "new_themes": new_themes,
        "removed_themes": removed_themes,
        "changed_themes": changed_themes,
        "stats": {
            "nodes_added": len(new_nodes),
            "nodes_removed": len(removed_nodes),
            "nodes_changed": len(changed_nodes),
            "edges_added": len(new_edges),
            "edges_removed": len(removed_edges),
            "themes_added": len(new_themes),
            "themes_removed": len(removed_themes),
            "themes_changed": len(changed_themes),
        },
    }


def format_diff_text(diff: dict) -> str:
    """Plain-text summary of the diff for console output / email fallback."""
    s = diff["stats"]
    lines = [
        f"Ontology shift: {diff['prev_date']} → {diff['curr_date']}",
        "=" * 52,
        f"Nodes  : +{s['nodes_added']} new  -{s['nodes_removed']} removed  "
        f"~{s['nodes_changed']} prominence change",
        f"Edges  : +{s['edges_added']} new  -{s['edges_removed']} removed",
        f"Themes : +{s['themes_added']} new  -{s['themes_removed']} removed  "
        f"~{s['themes_changed']} prominence change",
    ]

    def _block(title: str, items: list[dict], prefix: str, key: str = "label",
               extra: str | None = None) -> None:
        if not items:
            return
        lines.append(f"\n{title}")
        for item in items[:12]:
            label = item.get(key, "?")
            prom = item.get("prominence", 0)
            detail = f"prominence={prom:.2f}"
            if extra == "delta":
                delta = item.get("prominence_delta", 0)
                arrow = "↑" if delta > 0 else "↓"
                detail = f"{item.get('prev_prominence', 0):.2f} → {prom:.2f} ({arrow}{abs(delta):.2f})"
            lines.append(f"  {prefix} {label}  [{detail}]")
            if item.get("description"):
                lines.append(f"      {item['description']}")

    _block("EMERGING THEMES", diff["new_themes"], "+")
    _block("FADING THEMES", diff["removed_themes"], "-")
    _block("SHIFTING THEMES", diff["changed_themes"], "~", extra="delta")
    _block("NEW ENTITIES", diff["new_nodes"], "+")
    _block("DROPPED ENTITIES", diff["removed_nodes"], "-")

    return "\n".join(lines)

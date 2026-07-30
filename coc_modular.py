"""
Modular graph-of-graphs merge for core-of-cores (v2).

Assembles ``n`` independent Vester sub-cores into:

    G = (bigsqcup_i G_i)  ∪  E_cross  ∪  E_concept

Each sub-core keeps its internal adjacency ``A_i``; sparse off-diagonal blocks
``B_ij`` hold variable-level cross-core bridges. Module-to-module edges are
*derived* from those bridges (never LLM-invented independently).

Local Vester roles are preserved. External / global roles and a bridge score
are computed separately so dense local matrices do not dominate.
"""

from __future__ import annotations

import math
import re
import textwrap
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Edge / node taxonomies
# --------------------------------------------------------------------------- #
CAUSAL_EDGE_KINDS = {"intra_causal", "cross_causal", "catalyst_affects"}
STRUCTURAL_EDGE_KINDS = {"contains", "instance_of"}
DERIVED_EDGE_KINDS = {"derived_summary"}

ROLE_COLOR = {
    "active (lever)":    "#2ca02c",
    "critical (risk)":   "#d62728",
    "reactive (sensor)": "#17becf",
    "buffering (inert)": "#cccccc",
    "active (bridge)":   "#ff7f0e",
}
SIGN_STYLE = {
    "+": dict(color="#1a9850", style="solid"),
    "-": dict(color="#d73027", style="solid"),
    "?": dict(color="#8a8a8a", style="dashed"),
}


def slugify(s: str, max_len: int = 32) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (s or "x").lower()).strip("_")
    return (s[:max_len] or "x")


def wrap_label(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text or "", width=width)) or (text or "")


# --------------------------------------------------------------------------- #
# Graph helpers (DiGraph + MultiDiGraph)
# --------------------------------------------------------------------------- #
def iter_edges_with_keys(G):
    if G.is_multigraph():
        yield from G.edges(keys=True, data=True)
    else:
        for u, v, d in G.edges(data=True):
            yield u, v, None, d


def edge_projection(G, allowed_kinds: Iterable[str]) -> nx.DiGraph:
    allowed = set(allowed_kinds)
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    for u, v, _k, d in iter_edges_with_keys(G):
        if d.get("edge_kind") not in allowed:
            continue
        w = float(d.get("strength", 1) or 1)
        if H.has_edge(u, v):
            H[u][v]["strength"] = H[u][v].get("strength", 1) + w
        else:
            nd = dict(d)
            nd["strength"] = w
            H.add_edge(u, v, **nd)
    return H


def _scalar(val):
    import json
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (dict, list, set, tuple)):
        return json.dumps(list(val) if isinstance(val, (set, tuple)) else val,
                          ensure_ascii=False)
    return "" if val is None else str(val)


def gephi_safe_multigraph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **{k: _scalar(v) for k, v in d.items()})
    for u, v, k, d in G.edges(keys=True, data=True):
        H.add_edge(u, v, key=k, **{kk: _scalar(vv) for kk, vv in d.items()})
    return H


# --------------------------------------------------------------------------- #
# 1. Assemble modular MultiDiGraph from sub_cores
# --------------------------------------------------------------------------- #
def assemble_modular_graph(sub_cores: dict) -> nx.MultiDiGraph:
    """Namespace local variables, add module nodes + containment + intra edges
    + catalysts. Preserves local AS/PS/role. Does NOT invent cross-core edges."""
    CoC = nx.MultiDiGraph()
    for core_id, sub in sub_cores.items():
        sidx = {vid: i for i, vid in enumerate(sub.var_ids)}
        CoC.add_node(
            f"core::{core_id}",
            label=getattr(sub, "narrative_title", None) or core_id,
            node_kind="core",
            parent_core=core_id,
            n_vars=sub.N,
            n_catalysts=len(sub.catalysts),
        )
        for v in sub.variables:
            vid = v["id"]
            nid = f"{core_id}::{vid}"
            i = sidx[vid]
            AS = int(sub.AS[i]); PS = int(sub.PS[i])
            role = sub.roles.loc[vid, "role"]
            CoC.add_node(
                nid,
                label=v.get("name", vid),
                node_kind="variable",
                parent_core=core_id,
                local_id=vid,
                local_role=role,
                local_AS=AS,
                local_PS=PS,
                ontology_type=v.get("ontology_type", "state"),
                description=v.get("description", ""),
                concept_id=None,
                AS=AS, PS=PS, P=float(AS * PS),
                role=role,
            )
            CoC.add_edge(f"core::{core_id}", nid, edge_kind="contains")
        for u, w, d in sub.graph.edges(data=True):
            CoC.add_edge(
                f"{core_id}::{u}", f"{core_id}::{w}",
                edge_kind="intra_causal",
                strength=int(d.get("strength", 1)),
                sign=d.get("sign", "?"),
            )
        score_by_id = (
            dict(zip(sub.catalysts_df["id"], sub.catalysts_df["impact_score"]))
            if not sub.catalysts_df.empty else {}
        )
        for c in sub.catalysts:
            cid = f"{core_id}::cat::{c['id']}"
            CoC.add_node(
                cid,
                label=c["title"],
                node_kind="catalyst",
                parent_core=core_id,
                impact=float(score_by_id.get(c["id"], 0.0)),
                probability=float(c.get("probability", 0.0)),
                timeline=(c.get("timeline") or {}).get("label", ""),
                P=0.0,
            )
            for a in c.get("affected", []):
                tgt = f"{core_id}::{a['variable_id']}"
                if tgt in CoC:
                    CoC.add_edge(
                        cid, tgt,
                        edge_kind="catalyst_affects",
                        strength=int(a.get("intensity", 1)),
                        sign=a.get("sign", "?"),
                    )
    return CoC


# --------------------------------------------------------------------------- #
# 2. Interface-variable selection
# --------------------------------------------------------------------------- #
def interface_prompt(core_id: str, core_label: str, variables: Sequence[dict],
                     k: int = 5) -> str:
    block = "\n".join(
        f"- {v['id']}: {v['name']} [{v.get('ontology_type','')}] "
        f"role={v.get('local_role','')} AS={v.get('local_AS',0)} PS={v.get('local_PS',0)}"
        + (f" — {v.get('description','')}" if v.get("description") else "")
        for v in variables
    )
    return f"""You are selecting INTERFACE VARIABLES for a geopolitical sub-core
in a modular graph-of-graphs model.

SUB-CORE: {core_id} ({core_label})

An interface variable is one that plausibly:
- transmits effects outside its own sub-core; and/or
- receives effects from other sub-cores; and/or
- corresponds to a shared geopolitical resource, constraint, or outcome
  (energy prices, sanctions, shipping, defence capacity, domestic support, etc.).

Select between 3 and {k} interface variables from the list below.
Prefer high local_AS or high local_PS when tied; prefer concrete measurable
pressures / capacities over vague themes.

Return strict JSON:
{{ "interface_ids": ["<id>", "..."], "rationale": "<one sentence>" }}

VARIABLES:
{block}
"""


def heuristic_interfaces(CoC: nx.MultiDiGraph, core_id: str, k: int = 5) -> list[str]:
    """Fallback: top-k by P=AS*PS within the sub-core."""
    vars_ = [
        (n, d) for n, d in CoC.nodes(data=True)
        if d.get("node_kind") == "variable" and d.get("parent_core") == core_id
    ]
    vars_.sort(key=lambda x: float(x[1].get("P", 0) or 0), reverse=True)
    return [n.split("::", 1)[1] for n, _ in vars_[:k]]


# --------------------------------------------------------------------------- #
# 3. Sparse pairwise bridge prompts
# --------------------------------------------------------------------------- #
def bridge_prompt(src_core: str, tgt_core: str,
                  src_vars: Sequence[dict], tgt_vars: Sequence[dict],
                  max_bridges: int = 4) -> str:
    def _block(vs):
        return "\n".join(
            f"- {v['id']}: {v['name']}"
            + (f" — {v.get('description','')}" if v.get("description") else "")
            for v in vs
        )
    return f"""Identify direct causal effects from variables in source core A
to variables in target core B.

SOURCE CORE A: {src_core}
TARGET CORE B: {tgt_core}

Return at most {max_bridges} bridges.

Do not count:
- thematic similarity;
- synonyms;
- shared causes;
- containment;
- effects mediated entirely through another listed variable;
- vague long-run possibilities.

A bridge must specify a concrete mechanism and must represent a change in the
source variable causing a change in the target variable within the stated
time horizon.

For each bridge return:
- source_variable_id  (from SOURCE list)
- target_variable_id  (from TARGET list)
- sign: "+" or "-"
- strength: 1, 2, or 3
- confidence: 0 to 1
- mechanism: one sentence
- horizon: e.g. "0-6 months" or "6-18 months"

Return strict JSON:
{{ "bridges": [ {{...}}, ... ] }}

SOURCE VARIABLES (interfaces of A):
{_block(src_vars)}

TARGET VARIABLES (interfaces of B):
{_block(tgt_vars)}
"""


def add_cross_causal_bridges(CoC: nx.MultiDiGraph, bridges: Sequence[dict],
                             *, min_confidence: float = 0.55) -> int:
    """Add validated cross_causal edges. Returns count added."""
    n = 0
    for b in bridges:
        src_core = b.get("source_core") or b.get("src_core")
        tgt_core = b.get("target_core") or b.get("tgt_core")
        s = str(b.get("source_variable") or b.get("source_variable_id") or "")
        t = str(b.get("target_variable") or b.get("target_variable_id") or "")
        if not (src_core and tgt_core and s and t):
            continue
        conf = float(b.get("confidence", 0.5) or 0.5)
        if conf < min_confidence:
            continue
        src, tgt = f"{src_core}::{s}", f"{tgt_core}::{t}"
        if src not in CoC or tgt not in CoC:
            continue
        if CoC.nodes[src].get("parent_core") != src_core:
            continue
        if CoC.nodes[tgt].get("parent_core") != tgt_core:
            continue
        CoC.add_edge(
            src, tgt,
            edge_kind="cross_causal",
            sign=b.get("sign", "?"),
            strength=int(b.get("strength", 1) or 1),
            confidence=conf,
            mechanism=b.get("mechanism", ""),
            horizon=b.get("horizon", ""),
            source_core=src_core,
            target_core=tgt_core,
        )
        n += 1
    return n


# --------------------------------------------------------------------------- #
# 4. Shared concepts (embedding clustering) — reuse union-find pattern
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_concepts(var_records: Sequence[dict], embeddings: np.ndarray, *,
                     sim_threshold: float = 0.82, min_subcores: int = 2) -> list[dict]:
    n = len(var_records)
    if n == 0 or embeddings is None or len(embeddings) != n:
        return []
    sims = embeddings @ embeddings.T
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= sim_threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    concepts: list[dict] = []
    used: set[str] = set()
    for members in groups.values():
        parents = {var_records[i]["parent_core"] for i in members}
        if len(parents) < min_subcores:
            continue
        names = [var_records[i]["name"] for i in members]
        label = min(names, key=len)
        cid = slugify(label, max_len=28)
        base, k = cid, 2
        while cid in used:
            cid = f"{base}_{k}"
            k += 1
        used.add(cid)
        concepts.append({
            "concept_id": cid,
            "label": label,
            "members": [var_records[i]["node_id"] for i in members],
            "parents": parents,
        })
    return concepts


def attach_concepts(CoC: nx.MultiDiGraph, concepts: Sequence[dict]) -> int:
    for c in concepts:
        cnode = f"concept::{c['concept_id']}"
        CoC.add_node(
            cnode,
            label=c["label"],
            node_kind="shared_concept",
            parent_core=None,
            members=len(c["members"]),
            P=0.0,
        )
        for m in c["members"]:
            if m in CoC:
                CoC.add_edge(m, cnode, edge_kind="instance_of")
                CoC.nodes[m]["concept_id"] = c["concept_id"]
    return len(concepts)


# --------------------------------------------------------------------------- #
# 5. Derive module-to-module quotient graph from bridges
# --------------------------------------------------------------------------- #
def derive_module_summary_edges(CoC: nx.MultiDiGraph, *,
                                write_to_coc: bool = True) -> nx.DiGraph:
    """Quotient graph: W_ij = sum(s*c) / (|Vi|*|Vj|), plus signed aggregates.

    Marks edges as ``derived_summary`` so they are never treated as primary
    causal evidence alongside the variable-level bridges.
    """
    core_ids = sorted({
        d["parent_core"] for _, d in CoC.nodes(data=True)
        if d.get("node_kind") == "variable" and d.get("parent_core")
    })
    sizes = {
        cid: sum(1 for _, d in CoC.nodes(data=True)
                 if d.get("node_kind") == "variable" and d.get("parent_core") == cid)
        for cid in core_ids
    }
    # accumulate signed sums
    pos: dict[tuple[str, str], float] = {}
    neg: dict[tuple[str, str], float] = {}
    for u, v, _k, d in iter_edges_with_keys(CoC):
        if d.get("edge_kind") != "cross_causal":
            continue
        si = CoC.nodes[u].get("parent_core")
        sj = CoC.nodes[v].get("parent_core")
        if not si or not sj or si == sj:
            continue
        s = float(d.get("strength", 1) or 1)
        c = float(d.get("confidence", 0.5) or 0.5)
        w = s * c
        sign = d.get("sign", "+")
        key = (si, sj)
        if sign == "-":
            neg[key] = neg.get(key, 0.0) + w
        else:
            pos[key] = pos.get(key, 0.0) + w

    Q = nx.DiGraph()
    for cid in core_ids:
        Q.add_node(f"core::{cid}", **dict(CoC.nodes[f"core::{cid}"]))
    for (si, sj), wp in {**{k: 0.0 for k in neg}, **pos}.items():
        wn = neg.get((si, sj), 0.0)
        wp = pos.get((si, sj), 0.0)
        denom = max(sizes.get(si, 1) * sizes.get(sj, 1), 1)
        W = (wp + wn) / denom
        W_pos = wp / denom
        W_neg = wn / denom
        if W <= 0 and W_pos <= 0 and W_neg <= 0:
            continue
        net_sign = "+" if wp >= wn else "-"
        attrs = dict(
            edge_kind="derived_summary",
            strength=max(1, min(3, int(round(3 * W / max(W, 1e-9))) if W > 0 else 1)),
            W=float(W), W_pos=float(W_pos), W_neg=float(W_neg),
            weight=float(W),
            sign=net_sign,
            n_bridges=sum(
                1 for u, v, _k, d in iter_edges_with_keys(CoC)
                if d.get("edge_kind") == "cross_causal"
                and CoC.nodes[u].get("parent_core") == si
                and CoC.nodes[v].get("parent_core") == sj
            ),
        )
        # strength from total bridge count for visual weight
        attrs["strength"] = max(1, min(3, attrs["n_bridges"]))
        Q.add_edge(f"core::{si}", f"core::{sj}", **attrs)
        if write_to_coc and f"core::{si}" in CoC and f"core::{sj}" in CoC:
            CoC.add_edge(f"core::{si}", f"core::{sj}", **attrs)
    return Q


# --------------------------------------------------------------------------- #
# 6. Local / external / global roles + bridge score
# --------------------------------------------------------------------------- #
def _classify(as_: float, ps_: float, med_AS: float, med_PS: float) -> str:
    if as_ >= med_AS and ps_ < med_PS:
        return "active (lever)"
    if as_ < med_AS and ps_ >= med_PS:
        return "reactive (sensor)"
    if as_ >= med_AS and ps_ >= med_PS:
        return "critical (risk)"
    return "buffering (inert)"


def compute_external_scores(CoC: nx.MultiDiGraph) -> dict[str, dict]:
    """AS_ext / PS_ext from cross_causal edges only (weight = strength * confidence)."""
    scores: dict[str, dict] = {}
    for n, d in CoC.nodes(data=True):
        if d.get("node_kind") != "variable":
            continue
        scores[n] = {"AS_ext": 0.0, "PS_ext": 0.0, "k_cores": set()}
    for u, v, _k, d in iter_edges_with_keys(CoC):
        if d.get("edge_kind") != "cross_causal":
            continue
        w = float(d.get("strength", 1) or 1) * float(d.get("confidence", 0.5) or 0.5)
        if u in scores:
            scores[u]["AS_ext"] += w
            pc = CoC.nodes[v].get("parent_core")
            if pc:
                scores[u]["k_cores"].add(pc)
        if v in scores:
            scores[v]["PS_ext"] += w
            pc = CoC.nodes[u].get("parent_core")
            if pc:
                scores[v]["k_cores"].add(pc)
    return scores


def apply_global_roles(CoC: nx.MultiDiGraph, *, alpha: float = 0.6) -> pd.DataFrame:
    """Write AS_ext/PS_ext/global_AS/global_PS/global_role/bridge_score onto nodes;
    return a tidy roles DataFrame."""
    ext = compute_external_scores(CoC)
    rows = []
    for n, d in CoC.nodes(data=True):
        if d.get("node_kind") != "variable":
            continue
        e = ext.get(n, {"AS_ext": 0.0, "PS_ext": 0.0, "k_cores": set()})
        AS_loc = float(d.get("local_AS", 0) or 0)
        PS_loc = float(d.get("local_PS", 0) or 0)
        AS_ext = float(e["AS_ext"])
        PS_ext = float(e["PS_ext"])
        AS_g = alpha * AS_loc + (1 - alpha) * AS_ext
        PS_g = alpha * PS_loc + (1 - alpha) * PS_ext
        k = len(e["k_cores"])
        B = (AS_ext + PS_ext) * math.log(1 + k)
        # stash
        d["AS_ext"] = AS_ext
        d["PS_ext"] = PS_ext
        d["global_AS"] = AS_g
        d["global_PS"] = PS_g
        d["bridge_score"] = B
        d["n_bridged_cores"] = k
        rows.append({
            "node": n,
            "parent_core": d.get("parent_core"),
            "label": d.get("label"),
            "local_AS": AS_loc,
            "local_PS": PS_loc,
            "local_role": d.get("local_role"),
            "AS_ext": AS_ext,
            "PS_ext": PS_ext,
            "global_AS": AS_g,
            "global_PS": PS_g,
            "bridge_score": B,
            "n_bridged_cores": k,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    med_AS = float(df["global_AS"].median())
    med_PS = float(df["global_PS"].median())
    df["global_role"] = [
        _classify(a, p, med_AS, med_PS)
        for a, p in zip(df["global_AS"], df["global_PS"])
    ]
    for _, r in df.iterrows():
        CoC.nodes[r["node"]]["global_role"] = r["global_role"]
        # keep display role = local by default; global available separately
        CoC.nodes[r["node"]]["role"] = r["local_role"]
    return df.sort_values("bridge_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Layouts for modular views
# --------------------------------------------------------------------------- #
def quotient_layout(Q: nx.DiGraph, *, seed: int = 42, canvas: float = 8.0) -> dict:
    if Q.number_of_nodes() == 0:
        return {}
    if Q.number_of_edges() == 0:
        # circle for isolated modules
        nodes = list(Q.nodes())
        n = len(nodes)
        return {
            nd: (canvas * math.cos(2 * math.pi * i / n),
                 canvas * math.sin(2 * math.pi * i / n))
            for i, nd in enumerate(nodes)
        }
    raw = nx.spring_layout(Q, seed=seed, weight="weight", k=2.2, iterations=300)
    return {n: (float(canvas * p[0]), float(canvas * p[1])) for n, p in raw.items()}


def interface_map_layout(CoC: nx.MultiDiGraph, interface_ids: dict[str, list[str]],
                         Q_pos: dict, *, cluster_offset: float = 3.5,
                         seed: int = 42) -> dict:
    """Place modules via quotient layout; park interface vars + bridge endpoints
    in outward clusters around each module."""
    pos = dict(Q_pos)
    if not Q_pos:
        return pos
    centre = np.mean(np.asarray(list(Q_pos.values()), dtype=float), axis=0)
    for core_id, local_ids in interface_ids.items():
        parent = f"core::{core_id}"
        if parent not in Q_pos:
            continue
        p = np.asarray(Q_pos[parent], dtype=float)
        outward = p - centre
        nrm = float(np.linalg.norm(outward))
        outward = np.array([1.0, 0.0]) if nrm < 1e-9 else outward / nrm
        bridge_nodes = set()
        for u, v, _k, d in iter_edges_with_keys(CoC):
            if d.get("edge_kind") != "cross_causal":
                continue
            for e in (u, v):
                if CoC.nodes[e].get("parent_core") == core_id:
                    bridge_nodes.add(e)
        nodes = [f"{core_id}::{lid}" for lid in local_ids if f"{core_id}::{lid}" in CoC]
        nodes = list(dict.fromkeys(nodes + list(bridge_nodes)))  # unique, preserve order
        if not nodes:
            continue
        L = nx.DiGraph()
        L.add_nodes_from(nodes)
        for u, v, _k, d in iter_edges_with_keys(CoC):
            if u in L and v in L and d.get("edge_kind") in ("intra_causal", "cross_causal"):
                L.add_edge(u, v)
        raw = nx.spring_layout(L, seed=(seed + abs(hash(core_id))) % (2 ** 32), iterations=150)
        arr = np.asarray([raw[n] for n in nodes], dtype=float)
        arr -= arr.mean(axis=0)
        mx = max(float(np.linalg.norm(arr, axis=1).max()), 1e-9)
        radius = 1.5 + 0.3 * math.sqrt(len(nodes))
        arr = radius * arr / mx
        cluster = p + outward * cluster_offset
        for i, n in enumerate(nodes):
            pos[n] = (float(cluster[0] + arr[i, 0]), float(cluster[1] + arr[i, 1]))
    # concepts at barycentre of members
    for n, d in CoC.nodes(data=True):
        if d.get("node_kind") != "shared_concept":
            continue
        members = [u for u, v, _k, ed in iter_edges_with_keys(CoC)
                   if v == n and ed.get("edge_kind") == "instance_of" and u in pos]
        if members:
            pos[n] = tuple(np.mean([pos[m] for m in members], axis=0) + np.array([0.0, 1.0]))
    return pos


def full_modular_layout(CoC: nx.MultiDiGraph, Q_pos: dict, *,
                        cluster_offset: float = 5.0, base_radius: float = 2.2,
                        seed: int = 42) -> dict:
    """Place *every* node: modules from ``Q_pos``, then each sub-core's full
    variable set (and catalysts) in an outward cluster. Concepts at member
    barycentres. Unlike ``interface_map_layout``, nothing is dropped."""
    pos = dict(Q_pos)
    if not Q_pos:
        # fall back: spring on cores alone
        cores = [n for n, d in CoC.nodes(data=True) if d.get("node_kind") == "core"]
        if not cores:
            return {}
        raw = nx.spring_layout(nx.DiGraph(cores), seed=seed, iterations=100)
        pos = {n: (float(8 * p[0]), float(8 * p[1])) for n, p in raw.items()}
        Q_pos = pos

    centre = np.mean(np.asarray(list(Q_pos.values()), dtype=float), axis=0)
    core_ids = sorted({
        d["parent_core"] for _, d in CoC.nodes(data=True)
        if d.get("node_kind") == "variable" and d.get("parent_core")
    })

    for core_id in core_ids:
        parent = f"core::{core_id}"
        if parent not in pos:
            continue
        p = np.asarray(pos[parent], dtype=float)
        outward = p - centre
        nrm = float(np.linalg.norm(outward))
        outward = np.array([1.0, 0.0]) if nrm < 1e-9 else outward / nrm

        variables = [
            n for n, d in CoC.nodes(data=True)
            if d.get("node_kind") == "variable" and d.get("parent_core") == core_id
        ]
        if not variables:
            continue
        L = nx.DiGraph()
        L.add_nodes_from(variables)
        for u, v, _k, d in iter_edges_with_keys(CoC):
            if u in L and v in L and d.get("edge_kind") == "intra_causal":
                L.add_edge(u, v, strength=d.get("strength", 1))
        raw = nx.spring_layout(
            L, seed=(seed + abs(hash(core_id))) % (2 ** 32),
            weight="strength", iterations=200,
        )
        arr = np.asarray([raw[n] for n in variables], dtype=float)
        arr -= arr.mean(axis=0)
        mx = max(float(np.linalg.norm(arr, axis=1).max()), 1e-9)
        radius = base_radius + 0.35 * math.sqrt(len(variables))
        arr = radius * arr / mx
        cluster = p + outward * cluster_offset
        for i, n in enumerate(variables):
            pos[n] = (float(cluster[0] + arr[i, 0]), float(cluster[1] + arr[i, 1]))

        # catalysts near barycentre of their targets
        cats = [
            n for n, d in CoC.nodes(data=True)
            if d.get("node_kind") == "catalyst" and d.get("parent_core") == core_id
        ]
        for cid in cats:
            targets = [
                v for u, v, _k, d in iter_edges_with_keys(CoC)
                if u == cid and d.get("edge_kind") == "catalyst_affects" and v in pos
            ]
            if targets:
                bary = np.mean([pos[t] for t in targets], axis=0)
                nudge = bary - p
                nn = float(np.linalg.norm(nudge))
                nudge = np.array([1.0, 0.0]) if nn < 1e-9 else nudge / nn
                pos[cid] = tuple(bary + 0.85 * nudge)
            else:
                pos[cid] = tuple(p + outward * (cluster_offset + radius + 0.8))

    # shared concepts at member barycentre
    for n, d in CoC.nodes(data=True):
        if d.get("node_kind") != "shared_concept":
            continue
        members = [
            u for u, v, _k, ed in iter_edges_with_keys(CoC)
            if v == n and ed.get("edge_kind") == "instance_of" and u in pos
        ]
        if members:
            pos[n] = tuple(np.mean([pos[m] for m in members], axis=0)
                           + np.array([0.0, 1.3]))
        else:
            pos[n] = (0.0, 0.0)

    # any leftover nodes (shouldn't happen, but keep layout complete)
    for n in CoC.nodes():
        if n not in pos:
            pc = CoC.nodes[n].get("parent_core")
            pos[n] = pos.get(f"core::{pc}", (0.0, 0.0)) if pc else (0.0, 0.0)
    return pos


# --------------------------------------------------------------------------- #
# 8. Drawing helpers (quotient + interface map + full modular)
# --------------------------------------------------------------------------- #
def draw_quotient_graph(Q: nx.DiGraph, pos: dict, *, title: str, output_path: str,
                        figsize=(12, 10), show: bool = True, dpi: int = 140):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)
    for u, v, d in Q.edges(data=True):
        if u not in pos or v not in pos:
            continue
        sign = d.get("sign", "+")
        st = SIGN_STYLE.get(sign, SIGN_STYLE["?"])
        w = 1.0 + 2.0 * float(d.get("W", d.get("weight", 0.1)) or 0.1) * 20
        w = min(max(w, 1.0), 5.0)
        nx.draw_networkx_edges(
            Q, pos, edgelist=[(u, v)], ax=ax,
            edge_color=st["color"], style=st["style"], width=w,
            alpha=0.85, arrowsize=18, connectionstyle="arc3,rad=0.12",
        )
    nodes = list(Q.nodes())
    sizes = [1800 + 80 * Q.nodes[n].get("n_vars", 0) for n in nodes]
    nx.draw_networkx_nodes(Q, pos, nodelist=nodes, node_size=sizes,
                           node_color="#4c78a8", edgecolors="black",
                           linewidths=2.0, ax=ax)
    labels = {n: wrap_label(Q.nodes[n].get("label", n.split("::")[-1]), 18) for n in nodes}
    for n, txt in labels.items():
        x, y = pos[n]
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    handles = [
        Line2D([0], [0], color="#1a9850", lw=2.5, label="net + influence"),
        Line2D([0], [0], color="#d73027", lw=2.5, label="net − influence"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8",
               markeredgecolor="k", markersize=14, label="module (sub-core)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    ax.margins(0.15)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def draw_interface_map(CoC: nx.MultiDiGraph, pos: dict, interface_ids: dict,
                       *, title: str, output_path: str, figsize=(18, 14),
                       show: bool = True, dpi: int = 140):
    """Sparse view: module nodes + interface/bridge variables + cross_causal +
    instance_of. No full intra-subcore clutter."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)
    retained = set(pos)

    # containment (faint)
    el = [(u, v) for u, v, _k, d in iter_edges_with_keys(CoC)
          if d.get("edge_kind") == "contains" and u in retained and v in retained]
    if el:
        nx.draw_networkx_edges(CoC, pos, edgelist=el, ax=ax, edge_color="#e8e8e8",
                               width=0.4, alpha=0.35, arrows=False)
    # instance_of
    el = [(u, v) for u, v, _k, d in iter_edges_with_keys(CoC)
          if d.get("edge_kind") == "instance_of" and u in retained and v in retained]
    if el:
        nx.draw_networkx_edges(CoC, pos, edgelist=el, ax=ax, edge_color="#c9b458",
                               style="dashed", width=0.8, alpha=0.55, arrows=False)
    # cross_causal — strongest layer
    for u, v, _k, d in iter_edges_with_keys(CoC):
        if d.get("edge_kind") != "cross_causal" or u not in retained or v not in retained:
            continue
        sign = d.get("sign", "?")
        st = SIGN_STYLE.get(sign, SIGN_STYLE["?"])
        conf = float(d.get("confidence", 0.7) or 0.7)
        nx.draw_networkx_edges(
            CoC, pos, edgelist=[(u, v)], ax=ax,
            edge_color=st["color"], style=st["style"],
            width=0.6 + 0.9 * float(d.get("strength", 1)),
            alpha=max(0.45, min(1.0, conf)), arrowsize=14,
            connectionstyle="arc3,rad=0.12",
        )

    cores = [n for n, d in CoC.nodes(data=True) if d.get("node_kind") == "core" and n in pos]
    vars_ = [n for n, d in CoC.nodes(data=True) if d.get("node_kind") == "variable" and n in pos]
    concepts = [n for n, d in CoC.nodes(data=True)
                if d.get("node_kind") == "shared_concept" and n in pos]

    if cores:
        nx.draw_networkx_nodes(CoC, pos, nodelist=cores, node_size=1600,
                               node_color="#4c78a8", edgecolors="black",
                               linewidths=2.0, ax=ax)
    if vars_:
        colors = [ROLE_COLOR.get(CoC.nodes[n].get("local_role"), "#cccccc") for n in vars_]
        nx.draw_networkx_nodes(CoC, pos, nodelist=vars_, node_size=280,
                               node_color=colors, edgecolors="#333",
                               linewidths=0.7, ax=ax)
    if concepts:
        nx.draw_networkx_nodes(CoC, pos, nodelist=concepts, node_shape="h",
                               node_size=500, node_color="#f5f0b0",
                               edgecolors="#8a6d00", linewidths=1.2, ax=ax)

    labels = {}
    for n in cores:
        labels[n] = wrap_label(CoC.nodes[n].get("label", n), 16)
    for n in concepts:
        labels[n] = wrap_label(CoC.nodes[n].get("label", n), 14)
    # label only interface vars (not every bridge endpoint)
    iface_nodes = {f"{cid}::{lid}" for cid, lids in interface_ids.items() for lid in lids}
    for n in vars_:
        if n in iface_nodes:
            labels[n] = wrap_label(CoC.nodes[n].get("label", n), 14)
    for n, txt in labels.items():
        x, y = pos[n]
        fw = "bold" if CoC.nodes[n].get("node_kind") == "core" else "normal"
        fs = 8 if CoC.nodes[n].get("node_kind") == "core" else 6.5
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight=fw,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55))

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8",
               markeredgecolor="k", markersize=12, label="module"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=8, label="interface var (fill=local role)"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="#f5f0b0",
               markeredgecolor="#8a6d00", markersize=11, label="shared concept"),
        Line2D([0], [0], color="#1a9850", lw=2, label="cross-core +"),
        Line2D([0], [0], color="#d73027", lw=2, label="cross-core −"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    ax.margins(0.12)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def draw_full_modular_graph(CoC: nx.MultiDiGraph, pos: dict, *,
                            title: str, output_path: str,
                            interface_ids: Optional[dict] = None,
                            figsize=(22, 17), show: bool = True, dpi: int = 140):
    """Full merged-core view: every module, variable, catalyst, and concept node.
    Edge hierarchy: cross_causal strongest; intra_causal faint; contains /
    instance_of / catalyst structural. Labels: modules + interfaces + top
    bridge-score vars + concepts (all nodes still drawn)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)
    retained = set(pos)
    interface_ids = interface_ids or {}

    def _ok(u, v):
        return u in retained and v in retained

    # containment — very faint
    el = [(u, v) for u, v, _k, d in iter_edges_with_keys(CoC)
          if d.get("edge_kind") == "contains" and _ok(u, v)]
    if el:
        nx.draw_networkx_edges(CoC, pos, edgelist=el, ax=ax, edge_color="#ececec",
                               width=0.35, alpha=0.3, arrows=False)
    # instance_of
    el = [(u, v) for u, v, _k, d in iter_edges_with_keys(CoC)
          if d.get("edge_kind") == "instance_of" and _ok(u, v)]
    if el:
        nx.draw_networkx_edges(CoC, pos, edgelist=el, ax=ax, edge_color="#c9b458",
                               style="dashed", width=0.7, alpha=0.5, arrows=False)
    # catalyst_affects
    el = [(u, v) for u, v, _k, d in iter_edges_with_keys(CoC)
          if d.get("edge_kind") == "catalyst_affects" and _ok(u, v)]
    if el:
        nx.draw_networkx_edges(CoC, pos, edgelist=el, ax=ax, edge_color="#d0a000",
                               style="dotted", width=0.6, alpha=0.45, arrowsize=6)
    # intra_causal — pale, keep modular structure visible
    el = [(u, v, d) for u, v, _k, d in iter_edges_with_keys(CoC)
          if d.get("edge_kind") == "intra_causal" and _ok(u, v)]
    if el:
        nx.draw_networkx_edges(
            CoC, pos, edgelist=[(u, v) for u, v, _ in el], ax=ax,
            edge_color="#9aa5b1",
            width=[0.4 + 0.5 * float(d.get("strength", 1)) for _, _, d in el],
            alpha=0.25, arrowsize=6, connectionstyle="arc3,rad=0.05",
        )
    # skip derived_summary at variable level (module overview only)
    # cross_causal — strongest
    for u, v, _k, d in iter_edges_with_keys(CoC):
        if d.get("edge_kind") != "cross_causal" or not _ok(u, v):
            continue
        sign = d.get("sign", "?")
        st = SIGN_STYLE.get(sign, SIGN_STYLE["?"])
        conf = float(d.get("confidence", 0.7) or 0.7)
        nx.draw_networkx_edges(
            CoC, pos, edgelist=[(u, v)], ax=ax,
            edge_color=st["color"], style=st["style"],
            width=0.7 + 1.0 * float(d.get("strength", 1)),
            alpha=max(0.5, min(1.0, conf)), arrowsize=14,
            connectionstyle="arc3,rad=0.12",
        )

    cores = [n for n, d in CoC.nodes(data=True) if d.get("node_kind") == "core" and n in pos]
    vars_ = [n for n, d in CoC.nodes(data=True) if d.get("node_kind") == "variable" and n in pos]
    cats = [n for n, d in CoC.nodes(data=True) if d.get("node_kind") == "catalyst" and n in pos]
    concepts = [n for n, d in CoC.nodes(data=True)
                if d.get("node_kind") == "shared_concept" and n in pos]

    if cores:
        nx.draw_networkx_nodes(CoC, pos, nodelist=cores, node_size=1700,
                               node_color="#4c78a8", edgecolors="black",
                               linewidths=2.2, ax=ax)
    if vars_:
        pmax = max((float(CoC.nodes[n].get("P", 0) or 0) for n in vars_), default=1.0) or 1.0
        sizes = [140 + 220 * float(CoC.nodes[n].get("P", 0) or 0) / pmax for n in vars_]
        colors = [ROLE_COLOR.get(CoC.nodes[n].get("local_role"), "#cccccc") for n in vars_]
        nx.draw_networkx_nodes(CoC, pos, nodelist=vars_, node_size=sizes,
                               node_color=colors, edgecolors="#333",
                               linewidths=0.6, ax=ax)
    if cats:
        nx.draw_networkx_nodes(CoC, pos, nodelist=cats, node_shape="*",
                               node_size=180, node_color="#ffcc00",
                               edgecolors="black", linewidths=0.4, ax=ax)
    if concepts:
        nx.draw_networkx_nodes(CoC, pos, nodelist=concepts, node_shape="h",
                               node_size=550, node_color="#f5f0b0",
                               edgecolors="#8a6d00", linewidths=1.2, ax=ax)

    # labels: all cores + concepts; interfaces; top bridge-score vars per core
    labels = {}
    for n in cores:
        labels[n] = wrap_label(CoC.nodes[n].get("label", n), 16)
    for n in concepts:
        labels[n] = wrap_label(CoC.nodes[n].get("label", n), 14)
    iface_nodes = {f"{cid}::{lid}" for cid, lids in interface_ids.items() for lid in lids}
    for n in vars_:
        if n in iface_nodes:
            labels[n] = wrap_label(CoC.nodes[n].get("label", n), 13)
    # top-2 bridge-score vars per core (if computed)
    by_core: dict[str, list] = {}
    for n in vars_:
        by_core.setdefault(CoC.nodes[n].get("parent_core"), []).append(n)
    for cid, ns in by_core.items():
        ranked = sorted(ns, key=lambda n: float(CoC.nodes[n].get("bridge_score", 0) or 0),
                        reverse=True)
        for n in ranked[:2]:
            if n not in labels:
                labels[n] = wrap_label(CoC.nodes[n].get("label", n), 13)

    for n, txt in labels.items():
        x, y = pos[n]
        nk = CoC.nodes[n].get("node_kind")
        fw = "bold" if nk == "core" else "normal"
        fs = 8.5 if nk == "core" else 6.0
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight=fw,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.55))

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8",
               markeredgecolor="k", markersize=13, label="module"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=8, label="variable (fill=local role)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffcc00",
               markeredgecolor="k", markersize=12, label="catalyst"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="#f5f0b0",
               markeredgecolor="#8a6d00", markersize=11, label="shared concept"),
        Line2D([0], [0], color="#1a9850", lw=2.2, label="cross-core +"),
        Line2D([0], [0], color="#d73027", lw=2.2, label="cross-core −"),
        Line2D([0], [0], color="#9aa5b1", lw=1.2, label="intra-core causal"),
        Line2D([0], [0], color="#d0a000", lw=1.0, ls=":", label="catalyst link"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    ax.margins(0.12)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def print_graph_stats(CoC: nx.MultiDiGraph, name: str = "CoC") -> None:
    from collections import Counter
    nk = Counter(d.get("node_kind") for _, d in CoC.nodes(data=True))
    ek = Counter(d.get("edge_kind") for *_, d in iter_edges_with_keys(CoC))
    print(f"{name}: {CoC.number_of_nodes()} nodes, {CoC.number_of_edges()} edges")
    print("  node kinds:", dict(nk))
    print("  edge kinds:", dict(ek))


# --------------------------------------------------------------------------- #
# 9. Two-core merge → synthetic CoreOntology (for normal compass / effect plots)
# --------------------------------------------------------------------------- #
def resolve_core_key(sub_cores: dict, needle: str) -> str:
    """Match a short label (e.g. 'iran', 'ukraine') to a ``sub_cores`` key."""
    needle = (needle or "").lower().strip()
    if needle in sub_cores:
        return needle
    hits = [k for k in sub_cores if needle in k.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"No sub-core matching {needle!r}. Available: {list(sub_cores)}")
    # prefer the shortest key among hits (usually the cleanest id)
    return min(hits, key=len)


def _short_prefix(core_id: str, max_len: int = 10) -> str:
    """Readable namespace prefix from a core id (e.g. us_iran_conflict → iran)."""
    parts = [p for p in re.split(r"[_\-]+", core_id.lower()) if p]
    drop = {"us", "the", "and", "of", "a", "an"}
    keep = [p for p in parts if p not in drop]
    if not keep:
        keep = parts or ["core"]
    # prefer distinctive tokens: iran, ukraine, china, nato, energy…
    pref = keep[-1] if len(keep) > 1 else keep[0]
    # special-cases for common pairs
    joined = "_".join(parts)
    if "iran" in joined:
        pref = "iran"
    elif "ukraine" in joined or "russia" in joined:
        pref = "ukraine" if "ukraine" in joined else "russia"
    elif "china" in joined:
        pref = "china"
    elif "nato" in joined:
        pref = "nato"
    elif "energy" in joined:
        pref = "energy"
    return slugify(pref, max_len=max_len)


def merge_two_cores_as_ontology(
    sub_a,
    sub_b,
    *,
    core_id_a: str,
    core_id_b: str,
    bridges: Optional[Sequence[dict]] = None,
    label: Optional[str] = None,
    edge_threshold: int = 2,
):
    """Merge two Vester sub-cores into one synthetic ``CoreOntology``.

    - Variables are namespaced ``{prefix}__{local_id}`` so plotters stay happy.
    - Impact matrix is block-diagonal (each core's ``M``) with sparse off-diagonal
      fills from ``bridges`` (strength only; signs live on graph edges).
    - AS / PS / roles are recomputed on the combined matrix (same Vester rule).
    - Catalysts from both cores are kept, with namespaced ids / affected vars.

    The result can be passed to ``vester_core.plot_compass`` /
    ``plot_effect_graph`` exactly like a normal single-core run.
    """
    # late import avoids circular deps at module load
    import vester_core as vc

    pref_a, pref_b = _short_prefix(core_id_a), _short_prefix(core_id_b)
    if pref_a == pref_b:
        pref_b = slugify(core_id_b, max_len=12)

    def _ns(pref, local_id):
        return f"{pref}__{local_id}"

    variables = []
    var_ids = []
    var_names = {}
    local_to_ns = {}  # (core_id, local_id) -> namespaced id

    for pref, cid, sub in ((pref_a, core_id_a, sub_a), (pref_b, core_id_b, sub_b)):
        for v in sub.variables:
            nid = _ns(pref, v["id"])
            local_to_ns[(cid, v["id"])] = nid
            nv = dict(v)
            nv["id"] = nid
            nv["name"] = f"[{pref}] {v.get('name', v['id'])}"
            nv["parent_core"] = cid
            nv["local_id"] = v["id"]
            variables.append(nv)
            var_ids.append(nid)
            var_names[nid] = nv["name"]

    n_a, n_b = sub_a.N, sub_b.N
    N = n_a + n_b
    M = np.zeros((N, N), dtype=float)
    M[:n_a, :n_a] = np.asarray(sub_a.M, dtype=float)
    M[n_a:, n_a:] = np.asarray(sub_b.M, dtype=float)

    # off-diagonal from bridges (unsigned strength into M; sign on graph)
    idx = {vid: i for i, vid in enumerate(var_ids)}
    bridge_edges = []  # (src_ns, tgt_ns, strength, sign, mechanism, …)
    for b in bridges or []:
        sc = b.get("source_core") or core_id_a
        tc = b.get("target_core") or core_id_b
        s_loc = str(b.get("source_variable") or b.get("source_variable_id") or "")
        t_loc = str(b.get("target_variable") or b.get("target_variable_id") or "")
        src = local_to_ns.get((sc, s_loc))
        tgt = local_to_ns.get((tc, t_loc))
        if src is None or tgt is None:
            continue
        strength = int(b.get("strength", 1) or 1)
        sign = b.get("sign", "?")
        i, j = idx[src], idx[tgt]
        M[i, j] = max(M[i, j], strength)
        bridge_edges.append((src, tgt, strength, sign, b))

    M = np.clip(M, 0, 3)
    impact_df = pd.DataFrame(M, index=var_ids, columns=var_ids)
    AS, PS, med_AS, med_PS, roles = vc._roles(M, var_ids, var_names)

    # effect graph: keep local edges >= threshold + all bridge edges
    G = nx.DiGraph()
    for vid in var_ids:
        i = idx[vid]
        G.add_node(vid, name=var_names[vid], kind="variable",
                   AS=int(AS[i]), PS=int(PS[i]),
                   parent_core=next(
                       v["parent_core"] for v in variables if v["id"] == vid))
    # intra edges from each sub-core graph
    for pref, cid, sub in ((pref_a, core_id_a, sub_a), (pref_b, core_id_b, sub_b)):
        for u, w, d in sub.graph.edges(data=True):
            su, sw = _ns(pref, u), _ns(pref, w)
            if su in G and sw in G:
                G.add_edge(su, sw,
                           strength=int(d.get("strength", 1)),
                           sign=d.get("sign", "?"),
                           edge_kind="intra_causal")
    # also materialise strong M entries as edges if missing (keeps plot dense enough)
    for i, ri in enumerate(var_ids):
        for j, cj in enumerate(var_ids):
            if i == j or M[i, j] < edge_threshold:
                continue
            if not G.has_edge(ri, cj):
                G.add_edge(ri, cj, strength=int(M[i, j]), sign="?",
                           edge_kind="from_matrix")
    for src, tgt, strength, sign, b in bridge_edges:
        G.add_edge(
            src, tgt,
            strength=strength,
            sign=sign,
            edge_kind="cross_causal",
            mechanism=b.get("mechanism", ""),
            confidence=float(b.get("confidence", 0.5) or 0.5),
            # mark model-/LLM-proposed bridges for highlight plots
            predicted=bool(b.get("source_method") or b.get("predicted")),
            source_method=str(b.get("source_method") or ""),
        )

    # catalysts — namespace ids / affected vars, then re-score for compass overlay
    catalysts = []
    for pref, cid, sub in ((pref_a, core_id_a, sub_a), (pref_b, core_id_b, sub_b)):
        for c in sub.catalysts:
            new_id = f"{pref}__{c['id']}"
            affected = []
            for a in c.get("affected", []):
                loc = a.get("variable_id")
                ns = local_to_ns.get((cid, loc))
                if ns:
                    affected.append({**a, "variable_id": ns})
            catalysts.append({
                **c,
                "id": new_id,
                "title": f"[{pref}] {c.get('title', new_id)}",
                "type": c.get("type", "event"),
                "rationale": c.get("rationale", ""),
                "timeline": c.get("timeline") or {"label": "", "kind": ""},
                "affected": affected,
                "parent_core": cid,
            })
    catalysts_df = (
        vc._score_catalysts(catalysts, var_ids, AS, PS)
        if catalysts else pd.DataFrame()
    )

    title_a = getattr(sub_a, "narrative_title", None) or core_id_a
    title_b = getattr(sub_b, "narrative_title", None) or core_id_b
    return vc.CoreOntology(
        label=label or f"{pref_a}_{pref_b}_merge",
        as_of=getattr(sub_a, "as_of", None) or getattr(sub_b, "as_of", None),
        model=getattr(sub_a, "model", "") or getattr(sub_b, "model", ""),
        system_description={
            "narrative_title": f"Two-core merge: {title_a} ⊕ {title_b}",
            "narrative_summary": (
                f"Block-diagonal merge of '{core_id_a}' and '{core_id_b}' "
                f"with {len(bridge_edges)} sparse cross-core bridges."
            ),
            "merged_cores": [core_id_a, core_id_b],
            "prefixes": {core_id_a: pref_a, core_id_b: pref_b},
            "n_bridges": len(bridge_edges),
        },
        variables=variables,
        var_ids=var_ids,
        var_names=var_names,
        crit_matrix=pd.DataFrame(index=var_ids),
        M=M,
        impact_df=impact_df,
        AS=AS,
        PS=PS,
        med_AS=med_AS,
        med_PS=med_PS,
        roles=roles,
        graph=G,
        catalysts=catalysts,
        catalysts_df=catalysts_df,
    )

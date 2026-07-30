"""
Full core-of-cores v2 pipeline (headless port of core_of_cores_v2.ipynb Parts 2–4).

  Part 2 — BERTopic clustering → geo topic filter → issue extraction → geo core
  Part 3 — Route issues to topics → build sub-cores with catalysts
  Part 4 — Modular graph merge, roles, PNG exports, GEXF/GraphML/JSON

Designed for daily GitHub Actions via run_daily_pipeline.py.

Usage:
  python build_coc_pipeline.py
  python build_coc_pipeline.py --stamp 2026-07-30 --skip-part4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import permutations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import build_news_kg as kg
import coc_modular as cm
import vester_core as vc
from summarize_headlines import get_pipeline_date

KG_DIR = Path("kg")

# --- knobs (mirror notebook defaults) ---
BROAD_LOOKBACK_DAYS = 3
MIN_TITLE_LEN = 25
NR_TOPICS = 40
TOP_N_TOPICS = 40
SAMPLES_PER_TOPIC = 3
GEO_CORE_N = 5
MAX_GEO_HEADLINES_FOR_LLM = 600
TOPICS_PER_ISSUE_MAX = 4
SUB_CORE_N = 16
SUB_LOOKBACK_DAYS = 10
MIN_ISSUE_HEADLINES = 5
MAX_ISSUE_HEADLINES = 200
SUB_CORE_BUILD_ATTEMPTS = 3
N_INTERFACE = 5
MAX_BRIDGES = 4
MIN_CONF = 0.55
CONCEPT_SIM_THRESHOLD = 0.82
ALPHA = 0.6

GEO_SYS_MSG = (
    "You are a senior geopolitical strategist trained in Frederic Vester's "
    "Sensitivity Model. Across a corpus of world-news headlines you identify the "
    "salient ISSUES that structure the current global order: ongoing conflicts and "
    "event-streams, shifting balances of power, and points of contact or friction "
    "between major powers. You treat each issue as an ongoing process, not a "
    "one-off event or a named entity."
)

GEO_STEP1 = """You are given world news headlines from <<WINDOW_FROM>> to <<WINDOW_TO>>.

Step 1 - System Description (the global geopolitical order).
1. Describe the current global geopolitical order as a single open system in flux.
2. State its scope, the larger super-system it sits in, and 2-4 adjacent subsystems.

Return strict JSON:
{
  "narrative_title": "<short title, < 80 chars>",
  "narrative_summary": "<2-4 sentences on the dominant dynamics>",
  "scope": "<what is inside the system boundary>",
  "super_system": "<the larger system this is embedded in>",
  "adjacent_subsystems": ["<name>", "..."],
  "evidence_headlines": ["<3-6 verbatim or near-verbatim headlines>"]
}

HEADLINES:
<<HEADLINES>>
"""


@dataclass
class CocRunState:
    stamp: str
    window_days: tuple[date, date]
    articles: pd.DataFrame = field(default_factory=pd.DataFrame)
    topic_model: Any = None
    topic_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    geo_topic_ids: list[int] = field(default_factory=list)
    geo_articles: pd.DataFrame = field(default_factory=pd.DataFrame)
    geo_headlines: list[str] = field(default_factory=list)
    geo_issues: list[dict] = field(default_factory=list)
    geo_core: vc.CoreOntology | None = None
    issue_variables: list[dict] = field(default_factory=list)
    issue_to_topics: dict[str, list[int]] = field(default_factory=dict)
    sub_cores: dict[str, vc.CoreOntology] = field(default_factory=dict)
    coc_graph: nx.MultiDiGraph | None = None
    interface_ids: dict[str, list[str]] = field(default_factory=dict)
    quotient: nx.DiGraph | None = None
    roles_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _resolve_stamp(explicit: str | None, window_end: date) -> str:
    return explicit or os.environ.get("PIPELINE_DATE") or get_pipeline_date() or window_end.isoformat()


def load_corpus(*, lookback_days: int = BROAD_LOOKBACK_DAYS) -> tuple[pd.DataFrame, tuple[date, date]]:
    items = kg._discover_article_files()
    if not items:
        raise FileNotFoundError("No data/articles_YYYY-MM-DD.csv files found.")
    latest = items[-1][0]
    cutoff = latest - timedelta(days=lookback_days - 1)
    recent_items = [(d, p) for d, p in items if d >= cutoff]

    frames = []
    for d, p in recent_items:
        df = pd.read_csv(p)
        df["file_date"] = d
        frames.append(df)
    articles = pd.concat(frames, ignore_index=True)
    articles["doc"] = (
        articles["title"].fillna("") + " " + articles.get("content", pd.Series("")).fillna("")
    ).str.strip()
    articles = articles[articles["doc"].str.len() > 0].drop_duplicates(subset=["title"])
    articles["_clean"] = articles["title"].map(kg._clean_title)
    articles = articles[articles["_clean"].str.len() >= MIN_TITLE_LEN].reset_index(drop=True)

    window = (recent_items[0][0].date(), recent_items[-1][0].date())
    print(
        f"Loaded {len(articles):,} headlines from {len(recent_items)} days "
        f"({window[0]} to {window[1]})"
    )
    return articles, window


def fit_bertopic(articles: pd.DataFrame):
    from bertopic import BERTopic

    docs = articles["doc"].astype(str).tolist()
    print(f"BERTopic: fitting on {len(docs):,} documents → {NR_TOPICS} topics …")
    topic_model = BERTopic(verbose=False)
    topic_model.fit_transform(docs)
    topic_model.reduce_topics(docs, nr_topics=NR_TOPICS)
    articles["topic"] = topic_model.topics_

    topic_info = topic_model.get_topic_info()
    topic_catalog = topic_info[topic_info.Topic != -1].copy()
    topic_catalog["sample"] = topic_catalog["Representative_Docs"].apply(
        lambda d: (d[0][:90] + "...") if d else ""
    )
    n_topics = len(topic_catalog)
    n_outliers = int(topic_info.loc[topic_info.Topic == -1, "Count"].sum())
    print(f"BERTopic: {n_topics} topics (+ {n_outliers:,} outliers in topic -1)")
    return topic_model, topic_catalog


def filter_geo_topics(
    articles: pd.DataFrame,
    topic_catalog: pd.DataFrame,
    *,
    client,
) -> tuple[list[int], pd.DataFrame, list[str]]:
    top_topics = topic_catalog.nlargest(TOP_N_TOPICS, "Count")
    lines = []
    for r in top_topics.itertuples():
        reps = r.Representative_Docs if isinstance(r.Representative_Docs, list) else []
        samples = [str(h)[:120] for h in reps[:SAMPLES_PER_TOPIC]]
        kw = ", ".join(list(r.Representation)[:8])
        sample_txt = " | ".join(samples) if samples else r.sample
        lines.append(f"- topic {r.Topic} (n={r.Count}): {kw}\n  samples: {sample_txt}")

    geo_filter_prompt = f"""You are screening BERTopic news clusters for GEOPOLITICAL relevance.

Mark a topic relevant if it is relevant to geopolitics and the global economy.

Mark NOT relevant: single-stock/insider trades, earnings calls, sports, celebrity,
local crime, pure tech product reviews.

Return strict JSON:
{{ "relevant_topics": [<topic_id int>, ...], "notes": {{ "<topic_id>": "<short reason>" }} }}

TOPICS:
{chr(10).join(lines)}
"""
    geo_filter = vc.llm_json(
        client,
        vc.DEFAULT_MODEL,
        geo_filter_prompt,
        system="You classify news topic clusters for geopolitical relevance.",
        temperature=0.1,
    )
    valid_ids = set(top_topics["Topic"].astype(int))
    geo_topic_ids = sorted(
        {
            int(t)
            for t in geo_filter.get("relevant_topics", [])
            if str(t).lstrip("-").isdigit() and int(t) in valid_ids
        }
    )
    geo_articles = (
        articles[articles["topic"].isin(geo_topic_ids)]
        .drop_duplicates(subset=["_clean"])
        .sort_values("file_date")
    )
    geo_headlines = geo_articles["_clean"].tolist()
    print(f"Geopolitical topics: {len(geo_topic_ids)} / {TOP_N_TOPICS}; "
          f"{len(geo_headlines):,} headlines")
    return geo_topic_ids, geo_articles, geo_headlines


def extract_geo_issues(
    geo_headlines: list[str],
    window_days: tuple[date, date],
    *,
    client,
    geo_core_n: int = GEO_CORE_N,
) -> list[dict]:
    hl_block = geo_headlines
    if len(hl_block) > MAX_GEO_HEADLINES_FOR_LLM:
        print(f"Capping geo headlines for issue extraction: {len(hl_block)} → {MAX_GEO_HEADLINES_FOR_LLM}")
        hl_block = hl_block[-MAX_GEO_HEADLINES_FOR_LLM:]

    freeform_prompt = f"""You are given world-news headlines from {window_days[0]} to {window_days[1]}.

List the {geo_core_n} most salient ISSUES that currently structure the global
geopolitical order. Think in terms of ongoing situations and event-streams,
shifting balances of power, and points of contact or friction between major
powers - not one-off events and not named entities. Aim for broad coverage:
major wars, trade/tech rivalries, nuclear standoffs, energy chokepoints,
alliance realignments, sanctions regimes, and regional flashpoints. Each issue
must be distinct and at the same level of aggregation (no synonyms, no nesting).

No nesting: there shouldn't be more than 1 variable per issue (e.g. only one iran-us conflict related variable)
No variable may contain a named mechanism, geography or consequence that is also represented by another variable.

For each issue return:
  id           snake_case, <= 24 chars, globally unique
  name         human-readable display name
  description  1 sentence: which powers are involved and what is shifting
  kind         one of: conflict, rivalry, alliance, negotiation, power_shift, flashpoint

Return strict JSON: {{ "issues": [ {{"id": ..., "name": ..., "description": ..., "kind": ...}} ] }}

HEADLINES:
{chr(10).join(hl_block)}
"""
    resp = vc.llm_json(client, vc.DEFAULT_MODEL, freeform_prompt, system=GEO_SYS_MSG, temperature=0.3)
    geo_issues = resp.get("issues") or resp.get("variables") or []
    for v in geo_issues:
        v["ontology_type"] = "process"
    print(f"Extracted {len(geo_issues)} salient geopolitical issues")
    for v in geo_issues:
        print(f"  [{v.get('id')}] {v.get('name')}")
    return geo_issues


def build_geo_core(
    geo_headlines: list[str],
    geo_issues: list[dict],
    window_days: tuple[date, date],
    *,
    client,
    stamp: str,
) -> vc.CoreOntology:
    seed_issues = [dict(v) for v in geo_issues if (v.get("id") or v.get("name"))]
    geo_prompts = vc.PromptSet(sys_msg=GEO_SYS_MSG, step1=GEO_STEP1, core_n=len(seed_issues))
    print(f"Building geo core ({len(seed_issues)} issues, {len(geo_headlines):,} headlines) …")
    geo_core = vc.build_core_ontology(
        geo_headlines,
        prompts=geo_prompts,
        seed_variables=seed_issues,
        run_retention=False,
        with_catalysts=False,
        core_n=len(seed_issues),
        edge_threshold=2,
        label="geo",
        window_label=(str(window_days[0]), str(window_days[1])),
        client=client,
    )
    path = geo_core.save(KG_DIR, stamp=stamp)
    print(f"Geo core saved: {path.name} ({geo_core.N} issues)")
    return geo_core


def route_issues_to_topics(
    geo_core: vc.CoreOntology,
    topic_catalog: pd.DataFrame,
    articles: pd.DataFrame,
    *,
    client,
) -> dict[str, list[int]]:
    issue_variables = list(geo_core.variables)
    cat_lines = []
    for r in topic_catalog.sort_values("Count", ascending=False).itertuples():
        rep = ", ".join(list(r.Representation)[:6])
        cat_lines.append(f"- topic {r.Topic} (n={r.Count}): {rep} | e.g. {r.sample}")
    issue_block = "\n".join(
        f"- {v['id']}: {v['name']} - {v.get('description', '')}" for v in issue_variables
    )
    route_prompt = f"""Assign BERTopic clusters to geopolitical issues.

For EACH issue below, list the topic ids (from the catalog) whose headlines belong to it.
Choose up to {TOPICS_PER_ISSUE_MAX} of the most relevant topics per issue; a topic
should serve at most one issue; skip topics that fit no issue.

Return strict JSON: {{ "mapping": {{ "<issue_id>": [<topic_id>, ...], ... }} }}

ISSUES:
{issue_block}

TOPICS:
{chr(10).join(cat_lines)}
"""
    route = vc.llm_json(client, vc.DEFAULT_MODEL, route_prompt, system=GEO_SYS_MSG, temperature=0.1)
    valid_topics = {int(t) for t in topic_catalog["Topic"]}
    issue_to_topics: dict[str, list[int]] = {}
    for v in issue_variables:
        ids = route.get("mapping", {}).get(v["id"], []) or []
        ids = [int(t) for t in ids if str(t).lstrip("-").isdigit() and int(t) in valid_topics]
        issue_to_topics[v["id"]] = ids[:TOPICS_PER_ISSUE_MAX]

    unmapped = [iid for iid, ts in issue_to_topics.items() if not ts]
    if unmapped:
        topic_ids = [int(t) for t in topic_catalog["Topic"]]
        topic_texts = [
            f"{r.Name}: {', '.join(list(r.Representation)[:8])}"
            for r in topic_catalog.itertuples()
        ]
        tvec = vc.embed_texts(client, topic_texts)
        for iid in unmapped:
            v = next(x for x in issue_variables if x["id"] == iid)
            ivec = vc.embed_texts(client, [f"{v['name']}: {v.get('description', '')}"])[0]
            sims = tvec @ ivec
            issue_to_topics[iid] = [topic_ids[j] for j in np.argsort(-sims)[:2]]

    for iid, ts in issue_to_topics.items():
        n = int(articles["topic"].isin(ts).sum())
        print(f"  {iid:24s} -> topics {ts}  ({n} headlines)")
    return issue_to_topics


def _subcore_path(iid: str, stamp: str) -> Path:
    return KG_DIR / f"{vc._slugify(iid)}_core_{stamp}.json"


def build_sub_cores(
    issue_variables: list[dict],
    issue_to_topics: dict[str, list[int]],
    articles: pd.DataFrame,
    window_days: tuple[date, date],
    *,
    client,
    stamp: str,
) -> dict[str, vc.CoreOntology]:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    sub_cores: dict[str, vc.CoreOntology] = {}
    for v in tqdm(issue_variables, desc="sub-cores"):
        iid = v["id"]
        existing = _subcore_path(iid, stamp)
        if existing.exists():
            try:
                sub_cores[iid] = vc.CoreOntology.load(existing)
                print(f"  {iid}: reusing {existing.name}")
                continue
            except Exception as e:
                print(f"  {iid}: could not load {existing.name} ({e}); rebuilding")

        tids = issue_to_topics.get(iid, [])
        sel = articles[articles["topic"].isin(tids)] if tids else articles.iloc[0:0]
        sel = sel.drop_duplicates(subset=["_clean"]).sort_values("file_date")
        issue_headlines = sel["_clean"].tolist()
        if len(issue_headlines) < MIN_ISSUE_HEADLINES:
            print(f"  skip {iid}: only {len(issue_headlines)} headlines")
            continue
        if len(issue_headlines) > MAX_ISSUE_HEADLINES:
            sel = sel.tail(MAX_ISSUE_HEADLINES)
            issue_headlines = sel["_clean"].tolist()
        dated = list(zip(sel["file_date"].tolist(), sel["_clean"].tolist()))

        sub: vc.CoreOntology | None = None
        last_err: Exception | None = None
        for attempt in range(1, SUB_CORE_BUILD_ATTEMPTS + 1):
            try:
                sub = vc.build_core_ontology(
                    issue_headlines,
                    dated_headlines=dated,
                    with_catalysts=True,
                    core_n=SUB_CORE_N,
                    lookback_days=SUB_LOOKBACK_DAYS,
                    label=iid,
                    window_label=(str(window_days[0]), str(window_days[1])),
                    client=client,
                    show_progress=False,
                    max_headlines=MAX_ISSUE_HEADLINES,
                )
                break
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                last_err = e
                if attempt >= SUB_CORE_BUILD_ATTEMPTS:
                    break
                wait = 30 * attempt
                print(
                    f"  {iid}: {type(e).__name__} on attempt {attempt}/"
                    f"{SUB_CORE_BUILD_ATTEMPTS}; retry in {wait}s …"
                )
                time.sleep(wait)
                client = vc._make_client()

        if sub is None:
            print(
                f"  FAILED {iid}: {type(last_err).__name__}: {last_err}",
                file=sys.stderr,
            )
            continue

        sub.save(KG_DIR, stamp=stamp)
        sub_cores[iid] = sub
        print(f"  {iid}: {sub.N} vars, {len(sub.catalysts)} catalysts")

    if len(sub_cores) < 2:
        raise RuntimeError(
            f"Only {len(sub_cores)} sub-core(s) built; need at least 2 for CoC email."
        )
    print(f"Built {len(sub_cores)} sub-cores: {list(sub_cores)}")
    return sub_cores


def merge_modular_graph(
    sub_cores: dict[str, vc.CoreOntology],
    *,
    client,
) -> tuple[nx.MultiDiGraph, dict[str, list[str]], nx.DiGraph, pd.DataFrame]:
    coc = cm.assemble_modular_graph(sub_cores)
    core_ids = sorted(sub_cores.keys())
    cm.print_graph_stats(coc, "CoC (pre-bridges)")

    interface_ids: dict[str, list[str]] = {}
    for cid in tqdm(core_ids, desc="interface vars"):
        sub = sub_cores[cid]
        vars_payload = []
        for v in sub.variables:
            nid = f"{cid}::{v['id']}"
            d = coc.nodes[nid]
            vars_payload.append({
                "id": v["id"],
                "name": v.get("name", v["id"]),
                "description": v.get("description", ""),
                "ontology_type": v.get("ontology_type", ""),
                "local_role": d.get("local_role"),
                "local_AS": d.get("local_AS"),
                "local_PS": d.get("local_PS"),
            })
        prompt = cm.interface_prompt(
            cid, getattr(sub, "narrative_title", None) or cid,
            vars_payload, k=N_INTERFACE,
        )
        valid = {v["id"] for v in sub.variables}
        try:
            res = vc.llm_json(client, vc.DEFAULT_MODEL, prompt, system=GEO_SYS_MSG, temperature=0.1)
            ids = [str(x) for x in (res.get("interface_ids") or []) if str(x) in valid]
            ids = ids[:N_INTERFACE]
        except Exception as e:
            print(f"  {cid}: LLM interface select failed ({e}); using heuristic")
            ids = []
        if len(ids) < 3:
            ids = cm.heuristic_interfaces(coc, cid, k=N_INTERFACE)
        interface_ids[cid] = ids
        for lid in ids:
            nid = f"{cid}::{lid}"
            if nid in coc:
                coc.nodes[nid]["is_interface"] = True

    def _iface_payload(cid: str) -> list[dict]:
        out = []
        for lid in interface_ids[cid]:
            nid = f"{cid}::{lid}"
            if nid not in coc:
                continue
            d = coc.nodes[nid]
            out.append({
                "id": lid,
                "name": d.get("label", lid),
                "description": d.get("description", ""),
            })
        return out

    pairs = list(permutations(core_ids, 2))
    print(f"Directed core pairs to probe: {len(pairs)}")
    n_added = 0
    for src, tgt in tqdm(pairs, desc="cross-core bridges"):
        src_vars = _iface_payload(src)
        tgt_vars = _iface_payload(tgt)
        if not src_vars or not tgt_vars:
            continue
        prompt = cm.bridge_prompt(src, tgt, src_vars, tgt_vars, max_bridges=MAX_BRIDGES)
        try:
            res = vc.llm_json(client, vc.DEFAULT_MODEL, prompt, system=GEO_SYS_MSG, temperature=0.1)
        except Exception as e:
            print(f"  bridge {src} -> {tgt} failed: {e}")
            continue
        raw = res.get("bridges") or []
        src_ok = {v["id"] for v in src_vars}
        tgt_ok = {v["id"] for v in tgt_vars}
        batch = []
        for b in raw[:MAX_BRIDGES]:
            s = str(b.get("source_variable_id") or b.get("source_variable") or "")
            t = str(b.get("target_variable_id") or b.get("target_variable") or "")
            if s not in src_ok or t not in tgt_ok:
                continue
            batch.append({
                "source_core": src,
                "target_core": tgt,
                "source_variable": s,
                "target_variable": t,
                "sign": b.get("sign", "?"),
                "strength": int(b.get("strength", 1) or 1),
                "confidence": float(b.get("confidence", 0.5) or 0.5),
                "mechanism": b.get("mechanism", ""),
                "horizon": b.get("horizon", ""),
            })
        n_added += cm.add_cross_causal_bridges(coc, batch, min_confidence=MIN_CONF)
    print(f"Added {n_added} cross_causal bridges")
    cm.print_graph_stats(coc, "CoC (post-bridges)")

    var_records = [
        {
            "node_id": n,
            "parent_core": d["parent_core"],
            "name": d["label"],
            "text": f"{d['label']}: {d.get('description', '')}",
        }
        for n, d in coc.nodes(data=True)
        if d.get("node_kind") == "variable"
    ]
    if var_records:
        emb = vc.embed_texts(client, [r["text"] for r in var_records])
        concepts = cm.cluster_concepts(
            var_records, emb,
            sim_threshold=CONCEPT_SIM_THRESHOLD, min_subcores=2,
        )
        cm.attach_concepts(coc, concepts)
        print(f"Shared concepts spanning >=2 sub-cores: {len(concepts)}")

    quotient = cm.derive_module_summary_edges(coc, write_to_coc=True)
    roles_df = cm.apply_global_roles(coc, alpha=ALPHA)
    print(f"Quotient graph: {quotient.number_of_nodes()} modules, "
          f"{quotient.number_of_edges()} derived edges")
    return coc, interface_ids, quotient, roles_df


def export_modular_outputs(
    state: CocRunState,
    *,
    skip_pngs: bool = False,
) -> None:
    stamp = state.stamp
    KG_DIR.mkdir(parents=True, exist_ok=True)
    coc = state.coc_graph
    Q = state.quotient
    interface_ids = state.interface_ids
    roles_df = state.roles_df
    assert coc is not None and Q is not None

    if not skip_pngs:
        q_pos = cm.quotient_layout(Q, seed=42, canvas=8.0)
        cm.draw_quotient_graph(
            Q, q_pos,
            title=f"Core of cores — module quotient overview ({stamp})",
            output_path=str(KG_DIR / f"coc_v2_quotient_{stamp}.png"),
        )
        iface_pos = cm.interface_map_layout(coc, interface_ids, q_pos, cluster_offset=3.8, seed=42)
        cm.draw_interface_map(
            coc, iface_pos, interface_ids,
            title=f"Core of cores — sparse interface map ({stamp})",
            output_path=str(KG_DIR / f"coc_v2_interfaces_{stamp}.png"),
        )
        full_pos = cm.full_modular_layout(
            coc, q_pos, cluster_offset=5.2, base_radius=2.4, seed=42,
        )
        cm.draw_full_modular_graph(
            coc, full_pos,
            title=f"Core of cores — full modular graph (all nodes) ({stamp})",
            output_path=str(KG_DIR / f"coc_v2_full_{stamp}.png"),
            interface_ids=interface_ids,
            figsize=(24, 18),
        )
        print(f"Saved CoC PNGs for {stamp}")

    H = cm.gephi_safe_multigraph(coc)
    nx.write_gexf(H, KG_DIR / f"coc_v2_modular_{stamp}.gexf")
    nx.write_graphml(H, KG_DIR / f"coc_v2_modular_{stamp}.graphml")

    bridges_export = [
        {
            "source": u, "target": v,
            **{k: d[k] for k in (
                "sign", "strength", "confidence", "mechanism", "horizon",
                "source_core", "target_core",
            ) if k in d},
        }
        for u, v, _k, d in cm.iter_edges_with_keys(coc)
        if d.get("edge_kind") == "cross_causal"
    ]
    concepts_export = []
    for n, d in coc.nodes(data=True):
        if d.get("node_kind") != "shared_concept":
            continue
        members = [
            u for u, v, _k, ed in cm.iter_edges_with_keys(coc)
            if v == n and ed.get("edge_kind") == "instance_of"
        ]
        concepts_export.append({"node": n, "label": d["label"], "members": members})

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": [str(state.window_days[0]), str(state.window_days[1])],
        "architecture": "modular_graph_of_graphs",
        "target_issues": [v["id"] for v in state.issue_variables],
        "core_ids": sorted(state.sub_cores.keys()),
        "interface_ids": interface_ids,
        "alpha_global_roles": ALPHA,
        "bridges": bridges_export,
        "concepts": concepts_export,
        "quotient_edges": [
            {"source": u, "target": v, **dict(d)}
            for u, v, d in Q.edges(data=True)
        ],
        "roles": roles_df.to_dict(orient="records") if not roles_df.empty else [],
        "nodes": [
            {"id": n, **{k: cm._scalar(x) for k, x in d.items()}}
            for n, d in coc.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "key": k,
             **{kk: cm._scalar(vv) for kk, vv in d.items()}}
            for u, v, k, d in coc.edges(keys=True, data=True)
        ],
    }
    (KG_DIR / f"coc_v2_modular_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    roles_df.to_csv(KG_DIR / f"coc_v2_roles_{stamp}.csv", index=False)

    manifest = {
        "stamp": stamp,
        "window": payload["window"],
        "geo_core": f"geo_core_{stamp}.json",
        "sub_cores": [f"{iid}_core_{stamp}.json" for iid in sorted(state.sub_cores)],
        "modular_json": f"coc_v2_modular_{stamp}.json",
        "n_sub_cores": len(state.sub_cores),
        "generated_at": payload["generated_at"],
    }
    (KG_DIR / f"coc_v2_run_{stamp}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported modular graph + manifest for {stamp}")


def run_coc_pipeline(
    *,
    stamp: str | None = None,
    skip_part4: bool = False,
    skip_pngs: bool = False,
) -> CocRunState:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the CoC pipeline.")

    client = vc._make_client()
    articles, window_days = load_corpus()
    resolved_stamp = _resolve_stamp(stamp, window_days[1])
    state = CocRunState(stamp=resolved_stamp, window_days=window_days, articles=articles)

    print(f"\n{'='*60}\nPart 2 — geo core (BERTopic + issue discovery)\n{'='*60}")
    state.topic_model, state.topic_catalog = fit_bertopic(state.articles)
    state.geo_topic_ids, state.geo_articles, state.geo_headlines = filter_geo_topics(
        state.articles, state.topic_catalog, client=client,
    )
    if not state.geo_headlines:
        raise RuntimeError("No geopolitical headlines after topic filter.")

    state.geo_issues = extract_geo_issues(
        state.geo_headlines, window_days, client=client,
    )
    if not state.geo_issues:
        raise RuntimeError("Issue extraction returned no issues.")

    state.geo_core = build_geo_core(
        state.geo_headlines, state.geo_issues, window_days,
        client=client, stamp=resolved_stamp,
    )
    state.issue_variables = list(state.geo_core.variables)

    print(f"\n{'='*60}\nPart 3 — sub-cores + catalysts\n{'='*60}")
    state.issue_to_topics = route_issues_to_topics(
        state.geo_core, state.topic_catalog, state.articles, client=client,
    )
    state.sub_cores = build_sub_cores(
        state.issue_variables,
        state.issue_to_topics,
        state.articles,
        window_days,
        client=client,
        stamp=resolved_stamp,
    )

    if skip_part4:
        print("Skipping Part 4 (--skip-part4).")
        return state

    print(f"\n{'='*60}\nPart 4 — modular graph merge + export\n{'='*60}")
    state.coc_graph, state.interface_ids, state.quotient, state.roles_df = merge_modular_graph(
        state.sub_cores, client=client,
    )
    export_modular_outputs(state, skip_pngs=skip_pngs)
    cm.print_graph_stats(state.coc_graph, "final CoC")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full core-of-cores v2 pipeline.")
    parser.add_argument("--stamp", default=None, help="Output date stamp YYYY-MM-DD (default: PIPELINE_DATE).")
    parser.add_argument("--skip-part4", action="store_true", help="Stop after sub-cores (skip modular merge).")
    parser.add_argument("--skip-pngs", action="store_true", help="Skip Part 4 PNG renders (still export JSON/GEXF).")
    args = parser.parse_args(argv)

    try:
        state = run_coc_pipeline(
            stamp=args.stamp,
            skip_part4=args.skip_part4,
            skip_pngs=args.skip_pngs,
        )
    except Exception as e:
        print(f"CoC pipeline failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(
        f"\nCoC pipeline complete: stamp={state.stamp}, "
        f"{len(state.sub_cores)} sub-cores, "
        f"window {state.window_days[0]} → {state.window_days[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

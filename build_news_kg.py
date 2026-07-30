"""
Build a knowledge graph of the current news landscape from recent headlines.

1) Load the last LOOKBACK_DAYS of data/articles_<date>.csv.
2) Preprocess: dedupe titles, strip source cruft, drop noise (insider trades,
   earnings-call transcripts, analyst-rating boilerplate), cap headline count.
3) Extract a KG with kg-gen (LLM extraction + entity/relation clustering).
4) Aggregate relations into a weighted graph, detect communities, rank salient
   nodes, export structured JSON, and render a PNG via networkx + matplotlib.

Designed to be called by send_off_email. `build_news_kg()` returns a Path to the
PNG or None (e.g. no data / no API key / empty graph), mirroring the
build_summary_tsne_chart() contract.
"""

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path("data")
KG_DIR = Path("kg")
KG_IMAGE_PREFIX = "news_kg_"
KG_CACHE_PREFIX = "news_kg_"
KG_STRUCTURED_PREFIX = "news_kg_structured_"

LOOKBACK_DAYS = 3  # How many days of headlines define the "current" landscape
KG_MODEL = "openai/gpt-4o-mini"  # LiteLLM-format model string; swap to openai/gpt-4o for higher quality
KG_CHUNK_SIZE = 5000  # Chars per chunk passed to kg-gen
MAX_HEADLINES = 15000  # Cap headlines to bound LLM cost and graph density
TOP_N_NODES = 80  # Keep only the most-connected nodes when drawing
MIN_TITLE_LEN = 25  # Drop very short/low-signal titles
MIN_RELATIONS_FOR_CHART = 3  # Below this the graph is too sparse to be useful

KG_CONTEXT = (
    "Extract a knowledge graph from global macro and financial-markets news headlines. "
    "Focus on concrete news events: policy actions, sanctions, military developments, "
    "market moves, legal actions, company decisions, and institutional rulings. "
    "Use canonical entity names (e.g. 'Federal Reserve', 'Iran', 'OpenAI', 'Oil prices'). "
    "Avoid vague entities such as 'officials', 'critics', 'markets', or 'analysts' unless "
    "they are the subject of the event. Preserve direction of causality where possible. "
    "Prefer relation labels from this compact schema: "
    "sanctions/restricts, threatens/attacks, negotiates/talks_with, supports/backs, "
    "opposes/criticizes, invests_in, regulates/investigates, affects_market, "
    "raises_risk_of, grants/approves, blocks/bans, competes_with, supplies/provides, "
    "depends_on. "
    "Do not use generic relations like says, is, has, or may unless the speech act "
    "itself is the newsworthy event."
)

# Rule-based relation normalization (lowercase keyword -> category label).
_RELATION_KEYWORDS: list[tuple[list[str], str]] = [
    (["sanction", "restrict", "embargo", "tariff", "prohibit", "blacklist"], "restricts/sanctions"),
    (["attack", "strike", "bomb", "missile", "war", "military", "drone", "invade", "conflict", "kill"], "military risk"),
    (["market", "stock", "price", "rally", "fall", "drop", "surge", "index", "yield", "inflation"], "market impact"),
    (["invest", "fund", "back", "finance", "acquire", "buy", "purchase", "stake"], "investment/backing"),
    (["sue", "lawsuit", "court", "indict", "convict", "legal", "probe", "investigat", "regulat"], "legal action"),
    (["talk", "negotiat", "meet", "diplom", "summit", "ceasefire", "deal", "treaty"], "diplomacy/talks"),
    (["approv", "grant", "permit", "license", "authoriz", "clear"], "approval/permission"),
    (["warn", "risk", "threat", "fear", "concern", "alert", "caution"], "warning/risk"),
    (["tech", "ai", "chip", "semiconductor", "software", "compet"], "technology competition"),
    (["support", "endorse", "ally", "backed", "campaign", "elect"], "political support"),
    (["fifa", "sport", "media right", "broadcast", "streaming"], "sports/media rights"),
    (["block", "ban", "halt", "veto"], "restricts/sanctions"),
    (["oppose", "critic", "reject", "denounc"], "warning/risk"),
    (["supply", "provide", "export", "ship"], "market impact"),
]

_VAGUE_PREDICATES = frozenset({
    "says", "said", "is", "are", "was", "were", "has", "have", "had",
    "may", "might", "be", "being", "do", "does", "did", "can", "could",
})

# Simple node-type inference rules.
_COUNTRY_TERMS = frozenset({
    "iran", "russia", "ukraine", "china", "israel", "usa", "u.s.", "us",
    "united states", "eu", "europe", "uk", "britain", "germany", "france",
    "japan", "india", "saudi", "venezuela", "mexico", "canada", "brazil",
    "taiwan", "korea", "turkey", "syria", "iraq", "yemen", "oman", "qatar",
})

_ASSET_TERMS = frozenset({
    "oil", "gold", "bitcoin", "crypto", "bond", "treasury", "dollar", "euro",
    "yen", "nasdaq", "s&p", "stocks", "equities", "rates", "inflation",
    "gdp", "cpi", "hormuz", "gas", "copper", "commodity", "commodities",
})

_ORG_SUFFIXES = (" inc", " corp", " ltd", " llc", " bank", " group", " holdings")
_ORG_TERMS = frozenset({
    "fed", "federal reserve", "ecb", "opec", "nato", "imf", "world bank",
    "openai", "nvidia", "microsoft", "google", "apple", "amazon", "meta",
    "fifa", "sec", "doj", "treasury", "congress", "parliament",
})

# Titles matching any of these are dropped as noise (single-stock plumbing, not macro news).
# These mirror the insider-transaction and earnings-transcript clusters that BERTopic surfaces.
NOISE_PATTERNS = [
    re.compile(r"\b(sells|buys|sold|bought|purchase[sd]?)\b.*\b(shares|stock|director|ceo|cfo|evp|coo|vp|officer)\b", re.I),
    re.compile(r"\b(director|ceo|cfo|evp|coo|vp|officer|president)\b.*\b(sells|buys|sold|bought)\b", re.I),
    re.compile(r"earnings call transcript", re.I),
    re.compile(r"\bq[1-4]\s?20\d\d\b.*transcript", re.I),
    re.compile(r"\btranscript\b.*\b(earnings|call)\b", re.I),
    re.compile(r"price target", re.I),
    re.compile(r"\b(initiated|reiterat\w+|maintain\w+|upgrad\w+|downgrad\w+)\b.*\b(buy|sell|hold|neutral|outperform|overweight|underweight|rating)\b", re.I),
    re.compile(r"\b(up|down)graded to\b", re.I),
]

# Trailing "source" cruft appended to headlines by aggregators, e.g. " - Reuters", " | CNBC".
SOURCE_SUFFIX = re.compile(r"\s*[-\u2013\u2014|]\s*[A-Z][A-Za-z0-9.&' ]{1,30}$")


def _discover_article_files() -> list[tuple[datetime, Path]]:
    """Return sorted (date, path) for data/articles_YYYY-MM-DD.csv (dated files only)."""
    pattern = re.compile(r"articles_(\d{4}-\d{2}-\d{2})\.csv$")
    out: list[tuple[datetime, Path]] = []
    if not DATA_DIR.exists():
        return out
    for p in DATA_DIR.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            continue
        out.append((d, p))
    out.sort(key=lambda x: x[0])
    return out


def _load_recent_articles(lookback_days: int = LOOKBACK_DAYS):
    """
    Load and concat the last `lookback_days` daily article CSVs (relative to the
    most recent available file). Returns (DataFrame, latest_date) or (None, None).
    """
    items = _discover_article_files()
    if not items:
        return None, None
    latest_date = items[-1][0]
    cutoff = latest_date - timedelta(days=lookback_days - 1)
    recent = [(d, p) for d, p in items if d >= cutoff]
    if not recent:
        return None, None
    frames = []
    for _, p in recent:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    return df, latest_date


def _clean_title(title: str) -> str:
    """Normalize a headline: collapse whitespace, strip trailing source name."""
    t = re.sub(r"\s+", " ", str(title)).strip()
    t = SOURCE_SUFFIX.sub("", t).strip()
    return t


def _is_noise(title: str) -> bool:
    return any(p.search(title) for p in NOISE_PATTERNS)


def _preprocess_headlines(df: pd.DataFrame) -> list[str]:
    """Dedupe, drop noise, strip cruft, cap to the most recent MAX_HEADLINES."""
    if df is None or df.empty or "title" not in df.columns:
        return []

    df = df.copy()
    # Recency: prefer 'published', fall back to 'timestamp'
    df["_recency"] = pd.to_datetime(df.get("published"), errors="coerce", utc=True)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["_recency"] = df["_recency"].fillna(ts)

    df["_clean"] = df["title"].map(_clean_title)
    df = df[df["_clean"].str.len() >= MIN_TITLE_LEN]
    df = df[~df["_clean"].map(_is_noise)]
    df = df.drop_duplicates(subset=["_clean"])

    df = df.sort_values("_recency", ascending=False, na_position="last")
    headlines = df["_clean"].head(MAX_HEADLINES).tolist()
    # Feed to the LLM oldest-first so temporal narrative reads naturally
    return list(reversed(headlines))


def _headlines_hash(headlines: list[str]) -> str:
    h = hashlib.sha256("\n".join(headlines).encode("utf-8")).hexdigest()
    return h[:16]


def _cache_path(date: datetime) -> Path:
    return KG_DIR / f"{KG_CACHE_PREFIX}{date.strftime('%Y-%m-%d')}.json"


def _image_path(date: datetime) -> Path:
    return KG_DIR / f"{KG_IMAGE_PREFIX}{date.strftime('%Y-%m-%d')}.png"


def _structured_json_path(date: datetime) -> Path:
    return KG_DIR / f"{KG_STRUCTURED_PREFIX}{date.strftime('%Y-%m-%d')}.json"


def _load_cached_graph(date: datetime, headlines_hash: str):
    """Return (entities, relations) if a fresh cache exists for this date + inputs, else None."""
    path = _cache_path(date)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("headlines_hash") != headlines_hash:
        return None
    entities = data.get("entities", [])
    relations = [tuple(r) for r in data.get("relations", []) if len(r) == 3]
    return entities, relations


def _save_cached_graph(date: datetime, headlines_hash: str, entities, relations) -> None:
    KG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "headlines_hash": headlines_hash,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entities": sorted(entities),
        "relations": [list(r) for r in relations],
    }
    _cache_path(date).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_graph(headlines: list[str]):
    """Run kg-gen over the headlines. Returns (entities, relations) or (None, None)."""
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY set; skipping news KG.")
        return None, None
    try:
        from kg_gen import KGGen
    except Exception as e:
        print(f"kg-gen import failed; skipping news KG: {e}")
        return None, None

    text = "\n".join(headlines)
    kg = KGGen(model=KG_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
    try:
        graph = kg.generate(
            input_data=text,
            chunk_size=KG_CHUNK_SIZE,
            cluster=True,
            context=KG_CONTEXT,
        )
    except Exception as e:
        print(f"kg-gen generation failed; skipping news KG: {e}")
        return None, None

    entities = list(graph.entities or [])
    relations = [tuple(r) for r in (graph.relations or []) if len(tuple(r)) == 3]
    return entities, relations


def _normalize_relation(pred: str) -> str:
    """Map noisy kg-gen predicates to readable macro categories."""
    raw = str(pred).strip()
    if not raw:
        return "other"
    lower = raw.lower()
    if lower in _VAGUE_PREDICATES or len(lower) > 80:
        for keywords, label in _RELATION_KEYWORDS:
            if any(kw in lower for kw in keywords):
                return label
        return "other"
    for keywords, label in _RELATION_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return label
    if len(lower.split()) <= 4 and lower not in _VAGUE_PREDICATES:
        return raw
    return "other"


def _infer_node_type(node: str) -> str:
    """Heuristic node typing for analytics export."""
    n = str(node).strip()
    lower = n.lower()
    if lower in _COUNTRY_TERMS or lower.endswith(" government"):
        return "country"
    if lower in _ASSET_TERMS or any(t in lower for t in ("price", "market", "index", "yield")):
        return "asset/market"
    if any(lower.endswith(s) for s in _ORG_SUFFIXES) or lower in _ORG_TERMS:
        return "organization"
    if any(t in lower for t in ("agenda", "policy", "inflation", "tariff", "war", "risk")):
        return "theme"
    words = n.split()
    if 1 <= len(words) <= 3 and n[0].isupper() and not any(w.lower() in _ORG_TERMS for w in words):
        if words[0].lower() in {"trump", "putin", "biden", "musk", "zelenskiy", "powell", "xi"}:
            return "person"
        if len(words) == 2 and all(w[0].isupper() for w in words if w):
            return "person"
    return "other"


def _build_weighted_graph(relations: list[tuple]) -> nx.DiGraph:
    """Aggregate parallel relations; preserve predicate evidence on each edge."""
    edge_data: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"raw_predicates": [], "predicates": Counter()}
    )
    nodes: set[str] = set()

    for subj, pred, obj in relations:
        subj = str(subj).strip()
        obj = str(obj).strip()
        pred = str(pred).strip()
        if not subj or not obj:
            continue
        nodes.add(subj)
        nodes.add(obj)
        norm = _normalize_relation(pred)
        bucket = edge_data[(subj, obj)]
        bucket["raw_predicates"].append(pred)
        bucket["predicates"][norm] += 1

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for (u, v), data in edge_data.items():
        preds: Counter = data["predicates"]
        dominant = preds.most_common(1)[0][0] if preds else "other"
        G.add_edge(
            u, v,
            weight=sum(preds.values()),
            label=dominant,
            predicates=dict(preds),
            raw_predicates=list(data["raw_predicates"]),
        )
    return G


def _detect_communities(G: nx.DiGraph) -> dict[str, int]:
    """Greedy modularity communities on the undirected support graph."""
    if G.number_of_nodes() == 0:
        return {}
    try:
        undirected = G.to_undirected()
        if undirected.number_of_edges() == 0:
            return {n: 0 for n in G.nodes()}
        communities = nx.algorithms.community.greedy_modularity_communities(undirected)
        mapping: dict[str, int] = {}
        for cid, members in enumerate(communities):
            for node in members:
                mapping[node] = cid
        for node in G.nodes():
            mapping.setdefault(node, 0)
        return mapping
    except Exception as e:
        print(f"Warning: community detection failed ({e}); assigning single community.")
        return {n: 0 for n in G.nodes()}


def _weighted_degree(G: nx.DiGraph, node: str) -> float:
    indeg = sum(d.get("weight", 1) for _, _, d in G.in_edges(node, data=True))
    outdeg = sum(d.get("weight", 1) for _, _, d in G.out_edges(node, data=True))
    return indeg + outdeg


def _compute_pagerank(G: nx.DiGraph) -> dict[str, float]:
    if G.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(G, weight="weight")
    except Exception as e:
        print(f"Warning: PageRank failed ({e}); falling back to weighted degree.")
        degs = {n: _weighted_degree(G, n) for n in G.nodes()}
        total = sum(degs.values()) or 1.0
        return {n: v / total for n, v in degs.items()}


def _select_salient_nodes(G: nx.DiGraph, top_n: int) -> list[str]:
    """Rank nodes by blended PageRank and weighted degree."""
    nodes = list(G.nodes())
    if len(nodes) <= top_n:
        return nodes

    pagerank = nx.get_node_attributes(G, "pagerank") or _compute_pagerank(G)
    wdeg = {n: _weighted_degree(G, n) for n in nodes}

    pr_vals = list(pagerank.values()) or [0.0]
    wd_vals = list(wdeg.values()) or [0.0]
    pr_min, pr_max = min(pr_vals), max(pr_vals)
    wd_min, wd_max = min(wd_vals), max(wd_vals)

    def _norm(val: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if val > 0 else 0.0
        return (val - lo) / (hi - lo)

    scored = []
    for n in nodes:
        score = (
            0.55 * _norm(pagerank.get(n, 0.0), pr_min, pr_max)
            + 0.45 * _norm(wdeg.get(n, 0.0), wd_min, wd_max)
        )
        scored.append((n, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:top_n]]


def _community_summaries(G: nx.DiGraph) -> list[dict]:
    """Per-community stats for structured JSON export."""
    by_comm: dict[int, list[str]] = defaultdict(list)
    for n, data in G.nodes(data=True):
        by_comm[int(data.get("community", 0))].append(n)

    pagerank = nx.get_node_attributes(G, "pagerank")
    summaries: list[dict] = []
    for cid in sorted(by_comm):
        nodes = by_comm[cid]
        node_set = set(nodes)
        internal = incoming = outgoing = 0
        for u, v, data in G.edges(data=True):
            w = int(data.get("weight", 1))
            u_in = u in node_set
            v_in = v in node_set
            if u_in and v_in:
                internal += w
            elif v_in and not u_in:
                incoming += w
            elif u_in and not v_in:
                outgoing += w
        boundary = incoming + outgoing
        n = len(nodes)
        internal_density = internal / max(n * (n - 1), 1)
        denom = internal + boundary
        selectivity = internal / denom if denom > 0 else 0.0
        top_nodes = sorted(nodes, key=lambda x: pagerank.get(x, 0.0), reverse=True)[:5]
        summaries.append({
            "id": cid,
            "label": " / ".join(top_nodes[:5]),
            "nodes": nodes,
            "top_nodes": top_nodes,
            "internal_edges": internal,
            "boundary_edges": boundary,
            "internal_density": round(internal_density, 4),
            "boundary_selectivity": round(selectivity, 4),
        })
    return summaries


def _prepare_graph(relations: list[tuple]) -> nx.DiGraph:
    """Build weighted graph with types, communities, and PageRank."""
    G = _build_weighted_graph(relations)
    if G.number_of_nodes() == 0:
        return G

    communities = _detect_communities(G)
    pagerank = _compute_pagerank(G)
    for node in G.nodes():
        G.nodes[node]["community"] = communities.get(node, 0)
        G.nodes[node]["type"] = _infer_node_type(node)
        G.nodes[node]["pagerank"] = pagerank.get(node, 0.0)
        G.nodes[node]["degree"] = G.degree(node)
    return G


def _gephi_safe_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Copy graph with scalar attributes only (GEXF/GraphML requirement)."""
    H = nx.DiGraph()
    for n, data in G.nodes(data=True):
        attrs: dict[str, str | int | float | bool] = {}
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                attrs[key] = json.dumps(val, ensure_ascii=False)
            elif isinstance(val, (str, int, float, bool)):
                attrs[key] = val
            elif val is not None:
                attrs[key] = str(val)
        H.add_node(n, **attrs)
    for u, v, data in G.edges(data=True):
        attrs: dict[str, str | int | float | bool] = {}
        for key, val in data.items():
            if key == "label":
                attrs["relation"] = val if isinstance(val, (str, int, float, bool)) else str(val)
            elif isinstance(val, (dict, list)):
                attrs[key] = json.dumps(val, ensure_ascii=False)
            elif isinstance(val, (str, int, float, bool)):
                attrs[key] = val
            elif val is not None:
                attrs[key] = str(val)
        H.add_edge(u, v, **attrs)
    return H


def _export_gephi(G: nx.DiGraph, date: datetime) -> tuple[Path, Path]:
    """Export graph to GEXF and GraphML for Gephi. Returns (gexf_path, graphml_path)."""
    KG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.strftime("%Y-%m-%d")
    gexf_path = KG_DIR / f"news_kg_{stamp}.gexf"
    graphml_path = KG_DIR / f"news_kg_{stamp}.graphml"
    H = _gephi_safe_graph(G)
    nx.write_gexf(H, gexf_path)
    nx.write_graphml(H, graphml_path)
    print(f"News KG Gephi GEXF: {gexf_path}")
    print(f"News KG Gephi GraphML: {graphml_path}")
    return gexf_path, graphml_path


def _export_graph_json(G: nx.DiGraph, date: datetime) -> Path:
    """Write rich structured JSON alongside the PNG."""
    KG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _structured_json_path(date)
    nodes = [
        {
            "id": n,
            "type": d.get("type", "other"),
            "community": int(d.get("community", 0)),
            "pagerank": round(float(d.get("pagerank", 0.0)), 6),
            "degree": int(d.get("degree", 0)),
        }
        for n, d in G.nodes(data=True)
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "relation": d.get("label", ""),
            "weight": int(d.get("weight", 1)),
            "predicates": d.get("predicates", {}),
            "raw_predicates": d.get("raw_predicates", []),
        }
        for u, v, d in G.edges(data=True)
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": date.strftime("%Y-%m-%d"),
        "nodes": nodes,
        "edges": edges,
        "communities": _community_summaries(G),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"News KG structured JSON: {out_path}")
    return out_path


def _render_graph(relations, date: datetime) -> Path | None:
    """Prepare graph analytics, export JSON, render salient subgraph to PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless PNG export only — avoid breaking notebook inline plots
    import matplotlib.pyplot as plt

    if not relations:
        return None

    G = _prepare_graph(relations)
    if G.number_of_edges() == 0:
        return None

    n_communities = len({d.get("community", 0) for _, d in G.nodes(data=True)})
    print(
        f"News KG graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"after aggregation ({n_communities} communities)."
    )

    _export_graph_json(G, date)

    salient = _select_salient_nodes(G, TOP_N_NODES)
    G_vis = G.subgraph(salient).copy()
    if G_vis.number_of_edges() == 0:
        print("Warning: salient subgraph has no edges; rendering full graph.")
        G_vis = G

    pagerank = nx.get_node_attributes(G_vis, "pagerank")
    pr_vals = list(pagerank.values()) or [0.0]
    pr_min, pr_max = min(pr_vals), max(pr_vals)

    def _pr_size(n: str) -> float:
        val = pagerank.get(n, 0.0)
        if pr_max <= pr_min:
            norm = 1.0
        else:
            norm = (val - pr_min) / (pr_max - pr_min)
        return 300 + 1200 * norm

    communities = nx.get_node_attributes(G_vis, "community")
    unique_comms = sorted(set(communities.values()))
    comm_colors = plt.cm.tab20([i % 20 for i in range(max(len(unique_comms), 1))])
    comm_to_color = {c: comm_colors[i] for i, c in enumerate(unique_comms)}
    node_colors = [comm_to_color.get(communities.get(n, 0), "#2b6cb0") for n in G_vis.nodes()]

    pos = nx.spring_layout(G_vis, k=0.9, iterations=100, seed=42, weight="weight")
    font_sizes = {n: min(12, 7 + int(pagerank.get(n, 0) * 100)) for n in G_vis.nodes()}

    fig, ax = plt.subplots(figsize=(18, 12))
    edge_widths = [0.8 + 1.2 * min(d.get("weight", 1), 5) for _, _, d in G_vis.edges(data=True)]
    nx.draw_networkx_edges(
        G_vis, pos, ax=ax, edge_color="#9aa5b1", width=edge_widths,
        arrows=True, arrowsize=10, alpha=0.6,
        connectionstyle="arc3,rad=0.06",
    )
    nx.draw_networkx_nodes(
        G_vis, pos, ax=ax,
        node_size=[_pr_size(n) for n in G_vis.nodes()],
        node_color=node_colors, alpha=0.85, linewidths=0.5, edgecolors="white",
    )
    for node, (x, y) in pos.items():
        ax.text(
            x, y, node, fontsize=font_sizes[node], ha="center", va="center",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    edge_labels = {
        (u, v): d["label"]
        for u, v, d in G_vis.edges(data=True)
        if d.get("label") and (d.get("weight", 1) >= 2 or d.get("label") != "other")
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G_vis, pos, ax=ax, edge_labels=edge_labels, font_size=6,
            font_color="#4a5568", label_pos=0.5, rotate=False,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
        )

    ax.set_title(
        f"News Landscape Knowledge Graph - {date.strftime('%Y-%m-%d')} "
        f"(last {LOOKBACK_DAYS} days, top {TOP_N_NODES} salient nodes)",
        fontsize=16,
    )
    ax.axis("off")
    fig.tight_layout()

    KG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _image_path(date)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"News KG PNG: {out_path}")
    return out_path


def build_news_kg(lookback_days: int = LOOKBACK_DAYS, force: bool = False) -> Path | None:
    """
    Build (or reuse) the news-landscape KG image for the most recent data.
    Returns a Path to the PNG, or None if there is not enough data/graph.
    """
    df, latest_date = _load_recent_articles(lookback_days)
    if df is None or latest_date is None:
        print("No recent article CSVs found; skipping news KG.")
        return None

    headlines = _preprocess_headlines(df)
    if len(headlines) < MIN_RELATIONS_FOR_CHART:
        print(f"Only {len(headlines)} headlines after cleaning; skipping news KG.")
        return None
    print(f"News KG: {len(headlines)} cleaned headlines over last {lookback_days} days.")

    h = _headlines_hash(headlines)
    image_path = _image_path(latest_date)

    entities, relations = (None, None)
    if not force:
        cached = _load_cached_graph(latest_date, h)
        if cached is not None:
            entities, relations = cached
            print(f"News KG: reusing cached graph for {latest_date.strftime('%Y-%m-%d')}.")
            if image_path.exists():
                return image_path

    if relations is None:
        entities, relations = _generate_graph(headlines)
        if relations is None:
            return None
        _save_cached_graph(latest_date, h, entities, relations)
        print(f"News KG: extracted {len(entities)} entities, {len(relations)} relations.")

    if len(relations) < MIN_RELATIONS_FOR_CHART:
        print(f"News KG: only {len(relations)} relations; too sparse to chart.")
        return None

    return _render_graph(relations, latest_date)


if __name__ == "__main__":
    path = build_news_kg()
    print(f"News KG image: {path}" if path else "No news KG produced.")

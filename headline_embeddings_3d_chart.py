"""
Build a 3D Plotly chart from headline embeddings over the last 72 hours.
- X/Y: UMAP 2D projection of headline embeddings
- Z: publication time
- Clusters are detected, named, ranked by importance; top clusters visible by default.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

load_dotenv()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
CACHE_DIR = OUTPUT_DIR / "cache"
CHART_PATH = OUTPUT_DIR / "headlines_3d_chart.html"
EMBED_CACHE_PATH = CACHE_DIR / "headline_embeddings_72h.npz"

HOURS_WINDOW = 72
EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
CLUSTER_LLM_MODEL = "gpt-4o-mini"
MAX_CHARS_FOR_EMBED = 8000
DEFAULT_TOP_VISIBLE_CLUSTERS = 5
MAX_HOURS_WITHIN_CLUSTER = 18
MIN_HEADLINES = 20
OPENAI_BATCH_SIZE = 100

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _parse_published(value) -> datetime | None:
    if pd.isna(value) or value == "N/A" or not str(value).strip():
        return None
    try:
        dt = parsedate_to_datetime(str(value))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def load_headlines_last_72h() -> pd.DataFrame:
    """Load deduplicated headlines from the rolling 72-hour window."""
    now = datetime.now()
    cutoff = now - timedelta(hours=HOURS_WINDOW)
    frames: list[pd.DataFrame] = []
    for day_offset in range(5):
        day = (now - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        path = DATA_DIR / f"articles_{day}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No article CSV files found in data/")

    df = pd.concat(frames, ignore_index=True)
    df["_pub_dt"] = df["published"].map(_parse_published)
    df["_scrape_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["_scrape_dt"].dt.tz is not None:
        df["_scrape_dt"] = df["_scrape_dt"].dt.tz_convert(None)
    df["_time"] = df["_pub_dt"].fillna(df["_scrape_dt"])
    df = df[df["_time"].notna() & (df["_time"] >= cutoff)].copy()
    df = df.sort_values("_time")
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    if len(df) < MIN_HEADLINES:
        raise ValueError(f"Need at least {MIN_HEADLINES} headlines in the last {HOURS_WINDOW}h; found {len(df)}")
    return df


def _content_hash(texts: list[str]) -> str:
    joined = "\n".join(texts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _embed_openai(texts: list[str]) -> np.ndarray:
    if not client:
        raise RuntimeError("OpenAI client unavailable")
    out: list[np.ndarray] = []
    for start in range(0, len(texts), OPENAI_BATCH_SIZE):
        batch = [t[:MAX_CHARS_FOR_EMBED] for t in texts[start : start + OPENAI_BATCH_SIZE]]
        response = client.embeddings.create(input=batch, model=EMBED_MODEL)
        order = {item.index: item.embedding for item in response.data}
        out.extend(np.array(order[i], dtype=np.float64) for i in range(len(batch)))
    return np.vstack(out)


def _embed_local(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def embed_headlines(texts: list[str]) -> tuple[np.ndarray, str]:
    """Return embeddings and a label describing the embedding backend."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = _content_hash(texts)
    if EMBED_CACHE_PATH.exists():
        cached = np.load(EMBED_CACHE_PATH, allow_pickle=False)
        if str(cached["digest"]) == digest:
            return cached["embeddings"], str(cached["backend"])

    if client:
        print(f"Embedding {len(texts)} headlines with OpenAI {EMBED_MODEL}...")
        embeddings = _embed_openai(texts)
        backend = EMBED_MODEL
    else:
        print(f"No OPENAI_API_KEY; embedding {len(texts)} headlines locally ({LOCAL_EMBED_MODEL})...")
        embeddings = _embed_local(texts)
        backend = LOCAL_EMBED_MODEL

    np.savez(EMBED_CACHE_PATH, embeddings=embeddings, digest=digest, backend=backend)
    return embeddings, backend


def _umap_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    import umap

    n = len(embeddings)
    n_neighbors = min(30, max(5, n // 20))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.05,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
    )
    return reducer.fit_transform(embeddings)


def _cluster_headlines(
    xy: np.ndarray,
    times: np.ndarray,
    max_hours_within_cluster: float = MAX_HOURS_WITHIN_CLUSTER,
    random_state: int = 42,
) -> np.ndarray:
    hours = (times - times.min()) / np.timedelta64(1, "h")
    hours = hours.astype(float)
    xy_scale = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) or 1.0)
    time_scale = xy_scale / max(max_hours_within_cluster, 1.0)
    features = np.column_stack([xy, hours * time_scale])
    n = len(features)
    n_clusters = min(12, max(3, n // 35))
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(features)


def _name_cluster_llm(titles: list[str]) -> str:
    if not client or not titles:
        return ""
    sample = "\n".join(f"- {t[:200]}" for t in titles[:25])
    prompt = (
        "These financial/macro news headlines cluster together. "
        "Reply with exactly 3 to 7 words naming the shared theme or event. "
        "If one event dominates, name that event. No quotes, no period.\n\n"
        f"{sample}"
    )
    try:
        response = client.chat.completions.create(
            model=CLUSTER_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        label = (response.choices[0].message.content or "").strip()
        return label[:60] if label else ""
    except Exception:
        return ""


NEWS_STOP_WORDS = {
    "reuters", "wsj", "ft", "times", "financial", "news", "says", "said",
    "report", "reports", "latest", "update", "breaking", "market", "markets",
    "stock", "stocks", "share", "shares", "wall", "street", "journal",
    "google", "cnbc", "investing", "com", "form", "holdings", "corp", "plc",
}


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", str(title)).strip()
    title = re.sub(r"^Exclusive:\s*", "", title, flags=re.I)
    return title


def _name_cluster_tfidf(titles: list[str]) -> str:
    if not titles:
        return "Cluster"
    cleaned = [_clean_title(t) for t in titles]
    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | NEWS_STOP_WORDS),
        max_features=2000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-']+\b",
    )
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return cleaned[0][:50]
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(scores, terms), reverse=True)
    top_terms = [term for _, term in ranked[:4] if len(term) > 2]
    if top_terms:
        return " · ".join(top_terms[:3])
    return cleaned[0][:50]


def _name_cluster_from_centroid(
    indices: np.ndarray,
    embeddings: np.ndarray,
    titles: list[str],
) -> str:
    cluster_emb = embeddings[indices]
    centroid = cluster_emb.mean(axis=0)
    dists = np.linalg.norm(cluster_emb - centroid, axis=1)
    rep_idx = int(dists.argmin())
    rep_title = _clean_title(titles[rep_idx])
    if len(rep_title) > 55:
        rep_title = rep_title[:52].rsplit(" ", 1)[0] + "..."
    return rep_title


def _cluster_names(
    titles_by_cluster: dict[int, list[str]],
    indices_by_cluster: dict[int, np.ndarray],
    embeddings: np.ndarray,
) -> dict[int, str]:
    names: dict[int, str] = {}
    for cluster_id, titles in titles_by_cluster.items():
        indices = indices_by_cluster[cluster_id]
        llm_name = _name_cluster_llm(titles)
        if llm_name:
            names[cluster_id] = llm_name
        else:
            tfidf_name = _name_cluster_tfidf(titles)
            rep_name = _name_cluster_from_centroid(indices, embeddings, titles)
            names[cluster_id] = tfidf_name if len(tfidf_name) > 12 else rep_name
    return names


def _cluster_importance(
    cluster_labels: np.ndarray,
    times: np.ndarray,
) -> dict[int, float]:
    now = times.max()
    span_hours = max((now - times.min()) / np.timedelta64(1, "h"), 1.0)
    scores: dict[int, float] = {}
    for cluster_id in sorted(set(cluster_labels.tolist())):
        mask = cluster_labels == cluster_id
        count = int(mask.sum())
        mean_hours_ago = float(((now - times[mask]) / np.timedelta64(1, "h")).mean())
        recency = 1.0 - min(mean_hours_ago / span_hours, 1.0)
        scores[cluster_id] = count * (0.55 + 0.45 * recency)
    return scores


def _format_hover(row: pd.Series) -> str:
    time_str = row["_time"].strftime("%Y-%m-%d %H:%M")
    title = str(row["title"]).replace("<", "&lt;")
    source = str(row.get("source", ""))
    return f"<b>{time_str}</b><br>{source}<br>{title}"


def build_headline_3d_chart(
    top_visible: int = DEFAULT_TOP_VISIBLE_CLUSTERS,
) -> Path:
    df = load_headlines_last_72h()
    texts = df["title"].astype(str).tolist()
    embeddings, backend = embed_headlines(texts)

    print("Running UMAP...")
    xy = _umap_2d(embeddings)
    times = df["_time"].to_numpy(dtype="datetime64[ns]")
    hours = (times - times.min()) / np.timedelta64(1, "h")
    hours = hours.astype(float)

    cluster_labels = _cluster_headlines(xy, times)
    titles_by_cluster: dict[int, list[str]] = {}
    indices_by_cluster: dict[int, np.ndarray] = {}
    for cluster_id in sorted(set(cluster_labels.tolist())):
        mask = cluster_labels == cluster_id
        indices = np.where(mask)[0]
        indices_by_cluster[cluster_id] = indices
        titles_by_cluster[cluster_id] = df.loc[mask, "title"].astype(str).tolist()

    print("Naming clusters...")
    cluster_names = _cluster_names(titles_by_cluster, indices_by_cluster, embeddings)
    importance = _cluster_importance(cluster_labels, times)
    ranked_clusters = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    rank_by_cluster = {cluster_id: rank + 1 for rank, (cluster_id, _) in enumerate(ranked_clusters)}
    visible_clusters = {cluster_id for cluster_id, _ in ranked_clusters[:top_visible]}

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939",
    ]

    fig = go.Figure()
    for cluster_id in sorted(set(cluster_labels.tolist())):
        mask = cluster_labels == cluster_id
        indices = np.where(mask)[0]
        rank = rank_by_cluster[cluster_id]
        name = cluster_names[cluster_id]
        legend_name = f"#{rank} {name} ({len(indices)})"
        color = palette[cluster_id % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=xy[mask, 0],
                y=xy[mask, 1],
                z=hours[mask],
                mode="markers",
                name=legend_name,
                legendgroup=f"cluster_{cluster_id}",
                showlegend=True,
                visible=True if cluster_id in visible_clusters else "legendonly",
                marker=dict(size=4, color=color, opacity=0.85),
                text=[df.iloc[i]["title"] for i in indices],
                customdata=[[df.iloc[i]["source"], df.iloc[i]["_time"].strftime("%Y-%m-%d %H:%M")] for i in indices],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[0]}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

    annotations = []
    for cluster_id in sorted(set(cluster_labels.tolist())):
        mask = cluster_labels == cluster_id
        if not mask.any():
            continue
        cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
        cz = float(hours[mask].mean())
        rank = rank_by_cluster[cluster_id]
        label = f"#{rank} {cluster_names[cluster_id]}"
        annotations.append(
            dict(
                x=cx,
                y=cy,
                z=cz,
                text=label,
                showarrow=False,
                font=dict(size=11, color="#111111"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="#666666",
                borderwidth=1,
                borderpad=3,
            )
        )

    window_start = pd.Timestamp(times.min()).to_pydatetime()
    window_end = pd.Timestamp(times.max()).to_pydatetime()
    tick_hours = np.linspace(hours.min(), hours.max(), 6)
    tick_labels = [
        (window_start + timedelta(hours=float(h))).strftime("%m-%d %H:%M")
        for h in tick_hours
    ]

    fig.update_layout(
        title=dict(
            text=(
                f"Headline embedding map (last {HOURS_WINDOW}h)<br>"
                f"<sup>{len(df)} headlines · {len(set(cluster_labels))} clusters · "
                f"top {top_visible} visible by default · embeddings: {backend}</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis_title="Embedding dim 1 (UMAP)",
            yaxis_title="Embedding dim 2 (UMAP)",
            zaxis_title="Time (hours from window start)",
            zaxis=dict(
                tickvals=tick_hours.tolist(),
                ticktext=tick_labels,
            ),
            annotations=annotations,
        ),
        legend=dict(
            title="Clusters (ranked by importance)",
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=240, t=80, b=0),
        width=1200,
        height=800,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        CHART_PATH,
        include_plotlyjs="cdn",
        config=dict(displayModeBar=True, responsive=True),
    )

    summary_path = OUTPUT_DIR / "headlines_3d_cluster_summary.txt"
    lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Window: {window_start} → {window_end}",
        f"Headlines: {len(df)}",
        f"Embedding backend: {backend}",
        "",
        "Clusters ranked by importance (count × recency):",
    ]
    for cluster_id, score in ranked_clusters:
        rank = rank_by_cluster[cluster_id]
        visible = "visible" if cluster_id in visible_clusters else "hidden (legend)"
        count = int((cluster_labels == cluster_id).sum())
        lines.append(
            f"  #{rank} [{visible}] score={score:.1f} n={count} — {cluster_names[cluster_id]}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Chart saved to {CHART_PATH}")
    print(f"Cluster summary saved to {summary_path}")
    return CHART_PATH


if __name__ == "__main__":
    build_headline_3d_chart()

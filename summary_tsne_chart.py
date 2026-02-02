"""
Build a 2D embedding chart from summary embeddings (UMAP or t-SNE).
1) Embed each summary in summaries/ (OpenAI embeddings).
2) Reduce to 2D with UMAP (default) or t-SNE.
3) Plot scatter (points labeled/colored by date), save PNG.
Designed to be called by send_off_email. Returns path to PNG or None.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SUMMARIES_DIR = Path("summaries")
CHART_FILENAME = "summary_tsne_chart.png"
DRIFT_HISTOGRAM_FILENAME = "summary_drift_histogram.png"
EMBEDDING_CACHE_PREFIX = "embedding_"
# OpenAI text-embedding-3-small accepts ~8191 tokens; ~4 chars/token → truncate long summaries
MAX_CHARS_FOR_EMBED = 30000
EMBED_MODEL = "text-embedding-3-small"
MIN_SUMMARIES_FOR_CHART = 3
DEFAULT_CHART_DAYS = 30  # How many days backward the chart history goes (from latest summary)
CLUSTER_LLM_MODEL = "gpt-4o-mini"  # Model for cluster theme naming
MIN_CLUSTER_RADIUS_FRACTION = 0.03  # Min circle radius as fraction of axis span (for tiny clusters)
DEFAULT_MAX_DAYS_WITHIN_CLUSTER = 7  # Max days between points in same cluster (time-coherent clustering)
MIN_POINTS_FOR_HULL_BOUNDARY = 6  # Use circle instead of hull when cluster has fewer points (avoids triangles)
CLUSTER_BOUNDARY_INSET = 0.82  # Shrink hull toward centroid to reduce overlap (1.0 = no shrink)
CLUSTER_OLDEST_ALPHA = 0.4  # Alpha for least recent cluster (overlapping ones fade)
CLUSTER_NEWEST_ALPHA = 1.0  # Alpha for most recent cluster
# Marker shape per cluster (cycle if more clusters than markers)
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "P", "H", "<", ">", "d", "X", "8"]


def _embedding_cache_path(date: datetime) -> Path:
    """Path to cache file for one date's embedding."""
    return SUMMARIES_DIR / f"{EMBEDDING_CACHE_PREFIX}{date.strftime('%Y-%m-%d')}.npz"


def _load_cached_embedding(date: datetime, summary_path: Path) -> np.ndarray | None:
    """Load cached embedding if it exists and summary file mtime is unchanged. Else return None."""
    cache_path = _embedding_cache_path(date)
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        emb = data["embedding"]
        stored_mtime = float(data["mtime"])
        if summary_path.stat().st_mtime != stored_mtime:
            return None
        return emb
    except Exception:
        return None


def _save_cached_embedding(date: datetime, summary_path: Path, embedding: np.ndarray) -> None:
    """Save embedding and summary mtime to cache file."""
    cache_path = _embedding_cache_path(date)
    np.savez(cache_path, embedding=embedding, mtime=summary_path.stat().st_mtime)


def _discover_summary_files() -> list[tuple[datetime, Path]]:
    """Return sorted list of (date, path) for summary_YYYY-MM-DD.txt (exclude other files)."""
    pattern = re.compile(r"summary_(\d{4}-\d{2}-\d{2})\.txt$")
    out = []
    if not SUMMARIES_DIR.exists():
        return out
    for p in SUMMARIES_DIR.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d")
                out.append((d, p))
            except ValueError:
                continue
    out.sort(key=lambda x: x[0])
    return out


def _load_embeddings(
    days: int | None,
    min_summaries: int = MIN_SUMMARIES_FOR_CHART,
) -> tuple[list[datetime], np.ndarray] | tuple[None, None]:
    """
    Load or compute embeddings for summaries. days: how many days back from latest (None = all history).
    min_summaries: minimum number of summaries required.
    Returns (dates_sorted, embeddings) or (None, None) if not enough summaries or no API key.
    """
    if not client:
        return (None, None)
    items = _discover_summary_files()
    if not items:
        return (None, None)
    if days is not None:
        latest_date = max(d for d, _ in items)
        cutoff_date = latest_date - timedelta(days=days)
        items = [(d, p) for d, p in items if d >= cutoff_date]
    if len(items) < min_summaries:
        return (None, None)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    date_to_embedding: dict[datetime, np.ndarray] = {}
    to_embed_dates: list[datetime] = []
    to_embed_texts: list[str] = []
    to_embed_paths: list[Path] = []
    for d, p in items:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = ""
        if not text.strip():
            continue
        cached = _load_cached_embedding(d, p)
        if cached is not None:
            date_to_embedding[d] = cached
        else:
            to_embed_dates.append(d)
            to_embed_texts.append(text)
            to_embed_paths.append(p)
    if to_embed_texts:
        print(f"Embedding {len(to_embed_texts)} new or changed summaries...")
        new_embeddings = _embed_texts(to_embed_texts)
        for i, (d, path) in enumerate(zip(to_embed_dates, to_embed_paths)):
            date_to_embedding[d] = new_embeddings[i]
            _save_cached_embedding(d, path, new_embeddings[i])
    dates_sorted = sorted(date_to_embedding.keys())
    if len(dates_sorted) < min_summaries:
        return (None, None)
    embeddings = np.array([date_to_embedding[d] for d in dates_sorted], dtype=np.float64)
    return (dates_sorted, embeddings)


MIN_SUMMARIES_FOR_DRIFT_HIST = 2  # need at least 2 days to get one day-to-day change


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Call OpenAI embeddings API; return (n, dim) array. Truncates each text to MAX_CHARS_FOR_EMBED."""
    truncated = [t[:MAX_CHARS_FOR_EMBED] for t in texts]
    response = client.embeddings.create(input=truncated, model=EMBED_MODEL)
    # Preserve order (API returns in order)
    order = {e.index: e.embedding for e in response.data}
    return np.array([order[i] for i in range(len(truncated))], dtype=np.float64)


def _umap_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Reduce (n, dim) to (n, 2) using UMAP. Suited to text embeddings (cosine, local structure)."""
    import umap
    n = len(embeddings)
    n_neighbors = min(15, max(2, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
    )
    return reducer.fit_transform(embeddings)


def _tsne_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Reduce (n, dim) to (n, 2) using t-SNE. Perplexity scales with n, capped at 50."""
    from sklearn.manifold import TSNE
    n = len(embeddings)
    perplexity = min(50, max(5, (n - 1) // 3)) if n > 1 else 1
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    return tsne.fit_transform(embeddings)


def _cluster_2d(xy: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Cluster 2D points with KMeans; return integer labels (0, 1, ...). Kept for backward compatibility."""
    from sklearn.cluster import KMeans
    n = len(xy)
    n_clusters = min(8, max(2, n // 4))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(xy)


def _cluster_2d_with_time(
    xy: np.ndarray,
    dates: list[datetime],
    max_days_within_cluster: int,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster in (x, y, time) so points in the same cluster are close in both embedding space and time."""
    from sklearn.cluster import KMeans
    n = len(xy)
    ordinals = np.array([d.toordinal() for d in dates], dtype=np.float64)
    time_span = ordinals.max() - ordinals.min()
    time_norm = (ordinals - ordinals.min()) / (time_span + 1e-9)
    xy_scale = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) or 1.0)
    time_scale = xy_scale * time_span / max_days_within_cluster
    third = time_norm * time_scale
    X3 = np.column_stack([xy, third])
    n_clusters = min(8, max(2, n // 4))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(X3)


def _cluster_boundary_xy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x, y) closed polygon tracing the cluster boundary (concave hull or convex hull)."""
    n = points.shape[0]
    if n < 2:
        return None
    # Prefer alpha shape (concave) for banana-like boundaries; fall back to convex hull
    try:
        import alphashape
        if n < 3:
            return None
        alpha = alphashape.optimizealpha(points)
        polygon = alphashape.alphashape(points, alpha)
        if polygon is None or polygon.is_empty:
            raise ValueError("empty polygon")
        if hasattr(polygon, "exterior") and polygon.exterior is not None:
            xs = np.array(polygon.exterior.coords.xy[0])
            ys = np.array(polygon.exterior.coords.xy[1])
        elif hasattr(polygon, "geoms"):
            # MultiPolygon: take largest
            polys = list(polygon.geoms)
            poly = max(polys, key=lambda p: p.area)
            xs = np.array(poly.exterior.coords.xy[0])
            ys = np.array(poly.exterior.coords.xy[1])
        else:
            raise ValueError("unexpected polygon type")
        return (xs, ys)
    except Exception:
        pass
    # Fallback: convex hull (scipy)
    from scipy.spatial import ConvexHull
    if n < 3:
        return None
    hull = ConvexHull(points)
    verts = hull.vertices
    x = np.append(points[verts, 0], points[verts[0], 0])
    y = np.append(points[verts, 1], points[verts[0], 1])
    return (x, y)


def _smooth_boundary_xy(x: np.ndarray, y: np.ndarray, num_points: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a closed polygon into a round curve via periodic spline. Returns (x, y) smooth closed curve."""
    from scipy.interpolate import splprep, splev
    # Remove duplicate closing point if present
    if len(x) > 1 and x[0] == x[-1] and y[0] == y[-1]:
        x, y = x[:-1], y[:-1]
    if len(x) < 3:
        return (x, y)
    # Parametric periodic spline (closed curve)
    tck, u = splprep([x, y], s=0.0, per=1)
    u_smooth = np.linspace(0, 1, num_points, endpoint=False)
    smooth = splev(u_smooth, tck)
    return (np.array(smooth[0]), np.array(smooth[1]))


def _get_cluster_theme_name(summary_texts: list[str]) -> str:
    """Collate summaries and ask OpenAI: what unifies this theme in 3-7 words? Returns short label or fallback."""
    if not summary_texts or not client:
        return "Cluster"
    collated = "\n\n---\n\n".join(t[:10000] for t in summary_texts[:20])  # cap tokens
    prompt = (
        "The following are daily financial/macro summaries that cluster together in a narrative map. "
        "What single theme or narrative unifies them? Reply with exactly 3 to 7 words, nothing else. If its driven by one event in partifular, please mention the exact event by name, no need to label it by an overarching category like geopolitics, jsut name the event"
        "No quotation marks, no period."
    )
    try:
        response = client.chat.completions.create(
            model=CLUSTER_LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt + "\n\n" + collated},
            ],
        )
        label = (response.choices[0].message.content or "Cluster").strip()
        return label[:60] if label else "Cluster"
    except Exception:
        return "Cluster"


# Max chars per summary when sending to LLM for axis interpretation (keep prompt small)
AXIS_INTERPRET_SUMMARY_CHARS = 10000
AXIS_INTERPRET_N_EXTREMES = 3


def _interpret_axis_llm(
    xy: np.ndarray,
    dates: list[datetime],
    axis_index: int,
    n_extremes: int = AXIS_INTERPRET_N_EXTREMES,
) -> str:
    """
    Ask LLM: summaries at one end of an axis vs the other — what theme separates them?
    Returns a short phrase for the axis (e.g. for xlabel/ylabel).
    """
    if not client or axis_index not in (0, 1):
        return ""
    coords = xy[:, axis_index]
    n = len(coords)
    if n < 2 * n_extremes:
        return ""
    # Indices of low end and high end of axis
    order = np.argsort(coords)
    low_indices = order[:n_extremes]
    high_indices = order[-n_extremes:]
    texts_low: list[str] = []
    texts_high: list[str] = []
    for i in low_indices:
        d = dates[i]
        path = SUMMARIES_DIR / f"summary_{d.strftime('%Y-%m-%d')}.txt"
        if path.exists():
            try:
                texts_low.append(path.read_text(encoding="utf-8")[:AXIS_INTERPRET_SUMMARY_CHARS])
            except Exception:
                pass
    for i in high_indices:
        d = dates[i]
        path = SUMMARIES_DIR / f"summary_{d.strftime('%Y-%m-%d')}.txt"
        if path.exists():
            try:
                texts_high.append(path.read_text(encoding="utf-8")[:AXIS_INTERPRET_SUMMARY_CHARS])
            except Exception:
                pass
    if not texts_low or not texts_high:
        return ""
    collated_low = "\n---\n".join(texts_low)
    collated_high = "\n---\n".join(texts_high)
    prompt = (
        "In a narrative-drift map, daily summaries are placed in 2D. "
        "Below are summaries at one end of an axis (Group A) and at the other end (Group B). "
        "In max 3-4 words, what theme or dimension separates Group A from Group B? Describe it such a way that the axis becomes readable (ie what a more negative or more positive number mean)"
        "Reply with only those 3-4 words, no preamble.\n\n"
        "Group A (one end):\n" + collated_low + "\n\n"
        "Group B (other end):\n" + collated_high
    )
    print(prompt)
    try:
        response = client.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        label = (response.choices[0].message.content or "").strip()
        return label[:80] if label else ""
    except Exception:
        return ""


def _get_axis_interpretations(xy: np.ndarray, dates: list[datetime]) -> tuple[str, str]:
    """Get LLM-based human-readable labels for axis 0 (x) and axis 1 (y). Returns (label_x, label_y)."""
    label_x = _interpret_axis_llm(xy, dates, 0) if client else ""
    label_y = _interpret_axis_llm(xy, dates, 1) if client else ""
    return (label_x, label_y)


def _plot_embedding_2d(
    xy: np.ndarray,
    dates: list[datetime],
    output_path: Path,
    method_name: str = "UMAP",
    title: str | None = None,
    cluster_labels: np.ndarray | None = None,
    cluster_theme_names: list[str] | None = None,
    axis_labels: tuple[str | None, str | None] | None = None,
) -> Path:
    """Scatter: color by date, marker per cluster; legend outside. axis_labels = (xlabel, ylabel) from LLM if available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    n = len(dates)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ordinals = np.array([d.toordinal() for d in dates])
    norm = Normalize(vmin=ordinals.min(), vmax=ordinals.max())
    cmap = plt.cm.viridis
    point_size = 60 if n < 50 else 40
    # Trajectory line (date order), drawn first so points sit on top
    ax.plot(xy[:, 0], xy[:, 1], "k-", alpha=0.15, zorder=0)
    # Scatter: different marker per cluster (draw oldest first so more recent clusters on top)
    if cluster_labels is not None and cluster_theme_names is not None:
        cluster_recency: list[tuple[int, str, int]] = []
        for k, theme_name in enumerate(cluster_theme_names):
            mask = cluster_labels == k
            if not np.any(mask):
                continue
            indices = np.where(mask)[0]
            recency_ord = max(dates[i].toordinal() for i in indices)
            cluster_recency.append((k, theme_name, recency_ord))
        recency_ords = [r for _, _, r in cluster_recency]
        min_ord = min(recency_ords) if recency_ords else 0
        max_ord = max(recency_ords) if recency_ords else 1
        ord_span = (max_ord - min_ord) or 1
        cluster_recency.sort(key=lambda x: x[2])
        for k, theme_name, recency_ord in cluster_recency:
            mask = cluster_labels == k
            points = xy[mask]
            alpha = CLUSTER_OLDEST_ALPHA + (CLUSTER_NEWEST_ALPHA - CLUSTER_OLDEST_ALPHA) * (
                (recency_ord - min_ord) / ord_span
            )
            marker = CLUSTER_MARKERS[k % len(CLUSTER_MARKERS)]
            ax.scatter(
                points[:, 0], points[:, 1],
                c=ordinals[mask], cmap=cmap, norm=norm, alpha=alpha, s=point_size, zorder=1,
                marker=marker, label=theme_name, edgecolors="none",
            )
    else:
        ax.scatter(
            xy[:, 0], xy[:, 1], c=ordinals, cmap=cmap, norm=norm, alpha=0.8, s=point_size, zorder=1,
        )
    # Mark latest/today's point with a red star
    ax.scatter(
        xy[-1, 0], xy[-1, 1],
        marker="*", s=400, c="red", edgecolors="darkred", linewidths=0.5, zorder=2, label="Latest",
    )
    if cluster_labels is not None and cluster_theme_names is not None:
        handles, labels_leg = ax.get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc="upper left", bbox_to_anchor=(0.88, 0.98), fontsize=7, framealpha=0.9)
    # Date labels at regular intervals (every 4th point plus first and last)
    indices_to_label = set(range(0, n, 4)) | {0, n - 1}
    for i in indices_to_label:
        if i >= n:
            continue
        ax.annotate(
            dates[i].strftime("%Y-%m-%d"),
            (xy[i, 0], xy[i, 1]),
            fontsize=6,
            alpha=0.9,
            xytext=(3, 3),
            textcoords="offset points",
        )
    if title is None:
        title = "Narrative Drift"
    subtitle = "Similar summaries sit closer"
    if axis_labels and (axis_labels[0] or axis_labels[1]):
        bits = []
        if axis_labels[0]:
            bits.append("X: " + axis_labels[0][:50])
        if axis_labels[1]:
            bits.append("Y: " + axis_labels[1][:50])
        if bits:
            subtitle += " - tentative interpretations of axes."
    else:
        subtitle += " Axes have no units."
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel(axis_labels[0] if axis_labels and axis_labels[0] else f"{method_name} 1")
    ax.set_ylabel(axis_labels[1] if axis_labels and axis_labels[1] else f"{method_name} 2")
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Date")
    tick_ordinals = np.linspace(ordinals.min(), ordinals.max(), 5)
    cbar.set_ticks(tick_ordinals)
    cbar.set_ticklabels([datetime.fromordinal(int(u)).strftime("%Y-%m-%d") for u in tick_ordinals])
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_summary_tsne_chart(
    reducer: str = "tsne",
    days: int = DEFAULT_CHART_DAYS,
    max_days_within_cluster: int = DEFAULT_MAX_DAYS_WITHIN_CLUSTER,
) -> Path | None:
    """
    Load all summary_YYYY-MM-DD.txt, embed with OpenAI (using cache when valid), reduce to 2D with UMAP or t-SNE, plot, save PNG.
    reducer: "umap" (default) or "tsne".
    days: how many days backward the chart history goes from the latest summary (default 30).
    max_days_within_cluster: max days between points in the same cluster for time-coherent clustering (default 7).
    Returns path to PNG or None if not enough summaries or no API key.
    """
    dates_sorted, embeddings = _load_embeddings(days)
    if dates_sorted is None or embeddings is None:
        print("Not enough summaries or no OPENAI_API_KEY; skipping summary embedding chart.")
        return None
    reducer_lower = reducer.lower()
    if reducer_lower == "umap":
        print("Running UMAP...")
        xy = _umap_2d(embeddings)
        method_name = "UMAP"
    elif reducer_lower == "tsne":
        print("Running t-SNE...")
        xy = _tsne_2d(embeddings)
        method_name = "t-SNE"
    else:
        raise ValueError(f"reducer must be 'umap' or 'tsne', got {reducer!r}")
    # Cluster in (x, y, time) so clusters group points close in both embedding space and time
    cluster_labels = _cluster_2d_with_time(xy, dates_sorted, max_days_within_cluster)
    n_clusters = int(np.max(cluster_labels)) + 1
    cluster_theme_names: list[str] = []
    for k in range(n_clusters):
        indices = np.where(cluster_labels == k)[0]
        texts = []
        for i in indices:
            d = dates_sorted[i]
            path = SUMMARIES_DIR / f"summary_{d.strftime('%Y-%m-%d')}.txt"
            if path.exists():
                try:
                    texts.append(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        name = _get_cluster_theme_name(texts) if texts else f"Cluster {k + 1}"
        cluster_theme_names.append(name)
    # LLM-based axis interpretations (human-readable labels for X and Y)
    print("Interpreting axes (LLM)...")
    axis_labels = _get_axis_interpretations(xy, dates_sorted)
    if axis_labels[0] or axis_labels[1]:
        print(f"  X: {axis_labels[0] or '(none)'}")
        print(f"  Y: {axis_labels[1] or '(none)'}")
    output_path = SUMMARIES_DIR / CHART_FILENAME
    _plot_embedding_2d(
        xy, dates_sorted, output_path,
        method_name=method_name,
        cluster_labels=cluster_labels,
        cluster_theme_names=cluster_theme_names,
        axis_labels=axis_labels,
    )
    return output_path


def _explain_drift_change(summary_before: str, summary_after: str, date_before: datetime, date_after: datetime) -> str:
    """Ask LLM to explain narrative change between two consecutive days."""
    if not client:
        return "N/A"
    prompt = f"""Compare these two daily macro summaries and explain what narrative shift occurred:

Summary {date_before.strftime('%Y-%m-%d')}:
{summary_before[:3000]}

Summary {date_after.strftime('%Y-%m-%d')}:
{summary_after[:3000]}

In one short phrase (max 10 words), describe the main narrative shift. No preamble."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def build_drift_histogram(top_n: int = 7) -> tuple[Path | None, list[dict]]:
    """
    For each consecutive day pair, compute 1 - cosine_similarity(emb_t, emb_{t+1}) as "drift".
    Plot histogram of drift over full history; mark today's drift and its z-score vs history.
    Also generate a ranking of top_n days with biggest drift and LLM explanations.
    Returns (path to PNG or None, rankings text).
    """
    dates_sorted, embeddings = _load_embeddings(None, min_summaries=MIN_SUMMARIES_FOR_DRIFT_HIST)
    if dates_sorted is None or embeddings is None or len(dates_sorted) < 2:
        return (None, [])
    # L2-normalize so cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    # Drift = 1 - cos_sim for each consecutive pair
    drifts = np.array([
        1.0 - np.dot(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ], dtype=np.float64)
    today_drift = float(drifts[-1])
    mean_drift = float(np.mean(drifts))
    std_drift = float(np.std(drifts))
    z_score = (today_drift - mean_drift) / std_drift if std_drift > 0 else 0.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(drifts, bins=min(40, max(5, len(drifts) // 2)), color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(today_drift, color="red", linewidth=2, label=f"Today's drift: {today_drift:.3f} ({z_score:+.2f}σ)")
    ax.set_xlabel("Daily narrative drift (1 − cos similarity)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of day-to-day narrative drift (full history)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARIES_DIR / DRIFT_HISTOGRAM_FILENAME
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    
    # Rank days by drift magnitude and generate explanations
    drift_with_dates = [
        (float(drifts[i]), dates_sorted[i], dates_sorted[i + 1], i)
        for i in range(len(drifts))
    ]
    drift_with_dates.sort(key=lambda x: x[0], reverse=True)
    
    rankings: list[dict] = []
    for rank, (drift_val, date_before, date_after, idx) in enumerate(drift_with_dates[:top_n], 1):
        path_before = SUMMARIES_DIR / f"summary_{date_before.strftime('%Y-%m-%d')}.txt"
        path_after = SUMMARIES_DIR / f"summary_{date_after.strftime('%Y-%m-%d')}.txt"
        summary_before = path_before.read_text(encoding="utf-8") if path_before.exists() else ""
        summary_after = path_after.read_text(encoding="utf-8") if path_after.exists() else ""
        
        explanation = _explain_drift_change(summary_before, summary_after, date_before, date_after)
        z = (drift_val - mean_drift) / std_drift if std_drift > 0 else 0.0
        rankings.append({
            "rank": rank,
            "date_before": date_before.strftime("%Y-%m-%d"),
            "date_after": date_after.strftime("%Y-%m-%d"),
            "drift": drift_val,
            "z_score": z,
            "explanation": explanation,
        })
    
    return (output_path, rankings)


if __name__ == "__main__":
    out = build_summary_tsne_chart(reducer="tsne")
    
    if out:
        print(f"Chart saved to {out}")
    else:
        print("Could not build summary embedding chart.")
    
    drift_chart, drift_rankings = build_drift_histogram()

    if drift_chart:
        print(f"Drift chart saved to {drift_chart}")
        if drift_rankings:
            for r in drift_rankings:
                print(f"{r['rank']}. {r['date_before']} → {r['date_after']} (drift: {r['drift']:.3f}, {r['z_score']:+.2f}σ)")
                print(f"   {r['explanation']}")
    else:
        print("Could not build histogram.")

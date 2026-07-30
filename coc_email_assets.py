"""
Core-of-cores email assets: sub-core compass/effect graphs + compact catalyst timeline.

Loads the latest saved sub-cores from ``kg/*_core_YYYY-MM-DD.json`` (written by
``core_of_cores_v2.ipynb`` / a future automated CoC run), regenerates Part-3c-style
plots, and builds a one-page top-N-per-theme catalyst timeline PNG.

Usage:
  python coc_email_assets.py              # write PNGs under summaries/
  from coc_email_assets import build_coc_email_assets
  assets = build_coc_email_assets()       # for send_off_email.py
"""

from __future__ import annotations

import re
import textwrap
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import vester_core as vc

KG_DIR = Path("kg")
OUT_DIR = Path("summaries")
TOP_N_PER_THEME = 4
RANK_BY = "impact_score"

# Skip top-level geo core and pairwise merge snapshots (contain "__").
_SKIP_LABEL_PREFIXES = ("geo_core", "core_of_cores", "geo_", "iran")


def _parse_core_stamp(path: Path) -> str | None:
    m = re.search(r"_core_(\d{4}-\d{2}-\d{2})\.json$", path.name)
    return m.group(1) if m else None


def _core_label_from_name(path: Path) -> str:
    m = re.search(r"^(.*)_core_\d{4}-\d{2}-\d{2}\.json$", path.name)
    return m.group(1) if m else path.stem


def _should_skip_label(label: str) -> bool:
    if "__" in label:
        return True
    if label in {"geo", "geo_v2", "iran"}:
        return True
    return any(label.startswith(p) for p in _SKIP_LABEL_PREFIXES)


def discover_latest_subcores(kg_dir: Path = KG_DIR) -> dict[str, Path]:
    """
    Return {label: path} for sub-cores from the newest stamp date only.
    Excludes geo_core / pair-merge cores so the email stays to one CoC run.
    """
    candidates: list[tuple[str, str, Path]] = []
    for path in kg_dir.glob("*_core_????-??-??.json"):
        stamp = _parse_core_stamp(path)
        if not stamp:
            continue
        label = _core_label_from_name(path)
        if _should_skip_label(label):
            continue
        candidates.append((stamp, label, path))

    if not candidates:
        return {}

    latest_stamp = max(s for s, _, _ in candidates)
    # Prefer the latest stamp; if it has only one core (partial run), fall back
    # to the most recent stamp that has >= 2 themes.
    by_stamp: dict[str, list[tuple[str, Path]]] = {}
    for stamp, label, path in candidates:
        by_stamp.setdefault(stamp, []).append((label, path))

    chosen_stamp = latest_stamp
    if len(by_stamp[latest_stamp]) < 2:
        multi = sorted(
            (s for s, items in by_stamp.items() if len(items) >= 2),
            reverse=True,
        )
        if multi:
            chosen_stamp = multi[0]

    return {label: path for label, path in sorted(by_stamp[chosen_stamp])}


def _parse_catalyst_date(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date):
        return val
    s = str(val).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def catalyst_interval(row, *, fallback: date | None = None):
    single = _parse_catalyst_date(row.get("date"))
    start = _parse_catalyst_date(row.get("start"))
    end = _parse_catalyst_date(row.get("end"))
    if single:
        return single, single
    if start and end:
        if end < start:
            start, end = end, start
        return start, end
    if start:
        return start, start + timedelta(days=6)
    if end:
        return end - timedelta(days=6), end
    if fallback:
        return fallback, fallback + timedelta(days=6)
    return None


def _wrap_label(text: str, width: int = 40) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return t if len(t) <= width else "\n".join(textwrap.wrap(t, width=width))


def _short_label(text: str, max_len: int = 40) -> str:
    t = (text or "").strip()
    return t if len(t) <= max_len else t[: max_len - 1].rstrip() + "…"


def collect_catalysts(sub_cores: dict[str, vc.CoreOntology]) -> pd.DataFrame:
    rows = []
    for iid, sub in sub_cores.items():
        cdf = sub.catalysts_df
        if cdf is None or cdf.empty:
            continue
        theme = sub.narrative_title or iid
        for _, r in cdf.iterrows():
            iv = catalyst_interval(r, fallback=sub.as_of)
            rows.append(
                {
                    "theme_id": iid,
                    "theme": theme,
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "type": r.get("type", ""),
                    "timeline_label": r.get("timeline_label", ""),
                    "start": iv[0] if iv else pd.NaT,
                    "end": iv[1] if iv else pd.NaT,
                    "probability": r.get("probability"),
                    "impact_score": r.get("impact_score"),
                    "expected_impact": r.get("expected_impact"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_subcore_graphs(
    sub_cores: dict[str, vc.CoreOntology],
    out_dir: Path = OUT_DIR,
    *,
    stamp: str | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Write one compass+effect PNG per sub-core (Part 3c layout).
    Returns list of (path, content_id, title).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or date.today().isoformat()
    results: list[tuple[Path, str, str]] = []

    for iid, sub in sub_cores.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
        vc.plot_compass(sub, show_catalysts=True, ax=axes[0])
        axes[0].set_title(f"{iid} — role compass", fontsize=11)
        vc.plot_effect_graph(
            sub,
            show_catalysts=True,
            ax=axes[1],
            **vc.SPARSE_EFFECT_GRAPH_KW,
        )
        axes[1].set_title(f"{iid} — effect system", fontsize=11)
        fig.suptitle(sub.narrative_title or iid, fontsize=12, y=1.01)
        fig.tight_layout()
        slug = vc._slugify(iid)
        path = out_dir / f"coc_subcore_{slug}_{stamp}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        cid = f"coc_subcore_{slug}"
        title = sub.narrative_title or iid
        results.append((path, cid, title))
        print(f"Wrote {path}")
    return results


def save_catalyst_timeline(
    catalysts: pd.DataFrame,
    theme_order: list[str],
    sub_cores: dict[str, vc.CoreOntology],
    out_dir: Path = OUT_DIR,
    *,
    stamp: str | None = None,
    top_n: int = TOP_N_PER_THEME,
    rank_by: str = RANK_BY,
) -> Path | None:
    """Compact one-page timeline of top-N catalysts per theme. Returns PNG path."""
    if catalysts is None or catalysts.empty:
        print("No catalysts for timeline.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or date.today().isoformat()

    top = (
        catalysts.sort_values(["theme_id", rank_by], ascending=[True, False])
        .groupby("theme_id", sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    plotted = top.dropna(subset=["start", "end"]).copy()
    if plotted.empty:
        print("No dated catalysts for timeline.")
        return None

    theme_colors = {iid: plt.cm.tab10(i % 10) for i, iid in enumerate(theme_order)}
    n_rows = len(plotted)
    fig_h = min(9.5, max(4.5, 0.42 * n_rows + 0.55 * plotted["theme_id"].nunique()))
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = 0.0
    yticks, ylabels = [], []
    bar_h = 0.62
    theme_spans = []

    for iid in theme_order:
        block = plotted[plotted["theme_id"] == iid]
        if block.empty:
            continue
        group_top = y
        theme = block.iloc[0]["theme"]
        for _, r in block.iterrows():
            start, end = r["start"], r["end"]
            start_num = mdates.date2num(start)
            width = max((end - start).days + 1, 1)
            ax.broken_barh(
                [(start_num, width)],
                (y - bar_h / 2, bar_h),
                facecolors=theme_colors[iid],
                edgecolors="black",
                linewidth=0.5,
                alpha=0.85,
            )
            yticks.append(y)
            ylabels.append(_wrap_label(str(r["title"]), width=42))
            y += 0.85
        theme_spans.append((group_top, y - 0.85, theme, iid))
        y += 0.35

    for top_y, bottom, theme, iid in theme_spans:
        mid = (top_y + bottom) / 2
        ax.text(
            -0.012,
            mid,
            _wrap_label(theme, width=18),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=theme_colors[iid],
        )
        ax.axhline(bottom + 0.2, color="#e0e0e0", linewidth=0.7, zorder=0)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.tick_params(axis="y", pad=4)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlabel("Date", fontsize=8)
    ax.set_title(
        f"Top {top_n} catalysts / theme (by {rank_by})",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    legend_ids = [iid for iid in theme_order if iid in set(plotted["theme_id"])]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=theme_colors[iid], alpha=0.85) for iid in legend_ids
    ]
    labels = [
        _short_label(sub_cores[iid].narrative_title or iid, 28) for iid in legend_ids
    ]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        title="Themes",
        title_fontsize=8,
    )
    fig.subplots_adjust(left=0.34, right=0.78, top=0.92, bottom=0.08)
    path = out_dir / f"coc_catalyst_timeline_{stamp}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def build_coc_email_assets(
    *,
    kg_dir: Path = KG_DIR,
    out_dir: Path = OUT_DIR,
    top_n: int = TOP_N_PER_THEME,
) -> dict:
    """
    Build CoC email images from the latest kg sub-core JSONs.

    Returns dict with:
      inline_images: list[(path, cid)]
      graphs_html: HTML snippet with <img cid:...> per sub-core
      timeline_html: HTML snippet for the timeline (or empty)
      stamp: date stamp used
      n_cores: int
    """
    paths = discover_latest_subcores(kg_dir)
    if not paths:
        print("No sub-core JSON files found in kg/; skipping CoC email assets.")
        return {
            "inline_images": [],
            "graphs_html": "",
            "timeline_html": "",
            "stamp": None,
            "n_cores": 0,
        }

    sub_cores: dict[str, vc.CoreOntology] = {}
    stamps = []
    for label, path in paths.items():
        try:
            core = vc.CoreOntology.load(path)
        except Exception as e:
            print(f"Skip {path.name}: {type(e).__name__}: {e}")
            continue
        sub_cores[label] = core
        stamps.append(_parse_core_stamp(path) or "")
    if not sub_cores:
        return {
            "inline_images": [],
            "graphs_html": "",
            "timeline_html": "",
            "stamp": None,
            "n_cores": 0,
        }

    stamp = max(s for s in stamps if s) if any(stamps) else date.today().isoformat()
    print(f"Building CoC email assets from {len(sub_cores)} sub-cores (stamp={stamp})")

    graph_assets = save_subcore_graphs(sub_cores, out_dir=out_dir, stamp=stamp)
    catalysts = collect_catalysts(sub_cores)
    timeline_path = save_catalyst_timeline(
        catalysts,
        theme_order=list(sub_cores.keys()),
        sub_cores=sub_cores,
        out_dir=out_dir,
        stamp=stamp,
        top_n=top_n,
    )

    inline_images: list[tuple[str, str]] = [(str(p), cid) for p, cid, _ in graph_assets]
    graphs_html_parts = [
        "<h2>Geopolitical sub-cores (role compass + effect system)</h2>",
        f"<p>{len(graph_assets)} themes from today's core-of-cores run ({stamp}).</p>",
    ]
    for path, cid, title in graph_assets:
        graphs_html_parts.append(f"<h3>{title}</h3>")
        graphs_html_parts.append(
            f'<p><img src="cid:{cid}" alt="{title}" '
            f'style="max-width:100%; height:auto;" /></p>'
        )
    graphs_html = "\n".join(graphs_html_parts)

    timeline_html = ""
    if timeline_path and timeline_path.exists():
        inline_images.append((str(timeline_path), "coc_catalyst_timeline"))
        timeline_html = (
            f"<h2>Catalyst timeline (top {top_n} per theme)</h2>"
            '<p><img src="cid:coc_catalyst_timeline" alt="Catalyst timeline" '
            'style="max-width:100%; height:auto;" /></p>'
        )

    return {
        "inline_images": inline_images,
        "graphs_html": graphs_html,
        "timeline_html": timeline_html,
        "stamp": stamp,
        "n_cores": len(sub_cores),
    }


if __name__ == "__main__":
    assets = build_coc_email_assets()
    print(
        f"Done: {assets['n_cores']} cores, "
        f"{len(assets['inline_images'])} images, stamp={assets['stamp']}"
    )

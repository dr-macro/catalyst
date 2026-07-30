"""
Refresh catalysts on the latest geopolitical sub-cores for the daily email.

Loads the most recent ``kg/*_core_*.json`` snapshot per theme, re-runs Vester
Step 7 against recent headlines, and writes new JSON + CSV files stamped with
``PIPELINE_DATE``. Core variables, roles, and effect graphs stay as in the
saved snapshot; only upcoming catalysts and the as-of date are updated.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from coc_email_assets import discover_latest_subcores
from summarize_headlines import get_pipeline_date
import vester_core as vc

LOOKBACK_DAYS = 7
KG_DIR = Path("kg")


def _load_dated_headlines(lookback_days: int = LOOKBACK_DAYS) -> tuple[list[tuple], date | None]:
    from build_news_kg import _load_recent_articles

    df, latest = _load_recent_articles(lookback_days)
    if df is None or latest is None or df.empty:
        return [], None
    if "title" not in df.columns:
        return [], latest

    dated: list[tuple] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        title = row.get("title")
        if title is None or (isinstance(title, float) and pd.isna(title)):
            continue
        text = str(title).strip()
        if not text or text in seen:
            continue
        seen.add(text)

        d = latest
        for col in ("timestamp", "published"):
            if col in row.index and pd.notna(row[col]):
                try:
                    d = pd.to_datetime(row[col]).date()
                    break
                except Exception:
                    pass
        src = row.get("source")
        if src is not None and not (isinstance(src, float) and pd.isna(src)):
            text = f"{src}: {text}"
        dated.append((d, text))
    return dated, latest


def refresh_subcores(*, stamp: str | None = None, kg_dir: Path = KG_DIR) -> int:
    paths = discover_latest_subcores(kg_dir)
    if not paths:
        print("No sub-core JSON files in kg/; skipping catalyst refresh.")
        return 0

    stamp = stamp or get_pipeline_date()
    as_of = date.fromisoformat(stamp)
    dated_headlines, latest = _load_dated_headlines(LOOKBACK_DAYS)
    if not dated_headlines:
        print("No recent headlines found; skipping catalyst refresh.")
        return 0
    if latest:
        as_of = max(as_of, latest)
    print(
        f"Refreshing catalysts for {len(paths)} sub-cores "
        f"(stamp={stamp}, {len(dated_headlines)} headlines, lookback={LOOKBACK_DAYS}d)"
    )

    n_saved = 0
    for label, path in paths.items():
        try:
            core = vc.CoreOntology.load(path)
            refreshed = vc.refresh_catalysts(
                core,
                dated_headlines=dated_headlines,
                as_of=as_of,
                lookback_days=LOOKBACK_DAYS,
            )
            out = refreshed.save(kg_dir, stamp=stamp)
            print(f"  {label}: {len(refreshed.catalysts_df)} catalysts -> {out.name}")
            n_saved += 1
        except Exception as e:
            print(f"  {label}: skip ({type(e).__name__}: {e})", file=sys.stderr)
    return n_saved


def main() -> int:
    stamp = os.environ.get("PIPELINE_DATE") or get_pipeline_date()
    n = refresh_subcores(stamp=stamp)
    if n == 0:
        print("CoC catalyst refresh produced no updated cores.")
    else:
        print(f"CoC catalyst refresh done: {n} cores written for {stamp}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Backfill missing daily summaries for the last N days when too few exist.

Runs automatically from run_daily_pipeline.py after summarize_headlines.py.
If fewer than MIN_SUMMARIES exist in the lookback window, generates any missing
summaries for days that have data/articles_YYYY-MM-DD.csv.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from summarize_headlines import summarize_for_date

BACKFILL_DAYS = 15
MIN_SUMMARIES = 15
SUMMARIES_DIR = Path("summaries")
DATA_DIR = Path("data")


def _day_start(dt: datetime | None = None) -> datetime:
    ref = dt or datetime.today()
    return ref.replace(hour=0, minute=0, second=0, microsecond=0)


def _dates_in_window(days: int, end: datetime) -> list[datetime]:
    """Inclusive window: end and the previous (days - 1) calendar days."""
    return [end - timedelta(days=offset) for offset in range(days - 1, -1, -1)]


def _summary_path(day: datetime) -> Path:
    return SUMMARIES_DIR / f"summary_{day.strftime('%Y-%m-%d')}.txt"


def _has_nonempty_summary(day: datetime) -> bool:
    path = _summary_path(day)
    if not path.exists():
        return False
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def count_summaries_in_window(days: int = BACKFILL_DAYS, end: datetime | None = None) -> int:
    end_day = _day_start(end)
    return sum(1 for d in _dates_in_window(days, end_day) if _has_nonempty_summary(d))


def missing_backfill_dates(days: int = BACKFILL_DAYS, end: datetime | None = None) -> list[str]:
    """
    Dates in the window with articles CSV but no non-empty summary yet.
    Returns YYYY-MM-DD strings oldest-first.
    """
    end_day = _day_start(end)
    missing: list[str] = []
    for day in _dates_in_window(days, end_day):
        date_str = day.strftime("%Y-%m-%d")
        csv_path = DATA_DIR / f"articles_{date_str}.csv"
        if not csv_path.exists():
            continue
        if _has_nonempty_summary(day):
            continue
        missing.append(date_str)
    return missing


def run_backfill_if_needed(
    days: int = BACKFILL_DAYS,
    min_summaries: int = MIN_SUMMARIES,
    end: datetime | None = None,
) -> int:
    """
    Backfill missing summaries when count in window < min_summaries.
    Returns 0 on success, 1 if any generation failed.
    """
    end_day = _day_start(end)
    count = count_summaries_in_window(days, end_day)
    if count >= min_summaries:
        print(
            f"Found {count} summaries in the last {days} days "
            f"(>= {min_summaries}); no backfill needed."
        )
        return 0

    to_generate = missing_backfill_dates(days, end_day)
    print(
        f"Only {count} summaries in the last {days} days "
        f"(need {min_summaries}). Backfilling {len(to_generate)} day(s)..."
    )
    if not to_generate:
        print("No article CSVs found for missing summary dates; nothing to generate.")
        return 0

    failed: list[str] = []
    for date_str in to_generate:
        print(f"\n--- Backfill summary for {date_str} ---")
        result = summarize_for_date(date_str, skip_if_exists=True)
        if result is None:
            failed.append(date_str)

    final_count = count_summaries_in_window(days, end_day)
    print(f"\nSummaries in window after backfill: {final_count}/{min_summaries} target.")
    if failed:
        print(f"Failed to generate summaries for: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    sys.exit(run_backfill_if_needed())


if __name__ == "__main__":
    main()

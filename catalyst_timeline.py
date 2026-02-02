"""
Build a catalyst table: dates as index, main regions as columns, events in cells.
Covers last N weeks + next N weeks (configurable for longer horizon).
Uses identified top upcoming catalysts (incoming_catalysts) + calendar CSV + past headlines.
For each catalyst we infer event length and exact date range (LLM), then fill the table.
Returns HTML table string for use in email (no image). Designed to be called by send_off_email.
"""

import html as html_module
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

CALENDAR_CSV = Path("calendar/tradingeconomics_calendar_master.csv")
DATA_DIR = Path("data")
INCOMING_CATALYSTS_DIR = Path("incoming_catalysts")
SUMMARIES_DIR = Path("summaries")

LLM_MODEL = "gpt-4o-mini"
LABEL_MAX_CHARS = 45

# Time window: past and future weeks (increase for longer horizon)
WEEKS_PAST = 2
WEEKS_FUTURE = 2

# Table: index = dates, columns = main geographical regions
REGIONS = ["US", "Euro Area", "China", "UK", "Japan", "Other"]

# Map calendar 2-letter Country codes to our region column
COUNTRY_TO_REGION = {
    "US": "US",
    "EA": "Euro Area",
    "EU": "Euro Area",
    "GB": "UK",
    "JP": "Japan",
    "CN": "China",
    "DE": "Euro Area",
    "FR": "Euro Area",
    "IT": "Euro Area",
    "ES": "Euro Area",
    "NL": "Euro Area",
    "AU": "Other",
    "CA": "Other",
    "IN": "Other",
    "BR": "Other",
    "RU": "Other",
    "SA": "Other",
    "KR": "Other",
    "CH": "Other",
    "SG": "Other",
    "ZA": "Other",
    "TR": "Other",
    "MX": "Other",
}


def _parse_llm_catalyst_json(response: str) -> list[dict]:
    """Parse LLM response into list of {label, start_date, end_date, region}."""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        out = []
        for x in data:
            if not isinstance(x, dict):
                continue
            label = (x.get("label") or "").strip()[:LABEL_MAX_CHARS]
            start_s = x.get("start_date") or x.get("date")
            end_s = x.get("end_date") or start_s
            region = (x.get("region") or "Other").strip()
            if region not in REGIONS:
                region = "Other"
            if not label or not start_s:
                continue
            out.append({"label": label, "start_date": start_s, "end_date": end_s or start_s, "region": region})
        return out
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, list):
                    return _parse_llm_catalyst_json(json.dumps(data))
            except json.JSONDecodeError:
                pass
        return []


def _load_headlines_past_weeks(days: int) -> str:
    """Load headlines from data/articles_*.csv for past N days."""
    today = datetime.now().date()
    lines = []
    for d in range(1, days + 1):
        dt = today - timedelta(days=d)
        csv_path = DATA_DIR / f"articles_{dt.strftime('%Y-%m-%d')}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "title" not in df.columns:
            continue
        if "source" in df.columns:
            headlines = [f"{row['source']}: {row['title']}" for _, row in df.iterrows()]
        else:
            headlines = [str(row["title"]) for _, row in df.iterrows()]
        if not headlines:
            continue
        lines.append(f"## {dt.strftime('%Y-%m-%d')}")
        for h in headlines[:80]:
            lines.append(f"- {h[:200]}")
        lines.append("")
    return "\n".join(lines)


def _load_upcoming_catalysts_text() -> str:
    """Load calendar_catalysts and news_catalysts for today (or most recent)."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    parts = []
    for name in ("calendar_catalysts", "news_catalysts"):
        path = INCOMING_CATALYSTS_DIR / f"{name}_{today_str}.txt"
        if not path.exists():
            candidates = sorted(INCOMING_CATALYSTS_DIR.glob(f"{name}_*.txt"), key=lambda p: p.name, reverse=True)
            path = candidates[0] if candidates else None
        if path and path.exists():
            try:
                parts.append(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return "\n\n".join(parts)


def _get_catalysts_with_date_ranges_from_llm(
    text: str,
    direction: str,
    context_year_month: str,
) -> list[dict]:
    """
    Call LLM to parse catalysts into label, start_date, end_date, region.
    direction: "past" or "upcoming". For each catalyst infer event length and exact date range (guess if needed).
    """
    if not client or not text.strip():
        return []
    prompt = (
        "From the following list of catalysts (each with optional date/time hints), output a JSON array. "
        "Process EVERY catalyst listed. For each one:\n"
        "1) Assign a short label (max 45 chars).\n"
        "2) Infer the exact date range: start_date and end_date in YYYY-MM-DD. "
        "If the text says a single day (e.g. 'Feb 6' or '2026-02-06'), use that for both. "
        "If it says a range (e.g. 'Feb 4–5', 'early Feb'), set start_date and end_date to that range (guess if needed). "
        "Use context year/month when only day or month is given: "
        + context_year_month
        + "\n"
        "3) Assign one region: US, Euro Area, China, UK, Japan, or Other (for multi-region or unclear).\n"
        "Return a JSON array only, no other text. Each item: "
        '{"label": "Short event label", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "region": "US"|"Euro Area"|"China"|"UK"|"Japan"|"Other"}.'
    )
    if direction == "past":
        prompt += " These are PAST events that already happened; infer dates from the daily headlines."
    else:
        prompt += " These are UPCOMING events; infer dates from the text (e.g. 'Feb 6', 'Feb 4–5', 'early Feb')."
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt + "\n\n" + text[:12000]}],
            temperature=0.2,
        )
        content = (response.choices[0].message.content or "").strip()
        return _parse_llm_catalyst_json(content)
    except Exception as e:
        print(f"LLM catalysts error ({direction}): {e}")
        return []


def _load_calendar_events_in_range(
    start_date: datetime,
    end_date: datetime,
    major_regions_only: bool = True,
) -> list[dict]:
    """Load events from calendar CSV in [start_date, end_date]. Return list of {label, start_date, end_date, region}."""
    if not CALENDAR_CSV.exists():
        return []
    try:
        df = pd.read_csv(CALENDAR_CSV)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime"])
        df["date"] = df["Datetime"].dt.date
        start_d = start_date.date() if hasattr(start_date, "date") else start_date
        end_d = end_date.date() if hasattr(end_date, "date") else end_date
        mask = (df["date"] >= start_d) & (df["date"] <= end_d)
        df = df.loc[mask]
        if "Country" in df.columns:
            df["region"] = df["Country"].map(lambda c: COUNTRY_TO_REGION.get(str(c).strip().upper()[:2], "Other"))
            if major_regions_only:
                df = df[df["region"].isin(REGIONS)]
        else:
            df["region"] = "Other"
        if "Event" not in df.columns:
            return []
        out = []
        for _, row in df.iterrows():
            label = (str(row["Event"]).strip()[:LABEL_MAX_CHARS]) if pd.notna(row["Event"]) else ""
            if not label:
                continue
            d = row["date"]
            if hasattr(d, "strftime"):
                date_str = d.strftime("%Y-%m-%d")
            else:
                date_str = str(d)
            country = str(row.get("Country", "")).strip() if "Country" in row.index else ""
            out.append({
                "label": label,
                "start_date": date_str,
                "end_date": date_str,
                "region": row["region"],
                "country": country,
            })
        return out
    except Exception as e:
        print(f"Calendar load error: {e}")
        return []


def _normalize_event(
    e: dict,
    window_start: datetime,
    window_end: datetime,
    *,
    country: str | None = None,
    is_ranked: bool = False,
) -> list[tuple[datetime, str, str, bool]]:
    """
    Convert one event dict to list of (date, region, label_display, is_ranked) for each day in range.
    Clamp to window. For Euro Area/Other, append country code to label when available.
    """
    out = []
    try:
        start = datetime.strptime(e["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(e["end_date"], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return []
    if end < start:
        end = start
    region = e.get("region") or "Other"
    if region not in REGIONS:
        region = "Other"
    label = (e.get("label") or "").strip()[:LABEL_MAX_CHARS]
    if not label:
        return []
    country = country or e.get("country") or ""
    if country and region in ("Euro Area", "Other"):
        label_display = f"{label} ({country})"
    else:
        label_display = label
    is_ranked = is_ranked or e.get("ranked", False)
    w_start = window_start.date() if hasattr(window_start, "date") else window_start
    w_end = window_end.date() if hasattr(window_end, "date") else window_end
    d = start
    while d <= end:
        if w_start <= d <= w_end:
            out.append((datetime.combine(d, datetime.min.time()), region, label_display, is_ranked))
        d += timedelta(days=1)
    return out


def _build_events_table(
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Build table: index = dates (daily), columns = REGIONS, values = newline-separated event labels per cell.
    """
    today = datetime.now()
    context_ym = today.strftime("%Y-%m")

    # Past: headlines + calendar
    past_start = today - timedelta(weeks=WEEKS_PAST)
    headlines_text = _load_headlines_past_weeks(WEEKS_PAST * 7)
    past_llm = _get_catalysts_with_date_ranges_from_llm(headlines_text, "past", context_ym)
    past_calendar = _load_calendar_events_in_range(past_start, today)

    # Upcoming: incoming_catalysts text + calendar
    upcoming_text = _load_upcoming_catalysts_text()
    future_llm = _get_catalysts_with_date_ranges_from_llm(upcoming_text, "upcoming", context_ym)
    future_end = today + timedelta(weeks=WEEKS_FUTURE)
    future_calendar = _load_calendar_events_in_range(today, future_end)

    # Collect all (date, region, label, is_ranked) in window; tag LLM events as ranked
    window_start = start_date
    window_end = end_date
    all_entries: list[tuple[datetime, str, str, bool]] = []
    for e in past_llm:
        all_entries.extend(_normalize_event(e, window_start, window_end, is_ranked=True))
    for e in past_calendar:
        all_entries.extend(_normalize_event(e, window_start, window_end, country=e.get("country")))
    for e in future_llm:
        all_entries.extend(_normalize_event(e, window_start, window_end, is_ranked=True))
    for e in future_calendar:
        all_entries.extend(_normalize_event(e, window_start, window_end, country=e.get("country")))

    # Dedupe by (date, region, label) and build cell lists of (label, is_ranked)
    from collections import defaultdict
    cell: dict[tuple[datetime, str], list[tuple[str, bool]]] = defaultdict(list)
    seen = set()
    for dt, region, label, is_ranked in all_entries:
        d = dt.date()
        key = (d, region, label)
        if key in seen:
            continue
        seen.add(key)
        cell[(d, region)].append((label, is_ranked))

    # Only include dates that have at least one event (keeps table compact)
    dates_with_events = sorted({date_val for (date_val, _) in cell.keys()})
    if not dates_with_events:
        return pd.DataFrame(index=[], columns=REGIONS)

    # DataFrame: index = dates with events, columns = REGIONS; cells = list of (label, is_ranked)
    df = pd.DataFrame(index=dates_with_events, columns=REGIONS, dtype=object)
    for (date_val, region), items in cell.items():
        if date_val in df.index and region in df.columns:
            df.at[date_val, region] = items
    df = df.fillna("")
    for d in df.index:
        for c in df.columns:
            if df.at[d, c] == "":
                df.at[d, c] = []
    return df


def _table_df_to_html(table_df: pd.DataFrame, today: datetime) -> str:
    """Turn the catalyst DataFrame into a compact HTML table; today row and ranked events highlighted."""
    col_labels = list(table_df.columns)
    rows = []
    # Compact styling so table doesn't dominate the email
    table_style = "border-collapse:collapse; width:100%; font-size:11px;"
    rows.append("<tr><th>Date</th>" + "".join(f"<th>{html_module.escape(c)}</th>" for c in col_labels) + "</tr>")
    today_d = today.date()
    for d in table_df.index:
        date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        row_attr = ' style="background-color:#fff3cd;"' if d == today_d else ""
        cells = [f"<td{row_attr}>{html_module.escape(date_str)}</td>"]
        for c in col_labels:
            val = table_df.loc[d, c]
            if isinstance(val, list) and val:
                parts = []
                for item in val:
                    if isinstance(item, tuple) and len(item) >= 2:
                        label, is_ranked = item[0], item[1]
                    else:
                        label, is_ranked = str(item), False
                    escaped = html_module.escape(label)
                    if is_ranked:
                        parts.append(f'<span style="background:#e7f3ff; padding:0 2px; font-weight:bold;">{escaped}</span>')
                    else:
                        parts.append(escaped)
                cell_content = "<br />".join(parts)
                cells.append(f"<td{row_attr}>{cell_content}</td>")
            elif isinstance(val, str) and val:
                escaped = html_module.escape(val).replace("\n", "<br />")
                cells.append(f"<td{row_attr}>{escaped}</td>")
            else:
                cells.append(f"<td{row_attr}>&nbsp;</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<table border="1" cellpadding="4" cellspacing="0" style="{table_style}">'
        + "".join(rows)
        + "</table>"
    )


def build_catalyst_timeline() -> str | None:
    """
    Build catalyst table (dates x regions) and return HTML table string for the email.
    Return None if no data.
    """
    today = datetime.now()
    start_date = today - timedelta(weeks=WEEKS_PAST)
    end_date = today + timedelta(weeks=WEEKS_FUTURE)

    table_df = _build_events_table(start_date, end_date)
    if table_df.empty or table_df.index.empty:
        print("No catalyst table data; skipping.")
        return None

    html = _table_df_to_html(table_df, today)
    print("Catalyst table built (HTML).")
    return html


if __name__ == "__main__":
    out = build_catalyst_timeline()
    if out:
        print(f"Done: HTML length {len(out)}")
    else:
        print("Could not build catalyst table.")

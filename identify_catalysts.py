"""
Rank top upcoming catalysts from headlines + economic calendar (LLM).

Writes:
  incoming_catalysts/news_catalysts_<PIPELINE_DATE>.txt
  incoming_catalysts/calendar_catalysts_<PIPELINE_DATE>.txt

Uses PIPELINE_DATE when set (CI), and falls back to yesterday's articles CSV
when today's file is missing or still empty — otherwise delayed midnight runs
skip news catalysts and the email section comes out blank.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from summarize_headlines import get_csv_path, get_pipeline_date

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

no_of_top_catalysts = "5"
# Cap headlines fed to the ranking prompt (full-day CSVs can be huge).
MAX_HEADLINES_FOR_PROMPT = 400
LLM_MODEL = os.getenv("CATALYST_LLM_MODEL", "gpt-5-mini")


def load_headlines():
    csv_path, file_date = get_csv_path()
    if not csv_path:
        print("No headlines file found for pipeline date (checked today and yesterday)")
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Headlines CSV is empty: {csv_path}")
        return []
    if "source" in df.columns and "title" in df.columns:
        headlines = [f"{row['source']}: {row['title']}" for _, row in df.iterrows()]
    elif "title" in df.columns:
        headlines = [str(row["title"]) for _, row in df.iterrows()]
    else:
        headlines = [str(row) for _, row in df.iterrows()]
    headlines = [h for h in headlines if h and str(h).strip()]
    print(f"Loaded {len(headlines)} headlines from {csv_path} (file date {file_date})")
    if len(headlines) > MAX_HEADLINES_FOR_PROMPT:
        print(f"Truncating to most recent {MAX_HEADLINES_FOR_PROMPT} headlines for ranking prompt")
        headlines = headlines[-MAX_HEADLINES_FOR_PROMPT:]
    return headlines


def load_calendar_events():
    now = datetime.now()
    end_date = now + timedelta(days=14)
    calendar_path = "calendar/tradingeconomics_calendar_master.csv"
    if not os.path.exists(calendar_path):
        print(f"No calendar file found: {calendar_path}")
        return []
    events = []
    with open(calendar_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                event_date = datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if now < event_date <= end_date:
                events.append(
                    f"{event_date.strftime('%Y-%m-%d %H:%M')} | {row.get('Country', '')} | {row.get('Event', '')}"
                )
    return events


def build_news_prompt(headlines):
    return (
        "You are a financial analyst AI. "
        "We define catalysts as events which have not happened yet and which will or could bring about information that is new to the market."
        f"Given the following news headlines, identify and rank the top {no_of_top_catalysts} upcoming catalysts by their likely importance for markets and geopolitics over the next 2 weeks. "
        "Briefly explain in 1 short line your reasoning for each (if possible include a date of the next event or a timeline). Output a ranked list.\n\n"
        "Of coure it is hard to rank these, given the volume and interconnectivity but do it anyway."
        f"Top {no_of_top_catalysts} upcoming catalysts by importance:\n"
        + "\n".join(f"- {h}" for h in headlines)
        + "\n\nRanked News Catalysts:"
    )


def build_calendar_prompt(events):
    return (
        "You are a financial analyst AI. "
        "We define catalysts as events or data releases which have not happened yet and which will or could bring about information that is new to the market."
        f"Given the following upcoming calendar events over the next 2 weeks, identify and rank the top {no_of_top_catalysts} catalysts by their likely importance as  for markets and geopolitics. "
        "Briefly explain in 1 short line your reasoning for each. Output a ranked list.\n\n"
        "Of coure it is hard to rank these, given the volume and interconnectivity but do it anyway."
        f"Top {no_of_top_catalysts} upcoming catalysts by importance:\n"
        + "\n".join(f"- {e}" for e in events)
        + "\n\nRanked Calendar Catalysts:"
    )


def get_gpt_ranking(prompt: str) -> str:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot rank catalysts.")
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(f"LLM returned empty content (model={LLM_MODEL})")
    text = content.strip()
    if not text:
        raise RuntimeError(f"LLM returned blank ranking (model={LLM_MODEL})")
    return text


def _write_catalyst_file(path: str, text: str) -> None:
    if not (text or "").strip():
        print(f"Refusing to write empty catalyst file: {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {path} ({len(text)} chars)")


if __name__ == "__main__":
    today_str = get_pipeline_date()
    print(f"Pipeline date: {today_str}")
    headlines = load_headlines()
    events = load_calendar_events()

    os.makedirs("incoming_catalysts", exist_ok=True)

    if headlines:
        try:
            print("Prompting LLM to rank news catalysts...")
            news_ranking = get_gpt_ranking(build_news_prompt(headlines))
            _write_catalyst_file(f"incoming_catalysts/news_catalysts_{today_str}.txt", news_ranking)
        except Exception as e:
            print(f"ERROR ranking news catalysts: {type(e).__name__}: {e}")
            raise
    else:
        print("No headlines found for ranking.")

    if events:
        try:
            print("Prompting LLM to rank calendar catalysts...")
            calendar_ranking = get_gpt_ranking(build_calendar_prompt(events))
            _write_catalyst_file(
                f"incoming_catalysts/calendar_catalysts_{today_str}.txt", calendar_ranking
            )
        except Exception as e:
            print(f"ERROR ranking calendar catalysts: {type(e).__name__}: {e}")
            raise
    else:
        print("No calendar events found for ranking.")

from dotenv import load_dotenv
load_dotenv()

from email.utils import parsedate_to_datetime
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import wrap

def get_pipeline_date() -> str:
    """Report date for the daily pipeline (pinned at pipeline start in CI)."""
    return os.environ.get("PIPELINE_DATE") or datetime.now().strftime("%Y-%m-%d")


def _csv_has_rows(path: str) -> bool:
    """True if path exists and has at least one data row (not just a header)."""
    if not path or not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, nrows=1)
        return len(df) > 0
    except Exception:
        return False


def get_csv_path():
    """Resolve articles CSV for the pipeline date, falling back to the previous day.

    Returns (path, file_date_str) where file_date_str is the date of the CSV
    actually used (important when CI starts after midnight and today's file
    is missing or still empty).
    """
    try:
        ref = datetime.strptime(get_pipeline_date(), "%Y-%m-%d")
    except ValueError:
        ref = datetime.today()
    today_str = ref.strftime("%Y-%m-%d")
    csv_path = f"data/articles_{today_str}.csv"
    if _csv_has_rows(csv_path):
        return csv_path, today_str
    yest_str = (ref - timedelta(days=1)).strftime("%Y-%m-%d")
    yest_path = f"data/articles_{yest_str}.csv"
    if _csv_has_rows(yest_path):
        reason = "missing" if not os.path.exists(csv_path) else "empty"
        print(f"WARNING: CSV for {today_str} is {reason}; using {yest_path}")
        return yest_path, yest_str
    return None, today_str

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _parse_published(s):
    """Parse RSS 'published' string to datetime; return None on failure."""
    if pd.isna(s) or s == "N/A" or not str(s).strip():
        return None
    try:
        return parsedate_to_datetime(str(s))
    except (ValueError, TypeError):
        return None


def load_headlines(path):
    """Load headlines from CSV, ordered by publication date, each line prefixed with publication date."""
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if df.empty or "published" not in df.columns:
        return [f"{row['source']}: {row['title']}" for _, row in df.iterrows()]
    # Parse publication date and sort ascending (oldest first)
    df["_pub_dt"] = df["published"].map(_parse_published)
    df = df.sort_values("_pub_dt", na_position="last")
    # Format: date | source: title (use YYYY-MM-DD HH:MM for readability)
    lines = []
    for _, row in df.iterrows():
        pub = row["_pub_dt"]
        date_str = pub.strftime("%Y-%m-%d %H:%M") if pd.notna(pub) else "unknown"
        lines.append(f"[{date_str}] {row['source']}: {row['title']}")
    return lines

def chunk_headlines(headlines, max_chars=20000000):
    chunks, chunk, total_chars = [], [], 0
    for h in headlines:
        if total_chars + len(h) > max_chars:
            chunks.append("\n".join(chunk))
            chunk, total_chars = [h], len(h)
        else:
            chunk.append(h)
            total_chars += len(h)
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks

def summarize_chunk(text, as_of: datetime | None = None):
    now = (as_of or datetime.now()).strftime("%Y-%m-%d %H:%M")
    prompt = (
        f"Current date and time (at time of summarization): {now}\n\n"
        "You are a macro hedge fund analyst; please avoid using spammy words. "
        "Below are news headlines ordered by publication date; each line starts with [YYYY-MM-DD HH:MM]. "
        "Please summarize the main themes in brief numbered bullet points (1 short comprehensive sentence to explain). "
        "Include the numbers and hard facts from the headlines as smaller sub-bullet points (put them ONLY under the corresponding bullet points and just state them as briefly as possible, no need to write sentences, just put the facts), but no need to name the newssource. "
        "At the end provide a description (in 2-3 lines) of the current macro/markets regime we are in.\n\n"
        f"{text}\n\nSummarize:"
    )
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def overarching_summary(text, as_of: datetime | None = None):
    now = (as_of or datetime.now()).strftime("%Y-%m-%d %H:%M")
    prompt = (
        f"Current date and time (at time of summarization): {now}\n\n"
        "You are a macro hedge fund analyst; here are a few summaries of different sets of headlines. "
        "Please provide an overarching summary with numbers if useful. "
        "Please summarize the main themes in brief numbered bullet points (1 short comprehensive sentence to explain). "
        "Include the numbers and hard facts from the headlines as smaller sub-bullet points (put them ONLY under the corresponding bullet points and just state them as briefly as possible, no need to write sentences, just put the facts), but no need to name the newssource. "
        "At the end provide a description (in 2-3 lines) of the current macro/markets regime we are in.\n\n"
        f"{text}\n\nSummary:"
    )
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def summarize_for_date(date_str: str, *, skip_if_exists: bool = True) -> str | None:
    """
    Generate summaries/summary_<date_str>.txt from data/articles_<date_str>.csv.
    Returns path to summary file, or None if skipped or failed.
    """
    summary_path = Path("summaries") / f"summary_{date_str}.txt"
    if skip_if_exists and summary_path.exists():
        try:
            if summary_path.read_text(encoding="utf-8").strip():
                print(f"Summary already exists for {date_str}; skipping.")
                return str(summary_path)
        except OSError:
            pass

    csv_path = f"data/articles_{date_str}.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: No articles CSV for {date_str}.")
        return None

    headlines = load_headlines(csv_path)
    if not headlines:
        print(f"ERROR: No headlines in {csv_path}.")
        return None

    try:
        as_of = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        as_of = None

    print(f"Loaded {len(headlines)} headlines from {csv_path}.")
    chunks = chunk_headlines(headlines)
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summaries.append(summarize_chunk(chunk, as_of=as_of))

    full_summary = "\n\n".join(summaries)
    if len(summaries) > 1:
        print("Creating overarching summary...")
        summary_of_summaries = overarching_summary(full_summary, as_of=as_of)
    else:
        print("No need for creating overarching summary...")
        summary_of_summaries = full_summary

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_of_summaries, encoding="utf-8")
    print(f"OK: Summary saved to {summary_path}")
    return str(summary_path)


def summarize_all():
    csv_path, date_str = get_csv_path()
    if not csv_path:
        print("ERROR: No headlines found for today or yesterday.")
        return None
    # date_str is the CSV's own date (may be yesterday after midnight fallback)
    return summarize_for_date(date_str, skip_if_exists=False)

if __name__ == "__main__":
    summarize_all()



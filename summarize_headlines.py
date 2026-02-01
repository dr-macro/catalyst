from dotenv import load_dotenv
load_dotenv()

from email.utils import parsedate_to_datetime
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime, timedelta
from textwrap import wrap

# --- NEW: helper to get today or fallback to yesterday ---
def get_csv_path():
    today = datetime.today()
    # path for today
    today_str = today.strftime("%Y-%m-%d")
    csv_path = f"data/articles_{today_str}.csv"
    if os.path.exists(csv_path):
        return csv_path, today_str
    # fallback to yesterday
    yesterday = today - timedelta(days=1)
    yest_str = yesterday.strftime("%Y-%m-%d")
    yest_path = f"data/articles_{yest_str}.csv"
    if os.path.exists(yest_path):
        print(f"⚠️ No CSV for today, using yesterday’s file {yest_path}")
        return yest_path, yest_str
    # nothing found
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

def summarize_chunk(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
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

def overarching_summary(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
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

def summarize_all():
    csv_path, date_str = get_csv_path()
    headlines = load_headlines(csv_path)
    if not headlines:
        print("❌ No headlines found for today or yesterday.")
        return

    print(f"Loaded {len(headlines)} headlines from {csv_path}.")
    chunks = chunk_headlines(headlines)
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summaries.append(summarize_chunk(chunk))

    full_summary = "\n\n".join(summaries)
    if len(summaries) > 1:
        print("Creating overarching summary...")
        summary_of_summaries = overarching_summary(full_summary)
    else:
        print("No need for creating overarching summary...")
        summary_of_summaries = full_summary
        

    # Save under summaries/summary_<date-used>.txt
    summary_path = f"summaries/summary_{date_str}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_of_summaries)

    print(f"✅ Summary saved to {summary_path}")
    return summary_path, summary_of_summaries

if __name__ == "__main__":
    summarize_all()



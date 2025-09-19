from openai import OpenAI
import os
import pandas as pd
from datetime import datetime, timedelta   # <-- add timedelta
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

def load_headlines(path):
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return [f"{row['source']}: {row['title']}" for _, row in df.iterrows()]

def chunk_headlines(headlines, max_chars=250000):
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
    prompt = (
        "you are a macro hedge fund analyst; below are newsheadlines from today. "
        "please summarize the main themes. dont mind if you also include some numbers "
        "from the headlines; make it quite detailed and comprehensive. "
        "at the end provide a description of the current macro/markets regime we are in.\n\n"
        f"{text}\n\nSummarize:"
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def overarching_summary(text):
    prompt = (
        "you are a macro hedge fund analyst; here are a few summaries of different sets of headlines; "
        "please provide an overarching summary with numbers if useful. "
        "at the end provide a description of the current macro/markets regime we are in.\n\n"
        f"{text}\n\nSummary:"
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
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
    summary_of_summaries = overarching_summary(full_summary)

    # Save under summaries/summary_<date-used>.txt
    summary_path = f"summaries/summary_{date_str}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_of_summaries)

    print(f"✅ Summary saved to {summary_path}")
    return summary_path, summary_of_summaries

if __name__ == "__main__":
    summarize_all()

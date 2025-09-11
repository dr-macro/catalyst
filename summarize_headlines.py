from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
from textwrap import wrap

# Load CSV
today = datetime.today().strftime("%Y-%m-%d")
csv_path = f"data/articles_{today}.csv"

csv_path = f"articles_2025-09-10.csv"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_headlines(csv_path):
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    return [f"{row['source']}: {row['title']}" for _, row in df.iterrows()]

def chunk_headlines(headlines, max_chars=250000):
    """Chunk headlines into groups within token limits (~4 chars per token)."""
    chunks = []
    chunk = []
    total_chars = 0
    for h in headlines:
        if total_chars + len(h) > max_chars:
            chunks.append("\n".join(chunk))
            chunk = [h]
            total_chars = len(h)
        else:
            chunk.append(h)
            total_chars += len(h)
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks

def summarize_chunk(text):
    prompt = (
        "you are a macro hedge fund analyst; below are newsheadlines from today. please summarize the main themes."
        "dont mind if you also include some numbers from the headlines; make it quite detailed and comprehensive. \n\n"
        f"{text}\n\nSummarize:"
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content


def overarching_summary(text):
    prompt = (
        "you are a macro hedge fund analyst; here are a few summaries of different sets of newheadlines; please provide an overarching summary dont mind if you also include some numbers from the summaries; make it quite detailed and comprehensive.\n\n"
        f"{text}\n\nSummary:"
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content
    
def summarize_all():
    headlines = load_headlines(csv_path)
    if not headlines:
        print("❌ No headlines found.")
        return

    print(f"Loaded {len(headlines)} headlines.")
    chunks = chunk_headlines(headlines)
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarize_chunk(chunk)
        summaries.append(summary)

    full_summary = "\n\n".join(summaries)
    
    summary_of_summaries = overarching_summary(full_summary)
    
    summary_path = f"data/summary_{today}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_of_summaries)

    print(f"✅ Summary saved to {summary_path}")
    return summary_path, summary_of_summaries

if __name__ == "__main__":
    summarize_all()

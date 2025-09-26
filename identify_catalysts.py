import os
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

no_of_top_catalysts = "5"

def load_headlines():
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = f"data/articles_{today}.csv"
    if not os.path.exists(csv_path):
        print(f"No headlines file found for today: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    # Adjust column names as needed
    if 'source' in df.columns and 'title' in df.columns:
        return [f"{row['source']}: {row['title']}" for _, row in df.iterrows()]
    elif 'title' in df.columns:
        return [row['title'] for _, row in df.iterrows()]
    else:
        return [str(row) for _, row in df.iterrows()]

def load_calendar_events():
    today = datetime.now()
    end_date = today + timedelta(days=14)
    calendar_path = "calendar/tradingeconomics_calendar_master.csv"
    if not os.path.exists(calendar_path):
        print(f"No calendar file found: {calendar_path}")
        return []
    events = []
    with open(calendar_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                event_date = datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if today.date() <= event_date.date() <= end_date.date():
                events.append(
                    f"{event_date.strftime('%Y-%m-%d %H:%M')} | {row.get('Country', '')} | {row.get('Event', '')}"
                )
    return events

def build_news_prompt(headlines):
    prompt = (
        "You are a financial analyst AI. "
        "We define catalysts as events which have not happened yet and which will or could bring about information that is new to the market."
        f"Given the following news headlines, identify and rank the top {no_of_top_catalysts} upcoming catalysts by their likely importance for markets and geopolitics over the next 2 weeks. "
        "Briefly explain in 1 short line your reasoning for each (if possible include a date of the next event or a timeline). Output a ranked list.\n\n"
        "Of coure it is hard to rank these, given the volume and interconnectivity but do it anyway."
        f"Top {no_of_top_catalysts} upcoming catalysts by importance:\n"
        + "\n".join(f"- {h}" for h in headlines) +
        "\n\nRanked News Catalysts:"
    )
    return prompt

def build_calendar_prompt(events):
    prompt = (
        "You are a financial analyst AI. "
        "We define catalysts as events or data releases which have not happened yet and which will or could bring about information that is new to the market."
        f"Given the following upcoming calendar events over the next 2 weeks, identify and rank the top {no_of_top_catalysts} catalysts by their likely importance as  for markets and geopolitics. "
        "Briefly explain in 1 short line your reasoning for each. Output a ranked list.\n\n"
        "Of coure it is hard to rank these, given the volume and interconnectivity but do it anyway."
        f"Top {no_of_top_catalysts} upcoming catalysts by importance:\n"
        + "\n".join(f"- {e}" for e in events) +
        "\n\nRanked Calendar Catalysts:"
    )
    return prompt

def get_gpt_ranking(prompt):
    response = client.chat.completions.create(
        model="gpt-5-mini",  # Use a model you have access to
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    headlines = load_headlines()
    events = load_calendar_events()

    os.makedirs("incoming_catalysts", exist_ok=True)

    # News headlines ranking
    if headlines:
        news_prompt = build_news_prompt(headlines)
        print("Prompting LLM to rank news catalysts...")
        news_ranking = get_gpt_ranking(news_prompt)
        news_path = f"incoming_catalysts/news_catalysts_{today_str}.txt"
        with open(news_path, "w", encoding="utf-8") as f:
            f.write(news_ranking)
        print(f"News catalysts ranking saved to {news_path}")
    else:
        print("No headlines found for ranking.")

    # Calendar events ranking
    if events:
        calendar_prompt = build_calendar_prompt(events)
        print("Prompting LLM to rank calendar catalysts...")
        calendar_ranking = get_gpt_ranking(calendar_prompt)
        calendar_path = f"incoming_catalysts/calendar_catalysts_{today_str}.txt"
        with open(calendar_path, "w", encoding="utf-8") as f:
            f.write(calendar_ranking)
        print(f"Calendar catalysts ranking saved to {calendar_path}")
    else:
        print("No calendar events found for ranking.")
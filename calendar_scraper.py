import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re
import unicodedata
import os

OUTPUT_FILE = "calendar/tradingeconomics_calendar_master.csv"

def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()

def parse_event_row(row, current_date):
    try:
        tds = row.find_all("td")
        if len(tds) < 9:
            return None

        time_clean = " ".join(tds[0].get_text(strip=True).split())
        country_cell = tds[1].find("td", class_="calendar-iso")
        country = country_cell.text.strip() if country_cell else ""
        event_cell = tds[4].find("a", class_="calendar-event")
        event = event_cell.text.strip() if event_cell else ""
        event_href = event_cell["href"] if event_cell else ""
        actual = clean_text(tds[5].get_text())
        previous = clean_text(tds[6].get_text())
        consensus = clean_text(tds[7].get_text())
        forecast = clean_text(tds[8].get_text())

        # Combine date and time
        try:
            datetime_str = f"{current_date} {time_clean}"
            event_datetime = datetime.strptime(datetime_str, "%A %B %d %Y %I:%M %p")
        except Exception:
            event_datetime = None

        return {
            "Datetime": event_datetime,
            "Country": country,
            "Event": event,
            "URL": f"https://tradingeconomics.com{event_href}",
            "Actual": actual,
            "Previous": previous,
            "Consensus": consensus,
            "Forecast": forecast
        }
    except Exception as e:
        print(f"Error parsing row: {e}")
        return None

def scrape_te_calendar():
    url = "https://tradingeconomics.com/calendar"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    calendar = []
    current_date = None

    for row in soup.find_all("tr"):
        th = row.find("th", colspan="3")
        if th:
            current_date = " ".join(th.text.strip().split())
            continue

        if row.find_all("td"):
            event_data = parse_event_row(row, current_date)
            if event_data:
                calendar.append(event_data)

    return pd.DataFrame(calendar)

def append_to_csv(df_new, csv_path):
    """
    Append only new rows (unique by Datetime + Country + Event).
    """
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, parse_dates=["Datetime"])
    else:
        df_existing = pd.DataFrame()

    if not df_existing.empty:
        # unify datetime type
        df_existing["Datetime"] = pd.to_datetime(df_existing["Datetime"], errors="coerce")
        df_new["Datetime"] = pd.to_datetime(df_new["Datetime"], errors="coerce")
        merged = pd.concat([df_existing, df_new], ignore_index=True)
        merged.drop_duplicates(subset=["Datetime", "Country", "Event"], inplace=True)
    else:
        merged = df_new

    merged.to_csv(csv_path, index=False)
    print(f"✅ CSV updated. Total rows: {len(merged)}")

if __name__ == "__main__":
    df = scrape_te_calendar()
    if df.empty:
        print("No data extracted.")
    else:
        append_to_csv(df, OUTPUT_FILE)

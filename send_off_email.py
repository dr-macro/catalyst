import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import csv
import os
from dotenv import load_dotenv

load_dotenv()
smtp_token = os.getenv("SMTP_EMAIL_TOKEN")
destinataries = os.getenv("DESTINATARIES")
TO_EMAIL = [email.strip() for email in destinataries.split(",") if email.strip()]

# --- CONFIGURATION ---
SMTP_SERVER = "smtp.protonmail.ch"  # Proton Mail Bridge default
SMTP_PORT = 587           # Proton Mail Bridge default
USERNAME = "mm@macrodoomscrolling.org"
PASSWORD = smtp_token
FROM_EMAIL = "mm@macrodoomscrolling.org"
TO_EMAIL = TO_EMAIL

def get_calendar_of_the_day():
    today = datetime.now()
    end_date = today + timedelta(days=14)
    calendar_path = "calendar/tradingeconomics_calendar_master.csv"
    if not os.path.exists(calendar_path):
        return f"Today's date: {today.strftime('%Y-%m-%d')}\nNo calendar file found."
    events = []
    with open(calendar_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse the date from the Datetime column
            try:
                event_date = datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if today.date() <= event_date.date() <= end_date.date():
                events.append({
                    "Date": event_date.strftime("%Y-%m-%d"),
                    "Time": event_date.strftime("%H:%M"),
                    "Country": row.get("Country", ""),
                    "Event": row.get("Event", ""),
                    "Actual": row.get("Actual", ""),
                    "Previous": row.get("Previous", ""),
                    "Consensus": row.get("Consensus", ""),
                    "Forecast": row.get("Forecast", "")
                })
    if not events:
        return f"Today's date: {today.strftime('%Y-%m-%d')}\nNo events found in calendar."

    # Build a simple text table
    header = f"{'Date':<10} {'Time':<5} {'Country':<6} {'Event':<40} {'Actual':<10} {'Prev':<10} {'Cons':<10} {'Fcst':<10}"
    lines = [header, "-" * len(header)]
    for e in events:
        lines.append(
            f"{e['Date']:<10} {e['Time']:<5} {e['Country']:<6} {e['Event'][:38]:<40} {e['Actual']:<10} {e['Previous']:<10} {e['Consensus']:<10} {e['Forecast']:<10}"
        )
    return f"Calendar: Today + Next 2 Weeks\n" + "\n".join(lines)

def get_calendar_of_the_day_html():
    today = datetime.now()
    end_date = today + timedelta(days=14)
    calendar_path = "calendar/tradingeconomics_calendar_master.csv"
    if not os.path.exists(calendar_path):
        return f"<p>Today's date: {today.strftime('%Y-%m-%d')}<br>No calendar file found.</p>"
    events = []
    with open(calendar_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                event_date = datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if today.date() <= event_date.date() <= end_date.date():
                events.append({
                    "Date": event_date.strftime("%Y-%m-%d"),
                    "Time": event_date.strftime("%H:%M"),
                    "Country": row.get("Country", ""),
                    "Event": row.get("Event", ""),
                    "Actual": row.get("Actual", ""),
                    "Previous": row.get("Previous", ""),
                    "Consensus": row.get("Consensus", ""),
                    "Forecast": row.get("Forecast", "")
                })
    if not events:
        return f"<p>Today's date: {today.strftime('%Y-%m-%d')}<br>No events found in calendar.</p>"

    # Build HTML table
    html = [
        "<h2>Calendar: Today + Next 2 Weeks</h2>",
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>Date</th><th>Time</th><th>Country</th><th>Event</th><th>Actual</th><th>Prev</th><th>Cons</th><th>Fcst</th></tr>"
    ]
    for e in events:
        html.append(
            f"<tr><td>{e['Date']}</td><td>{e['Time']}</td><td>{e['Country']}</td>"
            f"<td>{e['Event']}</td><td>{e['Actual']}</td><td>{e['Previous']}</td>"
            f"<td>{e['Consensus']}</td><td>{e['Forecast']}</td></tr>"
        )
    html.append("</table>")
    return "\n".join(html)

def send_email(subject, body):
    for recipient in TO_EMAIL:
        msg = MIMEMultipart()
        msg["From"] = FROM_EMAIL
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(USERNAME, PASSWORD)
            server.send_message(msg, from_addr=FROM_EMAIL, to_addrs=[recipient])
        print(f"✅ Email sent to {recipient}.")

if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Load summary
    with open(f"summaries/summary_{today_str}.txt", "r", encoding="utf-8") as f:
        summary = f.read()

    # Load catalysts if available
    news_catalysts_path = f"incoming_catalysts/news_catalysts_{today_str}.txt"
    calendar_catalysts_path = f"incoming_catalysts/calendar_catalysts_{today_str}.txt"
    news_catalysts = ""
    calendar_catalysts = ""

    if os.path.exists(news_catalysts_path):
        with open(news_catalysts_path, "r", encoding="utf-8") as f:
            news_catalysts = f.read()
    if os.path.exists(calendar_catalysts_path):
        with open(calendar_catalysts_path, "r", encoding="utf-8") as f:
            calendar_catalysts = f.read()

    calendar_html = get_calendar_of_the_day_html()

    email_body = (
        "<p>Greetings,</p>"
        "<p>Please enjoy the automated doom scrolling content.</p>"
        f"<h2>LLM Politics-Headlines Catalysts</h2><pre>{news_catalysts}</pre>"
        f"<h2>LLM Eco-Calendar Catalysts</h2><pre>{calendar_catalysts}</pre>"
        f"<h2>Summary of Today's Headlines</h2><pre>{summary[:10000]}</pre>"
        f"{calendar_html}<hr>"
        "<p>Best regards,<br>MacroDoomscrolling</p>"
    )

    send_email(
        subject="Daily Macro Summary, Catalysts, and Calendar",
        body=email_body
    )
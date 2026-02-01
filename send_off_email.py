import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
import csv
import os
from pathlib import Path
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

def send_email(subject, body, inline_images=None):
    """
    inline_images: optional list of (file_path, content_id) e.g. [("summaries/summary_tsne_chart.png", "summary_tsne_chart")]
    """
    for recipient in TO_EMAIL:
        msg = MIMEMultipart()
        msg["From"] = FROM_EMAIL
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        if inline_images:
            for file_path, cid in inline_images:
                path = Path(file_path)
                if path.exists():
                    with open(path, "rb") as f:
                        img = MIMEImage(f.read())
                    img.add_header("Content-ID", f"<{cid}>")
                    img.add_header("Content-Disposition", "inline", filename=path.name)
                    msg.attach(img)

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

    # Summary embeddings t-SNE chart and drift histogram
    from summary_tsne_chart import build_summary_tsne_chart, build_drift_histogram
    chart_path = build_summary_tsne_chart()
    drift_path, drift_rankings = build_drift_histogram()
    inline_images = []
    chart_block = ""
    if chart_path:
        inline_images.append((str(chart_path), "summary_tsne_chart"))
        chart_block = (
            '<h2>Summary embeddings (t-SNE)</h2>'
            '<p><img src="cid:summary_tsne_chart" alt="Summary embeddings 2D t-SNE" style="max-width:100%;" /></p>'
        )
        print(f"Summary t-SNE chart attached: {chart_path}")
    if drift_path:
        inline_images.append((str(drift_path), "summary_drift_histogram"))
        chart_block += (
            '<h2>Daily narrative drift</h2>'
            '<p><img src="cid:summary_drift_histogram" alt="Drift histogram" style="max-width:100%;" /></p>'
        )
        if drift_rankings:
            chart_block += (
                '<h3>Top narrative shifts (by drift magnitude)</h3>'
                '<table border="1" cellpadding="4" cellspacing="0">'
                '<tr><th>Rank</th><th>From → To</th><th>Drift</th><th>σ</th><th>Explanation</th></tr>'
            )
            for r in drift_rankings:
                chart_block += (
                    f'<tr><td>{r["rank"]}</td>'
                    f'<td>{r["date_before"]} → {r["date_after"]}</td>'
                    f'<td>{r["drift"]:.3f}</td>'
                    f'<td>{r["z_score"]:+.2f}</td>'
                    f'<td>{html.escape(r["explanation"])}</td></tr>'
                )
            chart_block += '</table>'
        print(f"Drift histogram attached: {drift_path}")

    email_body = (
        "<p>Greetings,</p>"
        "<p>Please enjoy the automated doom scrolling content.</p>"
        f"{chart_block}"
        f"<h2>LLM Politics-Headlines Catalysts</h2><pre>{news_catalysts}</pre>"
        f"<h2>LLM Eco-Calendar Catalysts</h2><pre>{calendar_catalysts}</pre>"
        f"<h2>Summary of Today's Headlines</h2><pre>{summary[:10000]}</pre>"
        f"{calendar_html}<hr>"
        "<p>Best regards,<br>MacroDoomscrolling</p>"
    )

    send_email(
        subject="Daily Macro Summary, Catalysts, and Calendar",
        body=email_body,
        inline_images=inline_images if inline_images else None,
    )
"""
Daily macro email: summary, LLM catalysts, CoC graphs, compact catalyst timeline.

Usage:
  python send_off_email.py              # send email
  python send_off_email.py --preview    # write HTML + images, do not send
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
smtp_token = os.getenv("SMTP_EMAIL_TOKEN")
destinataries = os.getenv("DESTINATARIES") or ""
TO_EMAIL = [email.strip() for email in destinataries.split(",") if email.strip()]

SMTP_SERVER = "smtp.protonmail.ch"
SMTP_PORT = 587
USERNAME = "mm@macrodoomscrolling.org"
PASSWORD = smtp_token
FROM_EMAIL = "mm@macrodoomscrolling.org"


def get_calendar_of_the_day_html():
    today = datetime.now()
    end_date = today + timedelta(days=14)
    calendar_path = "calendar/tradingeconomics_calendar_master.csv"
    if not os.path.exists(calendar_path):
        return f"<p>Today's date: {today.strftime('%Y-%m-%d')}<br>No calendar file found.</p>"
    events = []
    with open(calendar_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                event_date = datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if today.date() <= event_date.date() <= end_date.date():
                events.append(
                    {
                        "Date": event_date.strftime("%Y-%m-%d"),
                        "Time": event_date.strftime("%H:%M"),
                        "Country": row.get("Country", ""),
                        "Event": row.get("Event", ""),
                        "Actual": row.get("Actual", ""),
                        "Previous": row.get("Previous", ""),
                        "Consensus": row.get("Consensus", ""),
                        "Forecast": row.get("Forecast", ""),
                    }
                )
    if not events:
        return (
            f"<p>Today's date: {today.strftime('%Y-%m-%d')}<br>"
            "No events found in calendar.</p>"
        )

    parts = [
        "<h2>Calendar: Today + Next 2 Weeks</h2>",
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>Date</th><th>Time</th><th>Country</th><th>Event</th>"
        "<th>Actual</th><th>Prev</th><th>Cons</th><th>Fcst</th></tr>",
    ]
    for e in events:
        parts.append(
            f"<tr><td>{e['Date']}</td><td>{e['Time']}</td><td>{e['Country']}</td>"
            f"<td>{html.escape(e['Event'])}</td><td>{e['Actual']}</td>"
            f"<td>{e['Previous']}</td><td>{e['Consensus']}</td>"
            f"<td>{e['Forecast']}</td></tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


def send_email(subject, body, inline_images=None):
    """
    inline_images: optional list of (file_path, content_id)
    """
    if not TO_EMAIL:
        raise RuntimeError("DESTINATARIES is empty; set it in .env before sending.")
    if not PASSWORD:
        raise RuntimeError("SMTP_EMAIL_TOKEN is empty; set it in .env before sending.")

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
                else:
                    print(f"WARNING: Missing inline image: {path}")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(USERNAME, PASSWORD)
            server.send_message(msg, from_addr=FROM_EMAIL, to_addrs=[recipient])
        print(f"OK: Email sent to {recipient}.")


def _load_catalyst_text(name: str, report_date: str) -> str:
    """Load catalyst file for report date, or the most recent non-empty file."""
    path = Path(f"incoming_catalysts/{name}_{report_date}.txt")
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
        print(f"WARNING: {path.name} exists but is empty; trying older files")

    candidates = sorted(
        Path("incoming_catalysts").glob(f"{name}_*.txt"),
        key=lambda p: p.name,
        reverse=True,
    )
    for fallback in candidates:
        try:
            text = fallback.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            print(f"WARNING: No usable {name} for {report_date}; using {fallback.name}")
            return text
    return ""


def _load_summary(today_str: str) -> str:
    summary_path = Path(f"summaries/summary_{today_str}.txt")
    if not summary_path.exists():
        candidates = sorted(
            Path("summaries").glob("summary_*.txt"),
            key=lambda p: p.name,
            reverse=True,
        )
        summary_path = candidates[0] if candidates else None
    if summary_path and summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8")
        if summary_path.name != f"summary_{today_str}.txt":
            stamp = summary_path.stem.replace("summary_", "")
            summary = (
                f"(Summary from {stamp} — no summary for today yet.)\n\n{summary}"
            )
        return summary
    return "(No summary file found.)"


def build_email_payload(
    today_str: str | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Assemble HTML body + inline image list for the daily email."""
    today_str = today_str or os.environ.get("PIPELINE_DATE") or datetime.now().strftime(
        "%Y-%m-%d"
    )
    print(f"Building email for report date: {today_str}")

    summary = _load_summary(today_str)
    news_catalysts = _load_catalyst_text("news_catalysts", today_str)
    calendar_catalysts = _load_catalyst_text("calendar_catalysts", today_str)

    if not news_catalysts:
        news_catalysts = "(No news catalysts available — identify_catalysts may have skipped headlines.)"
        print("WARNING: News catalysts section will show a placeholder.")
    if not calendar_catalysts:
        calendar_catalysts = "(No calendar catalysts available.)"
        print("WARNING: Calendar catalysts section will show a placeholder.")

    inline_images: list[tuple[str, str]] = []
    chart_block = ""

    # Summary embeddings t-SNE chart and drift histogram
    from summary_tsne_chart import (
        build_summary_tsne_chart,
        build_drift_histogram,
        build_day_over_day_narrative_blurb,
    )

    chart_path = build_summary_tsne_chart()
    drift_path, drift_rankings = build_drift_histogram()
    if chart_path:
        inline_images.append((str(chart_path), "summary_tsne_chart"))
        chart_block += (
            "<h2>Narrative Drift Map</h2>"
            '<p><img src="cid:summary_tsne_chart" alt="Summary embeddings 2D t-SNE" '
            'style="max-width:100%;" /></p>'
        )
        print(f"Summary t-SNE chart attached: {chart_path}")
    if drift_path:
        inline_images.append((str(drift_path), "summary_drift_histogram"))
        chart_block += (
            "<h2>Drifts VS Jumps</h2>"
            '<p><img src="cid:summary_drift_histogram" alt="Drift histogram" '
            'style="max-width:100%;" /></p>'
        )
        day_shift_blurb = build_day_over_day_narrative_blurb(today_str)
        if day_shift_blurb:
            chart_block += (
                "<h3>Day-over-day narrative shift</h3>"
                f"<p>{html.escape(day_shift_blurb)}</p>"
            )
        if drift_rankings:
            chart_block += (
                "<h3>Top narrative shifts (by drift magnitude)</h3>"
                '<table border="1" cellpadding="4" cellspacing="0">'
                "<tr><th>Rank</th><th>From → To</th><th>Drift</th>"
                "<th>σ</th><th>Explanation</th></tr>"
            )
            for r in drift_rankings:
                chart_block += (
                    f'<tr><td>{r["rank"]}</td>'
                    f'<td>{r["date_before"]} → {r["date_after"]}</td>'
                    f'<td>{r["drift"]:.3f}</td>'
                    f'<td>{r["z_score"]:+.2f}</td>'
                    f'<td>{html.escape(r["explanation"])}</td></tr>'
                )
            chart_block += "</table>"
        print(f"Drift histogram attached: {drift_path}")

    # Core-of-cores sub-core graphs + compact catalyst timeline (replaces old HTML table)
    from coc_email_assets import build_coc_email_assets

    coc = build_coc_email_assets()
    inline_images.extend(coc["inline_images"])
    coc_graphs_html = coc.get("graphs_html") or ""
    timeline_html = coc.get("timeline_html") or (
        "<p>No core-of-cores catalyst timeline available "
        "(run core_of_cores_v2 / ensure kg/*_core_*.json exist).</p>"
    )

    email_body = (
        "<p>Greetings,</p>"
        "<p>Please enjoy the automated doom scrolling content.</p>"
        f"{chart_block}"
        f"{coc_graphs_html}"
        f"<h2>LLM Politics-Headlines Catalysts</h2>"
        f"<pre>{html.escape(news_catalysts)}</pre>"
        f"<h2>LLM Eco-Calendar Catalysts</h2>"
        f"<pre>{html.escape(calendar_catalysts)}</pre>"
        f"<h2>Summary of Today's Headlines</h2>"
        f"<pre>{html.escape(summary[:10000])}</pre>"
        f"{timeline_html}<hr>"
        "<p>Best regards,<br>MacroDoomscrolling</p>"
    )
    return email_body, inline_images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send (or preview) the daily macro email.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write summaries/email_preview.html and assets; do not send.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Report date YYYY-MM-DD (default: PIPELINE_DATE or today).",
    )
    args = parser.parse_args(argv)

    body, inline_images = build_email_payload(args.date)

    if args.preview:
        preview_path = Path("summaries/email_preview.html")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite cid: refs to relative file paths for local browser preview
        preview_html = body
        for file_path, cid in inline_images:
            rel = Path(file_path)
            parts = rel.as_posix().replace("\\", "/")
            # preview lives in summaries/; sibling PNGs use basename
            if parts.startswith("summaries/") or rel.parent.name == "summaries":
                href = rel.name
            else:
                href = f"../{parts}"
            preview_html = preview_html.replace(f"cid:{cid}", href)
        preview_path.write_text(preview_html, encoding="utf-8")
        print(f"Preview written to {preview_path} ({len(inline_images)} images)")
        print("Open that HTML file in a browser to review.")
        return 0

    send_email(
        subject="Daily Macro Summary, Catalysts, and Calendar",
        body=body,
        inline_images=inline_images if inline_images else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

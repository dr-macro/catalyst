#!/usr/bin/env python3
"""
Daily pipeline: runs in order
  1. TradingEconomics calendar scraper
  2. Daily financial summary (summarize_headlines)
  3. Catalyst ranking (identify_catalysts)
  4. Daily macro email (send_off_email)

Designed to be the single entry point for a scheduled GitHub Action.
Exits on first failure (non-zero exit from any step).
"""

import subprocess
import sys
from pathlib import Path

# Order matters: calendar → summary → catalyst → email
STEPS = [
    ("TradingEconomics calendar", "calendar_scraper.py"),
    ("Daily financial summary", "summarize_headlines.py"),
    ("Catalyst ranking", "identify_catalysts.py"),
    ("Daily macro email", "send_off_email.py"),
]


def main():
    repo_root = Path(__file__).resolve().parent
    if repo_root != Path.cwd():
        print(f"Running from repo root: {repo_root}")
        # Allow running from repo root when invoked as python run_daily_pipeline.py
        pass

    for label, script in STEPS:
        script_path = repo_root / script
        if not script_path.exists():
            print(f"Error: {script} not found at {script_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n{'='*60}\n>>> {label}: {script}\n{'='*60}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            print(f"Pipeline failed at: {label} ({script})", file=sys.stderr)
            sys.exit(result.returncode)

    print("\n" + "="*60)
    print("Pipeline completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()

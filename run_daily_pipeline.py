#!/usr/bin/env python3
"""
Daily pipeline: runs in order
  1. TradingEconomics calendar scraper (calendar_scraper.py)
  2. Daily financial summary (summarize_headlines.py)
  2b. Summary backfill (backfill_summaries.py), if needed
  3. Catalyst ranking (identify_catalysts.py)
  4. Refresh sub-core catalysts from recent headlines (refresh_coc_subcores.py)
  5. CoC email assets from latest kg sub-cores (coc_email_assets.py)
  6. Daily macro email (send_off_email.py)

Designed to be the single entry point for a scheduled GitHub Action.
Exits on first failure (non-zero exit from any step).
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Order matters: calendar → summary → catalyst → CoC assets → email
STEPS = [
    ("TradingEconomics calendar", "calendar_scraper.py"),
    ("Daily financial summary", "summarize_headlines.py"),
    ("Summary backfill (if needed)", "backfill_summaries.py"),
    ("Catalyst ranking", "identify_catalysts.py"),
    ("CoC catalyst refresh", "refresh_coc_subcores.py"),
    ("Core-of-cores email assets", "coc_email_assets.py"),
    ("Daily macro email", "send_off_email.py"),
]


def main():
    repo_root = Path(__file__).resolve().parent
    if repo_root != Path.cwd():
        print(f"Running from repo root: {repo_root}")

    # Pin report date at pipeline start so later steps (after midnight UTC) still
    # use the same headlines CSV and output filenames.
    env = os.environ.copy()
    env.setdefault("PIPELINE_DATE", datetime.now().strftime("%Y-%m-%d"))
    print(f"Pipeline date: {env['PIPELINE_DATE']}")

    for label, script in STEPS:
        script_path = repo_root / script
        if not script_path.exists():
            print(f"Error: {script} not found at {script_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n{'='*60}\n>>> {label}: {script}\n{'='*60}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            env=env,
        )
        if result.returncode != 0:
            print(f"Pipeline failed at: {label} ({script})", file=sys.stderr)
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

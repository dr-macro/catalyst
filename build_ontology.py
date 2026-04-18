#!/usr/bin/env python3
"""
Pipeline step 3.5 — build today's ontology, store it, diff vs yesterday,
and generate visualisation charts.

Outputs written to:
  ontologies/ontology_<date>.json      — raw ontology
  ontologies/diff_<date>.txt           — plain-text diff summary
  summaries/ontology_diff_<date>.png   — network graph (prev vs curr)
  summaries/ontology_themes_<date>.png — theme prominence bar chart
"""

from datetime import datetime
from pathlib import Path

from ontology_builder import build_ontology
from ontology_diff import compute_diff, format_diff_text
from ontology_store import get_previous_ontology, save_ontology
from ontology_viz import build_ontology_diff_chart, build_theme_drift_chart

ONTOLOGIES_DIR = Path("ontologies")


def main() -> None:
    today_str = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*60}\nBuilding ontology for {today_str}\n{'='*60}")

    # --- build ---
    ontology = build_ontology(today_str)
    if not ontology:
        print("No source data found; skipping ontology step.")
        return

    # --- store ---
    saved_path = save_ontology(ontology)
    print(f"Saved: {saved_path}")

    # --- diff ---
    prev = get_previous_ontology(today_str)
    if not prev:
        print("No previous ontology found — diff skipped (first run).")
        return

    print(f"Comparing with {prev['date']}…")
    diff = compute_diff(prev, ontology)

    diff_text = format_diff_text(diff)
    print(diff_text)

    ONTOLOGIES_DIR.mkdir(exist_ok=True)
    diff_txt_path = ONTOLOGIES_DIR / f"diff_{today_str}.txt"
    diff_txt_path.write_text(diff_text, encoding="utf-8")
    print(f"Diff text saved: {diff_txt_path}")

    # --- visualise ---
    build_ontology_diff_chart(prev, ontology, diff)
    build_theme_drift_chart(prev, ontology, diff)

    print(f"\n{'='*60}\nOntology step complete.\n{'='*60}")


if __name__ == "__main__":
    main()

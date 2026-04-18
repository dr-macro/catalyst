"""
Persist and retrieve daily ontologies as JSON files in ontologies/.
"""

import json
from datetime import datetime
from pathlib import Path

ONTOLOGIES_DIR = Path("ontologies")


def save_ontology(ontology: dict) -> Path:
    ONTOLOGIES_DIR.mkdir(exist_ok=True)
    date_str = ontology.get("date", datetime.today().strftime("%Y-%m-%d"))
    path = ONTOLOGIES_DIR / f"ontology_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)
    return path


def load_ontology(date_str: str) -> dict | None:
    path = ONTOLOGIES_DIR / f"ontology_{date_str}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_previous_ontology(date_str: str) -> dict | None:
    """Return the most recent ontology strictly before date_str."""
    if not ONTOLOGIES_DIR.exists():
        return None
    target = datetime.strptime(date_str, "%Y-%m-%d")
    candidates = []
    for p in ONTOLOGIES_DIR.glob("ontology_*.json"):
        try:
            d = datetime.strptime(p.stem.replace("ontology_", ""), "%Y-%m-%d")
            if d < target:
                candidates.append((d, p))
        except ValueError:
            continue
    if not candidates:
        return None
    _, prev_path = max(candidates, key=lambda x: x[0])
    with open(prev_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_ontology_dates() -> list[str]:
    """Return all stored dates in ascending order."""
    if not ONTOLOGIES_DIR.exists():
        return []
    dates = []
    for p in ONTOLOGIES_DIR.glob("ontology_*.json"):
        try:
            datetime.strptime(p.stem.replace("ontology_", ""), "%Y-%m-%d")
            dates.append(p.stem.replace("ontology_", ""))
        except ValueError:
            continue
    return sorted(dates)

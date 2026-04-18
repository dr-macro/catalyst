"""
Ontology construction engine for the catalyst pipeline.

Inspired by the brain-in-the-fish-808 OWL decomposition approach: decompose
source documents into typed nodes (Entity, Theme) and labelled directed edges,
then merge partial extractions into a unified day-level knowledge graph.

Input:  data/articles_<date>.csv  +  any *.pdf in pdfs/
Output: dict with keys: date, nodes, edges, themes, source_headlines_count,
        source_pdf_count
"""

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = Path("data")
PDFS_DIR = Path("pdfs")
MODEL = "gpt-4o-mini"
MAX_CHUNK_CHARS = 15_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip())[:50].strip("_")


def _extract_json(raw: str) -> dict:
    """Pull JSON from LLM response, tolerating markdown fences."""
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if m:
        raw = m.group(1)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _get_csv_path() -> tuple[str | None, str]:
    today = datetime.today()
    for delta in range(3):
        d = today - timedelta(days=delta)
        s = d.strftime("%Y-%m-%d")
        p = DATA_DIR / f"articles_{s}.csv"
        if p.exists():
            return str(p), s
    return None, today.strftime("%Y-%m-%d")


def load_headlines(csv_path: str | None) -> list[str]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    if "title" not in df.columns:
        return []
    return df["title"].dropna().tolist()


def load_pdfs() -> list[str]:
    """Return list of extracted text blocks, one per PDF, capped at 8000 chars."""
    if not PDFS_DIR.exists():
        return []
    try:
        import pypdf  # noqa: PLC0415
    except ImportError:
        print("  Warning: pypdf not installed — skipping PDFs. pip install pypdf")
        return []

    texts = []
    for pdf_path in sorted(PDFS_DIR.glob("*.pdf")):
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            if text.strip():
                texts.append(f"[PDF: {pdf_path.name}]\n{text[:8000]}")
                print(f"  Loaded PDF: {pdf_path.name} ({len(text):,} chars)")
        except Exception as e:
            print(f"  Warning: could not read {pdf_path.name}: {e}")
    return texts


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a macro financial analyst building a knowledge graph from news content.

Extract from the text below:

1. ENTITIES — key actors: financial institutions, central banks, countries,
   political figures, companies, assets (stocks/bonds/commodities/FX),
   economic indicators, policies.

2. RELATIONSHIPS — directional labelled edges between entities
   (e.g. "raises interest rates", "imposes tariffs on", "reports earnings miss").

3. NARRATIVE THEMES — high-level macro/geopolitical stories tying entities
   together (e.g. "Monetary Policy Tightening", "US-China Trade Tension").

Scoring:
- prominence 0.0–1.0: how frequently mentioned and how central to the narrative.
- weight 0.0–1.0: strength/frequency of the relationship.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "nodes": [
    {{"id": "federal_reserve", "type": "Entity", "subtype": "Institution",
      "label": "Federal Reserve", "prominence": 0.9}}
  ],
  "edges": [
    {{"from": "federal_reserve", "to": "interest_rates",
      "relation": "raises", "label": "raises interest rates", "weight": 0.8}}
  ],
  "themes": [
    {{"id": "monetary_tightening", "label": "Monetary Policy Tightening",
      "description": "Central banks raising rates to combat inflation",
      "prominence": 0.85, "entities": ["federal_reserve", "interest_rates"]}}
  ]
}}

--- NEWS CONTENT ---
{text}
--- END ---

Return ONLY the JSON object:"""

_MERGE_PROMPT = """\
You are a macro financial analyst. Merge the partial knowledge graphs below
into ONE unified graph. Rules:
- Merge nodes referring to the same entity (e.g. "Fed" = "Federal Reserve").
- Use the most descriptive label; take the max prominence.
- Deduplicate edges with the same from/relation/to.
- Merge themes covering the same narrative; take max prominence.
- Keep at most: 35 nodes, 45 edges, 15 themes (most prominent).

Partial graphs (JSON array):
{graphs}

Return ONLY a single merged JSON object:
{{"nodes": [...], "edges": [...], "themes": [...]}}"""


def _extract_chunk(text: str) -> dict:
    prompt = _EXTRACTION_PROMPT.format(text=text[:MAX_CHUNK_CHARS])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return _extract_json(resp.choices[0].message.content)


def _merge(partial: list[dict]) -> dict:
    if len(partial) == 1:
        return partial[0]
    graphs_json = json.dumps(partial, indent=2)[:30_000]
    prompt = _MERGE_PROMPT.format(graphs=graphs_json)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    merged = _extract_json(resp.choices[0].message.content)
    if not merged.get("nodes"):
        merged = _simple_merge(partial)
    return merged


def _simple_merge(graphs: list[dict]) -> dict:
    """Fallback deduplication when LLM merge fails."""
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    themes: dict[str, dict] = {}

    for g in graphs:
        for n in g.get("nodes", []):
            nid = n.get("id") or _slug(n.get("label", "unknown"))
            if nid in nodes:
                nodes[nid]["prominence"] = max(
                    nodes[nid].get("prominence", 0), n.get("prominence", 0)
                )
            else:
                nodes[nid] = {**n, "id": nid}

        for e in g.get("edges", []):
            key = f"{e.get('from')}::{e.get('relation')}::{e.get('to')}"
            edges.setdefault(key, e)

        for t in g.get("themes", []):
            tid = t.get("id") or _slug(t.get("label", "unknown"))
            if tid in themes:
                themes[tid]["prominence"] = max(
                    themes[tid].get("prominence", 0), t.get("prominence", 0)
                )
            else:
                themes[tid] = {**t, "id": tid}

    return {
        "nodes": sorted(nodes.values(), key=lambda n: -n.get("prominence", 0))[:35],
        "edges": list(edges.values())[:45],
        "themes": sorted(themes.values(), key=lambda t: -t.get("prominence", 0))[:15],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ontology(date_str: str | None = None) -> dict | None:
    """
    Build a day-level ontology from headlines + PDFs.
    Returns None if no source data is available.
    """
    csv_path, resolved_date = _get_csv_path()
    if date_str is None:
        date_str = resolved_date

    headlines = load_headlines(csv_path)
    pdf_texts = load_pdfs()

    if not headlines and not pdf_texts:
        print("  No headlines or PDFs found.")
        return None

    print(f"  {len(headlines)} headlines, {len(pdf_texts)} PDFs loaded.")

    # --- chunk headlines ---
    chunks: list[str] = []
    buf, buf_len = [], 0
    for h in headlines:
        if buf_len + len(h) > MAX_CHUNK_CHARS:
            chunks.append("\n".join(buf))
            buf, buf_len = [h], len(h)
        else:
            buf.append(h)
            buf_len += len(h)
    if buf:
        chunks.append("\n".join(buf))

    # PDFs each get their own chunk
    chunks.extend(pdf_texts)

    # --- extract from each chunk ---
    partial: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Extracting ontology chunk {i}/{len(chunks)}…")
        g = _extract_chunk(chunk)
        if g.get("nodes"):
            partial.append(g)

    if not partial:
        print("  No ontology data extracted.")
        return None

    # --- merge ---
    if len(partial) > 1:
        print(f"  Merging {len(partial)} partial graphs…")
        merged = _merge(partial)
    else:
        merged = partial[0]

    # ensure stable IDs
    for node in merged.get("nodes", []):
        if not node.get("id"):
            node["id"] = _slug(node.get("label", "unknown"))

    ontology = {
        "date": date_str,
        "nodes": merged.get("nodes", []),
        "edges": merged.get("edges", []),
        "themes": merged.get("themes", []),
        "source_headlines_count": len(headlines),
        "source_pdf_count": len(pdf_texts),
    }

    print(
        f"  Built ontology: {len(ontology['nodes'])} nodes, "
        f"{len(ontology['edges'])} edges, {len(ontology['themes'])} themes."
    )
    return ontology


if __name__ == "__main__":
    from ontology_store import save_ontology  # noqa: PLC0415

    ont = build_ontology()
    if ont:
        p = save_ontology(ont)
        print(f"Saved: {p}")

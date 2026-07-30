"""
Vester Sensitivity-Model core-ontology engine.

Turns a list of news headlines into a "core ontology": a compact set of systemic
variables, their cross-impact matrix, cybernetic roles (Active/Passive sums), a
signed effect graph, and (optionally) upcoming catalysts scored against the
system.

This is the notebook pipeline from `making_a_news_kg.ipynb` (Steps 1-4, roles,
Step 6, Step 7) extracted into a single callable with progress bars and no
notebook globals. Entity grounding (Step 5) is intentionally excluded.

The prompts live in a `PromptSet` so the same engine can build both:
  * abstract-variable sub-cores (DEFAULT_PROMPTS), and
  * an issue-level "core of cores" (a tailored PromptSet).

Typical use:

    from vester_core import build_core_ontology
    core = build_core_ontology(headlines, dated_headlines=dated, label="iran")
    core.save()
    core.roles          # pandas DataFrame
    core.catalysts_df   # scored upcoming catalysts
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI, APITimeoutError, RateLimitError

try:  # progress bars; degrade gracefully if tqdm is missing
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback shim
    def _tqdm(*args, **kwargs):
        class _Nop:
            def update(self, *a, **k):
                pass

            def set_postfix_str(self, *a, **k):
                pass

            def set_description(self, *a, **k):
                pass

            def close(self):
                pass

        return _Nop()

load_dotenv()

KG_DIR = Path("kg")
DEFAULT_MODEL = "gpt-5-mini"

# Placeholder tokens used inside prompt templates. Custom tokens (rather than
# str.format) let templates contain raw JSON braces without escaping.
_TOKEN_RE = re.compile(r"<<([A-Z_]+)>>")


def _fill(template: str, **values: object) -> str:
    """Replace <<TOKEN>> placeholders in a template with stringified values."""
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            return m.group(0)
        return str(values[key])

    return _TOKEN_RE.sub(repl, template)


# --------------------------------------------------------------------------- #
# Fixed criteria set (Step 3) — Vester's 18 criteria.
# --------------------------------------------------------------------------- #
CRITERIA_18: list[tuple[str, str]] = [
    # 7 Spheres of Life (Vester)
    ("people",          "People — who is involved"),
    ("economy",         "Economy — what they do"),
    ("space",           "Realm of space — where it happens"),
    ("human_ecology",   "Human ecology — how they feel"),
    ("energy_waste",    "Energy & waste — exchange with environment"),
    ("infrastructure",  "Infrastructure — ways of communication"),
    ("laws_culture",    "Laws & culture — rules of the game"),
    # 3 Physical dimensions
    ("matter",          "Involves matter / material flows"),
    ("energy",          "Involves energy flows"),
    ("information",     "Involves information flows"),
    # 8 Dynamics
    ("openness",        "Openness to surrounding systems"),
    ("inner_outer",     "Inner / outer effect (acts both inward and outward)"),
    ("time_flow",       "Time-bound dynamics (delays, momentum)"),
    ("growth",          "Ability to grow / accumulate"),
    ("learning",        "Ability to learn / change rules"),
    ("distinctness",    "Distinctness from other variables"),
    ("sensitivity",     "Sensitivity to disturbance"),
    ("adaptability",    "Adaptability / self-renewal"),
]


# --------------------------------------------------------------------------- #
# Prompt set (overridable).
# --------------------------------------------------------------------------- #
_DEFAULT_SYS_MSG = (
    "You are an analyst trained in Frederic Vester's Sensitivity Model. "
    "You identify the single dominant narrative across a corpus of news headlines, "
    "and you carefully distinguish abstract systemic variables (states, processes) "
    "from concrete real-world entities (firms, states, institutions, assets, people) "
    "and from dated events (current or upcoming catalysts)."
)

_DEFAULT_STEP1 = """You are given news headlines from <<WINDOW_FROM>> to <<WINDOW_TO>>.

Step 1 — System Description.
1. Identify the SINGLE dominant narrative shaping this news window.
2. Describe it as an open system.
3. State the scope, the larger super-system, and 2-4 adjacent subsystems.

Return strict JSON:
{
  "narrative_title": "<short title, < 80 chars>",
  "narrative_summary": "<2-4 sentences>",
  "scope": "<what is inside the system boundary>",
  "super_system": "<the larger system this is embedded in>",
  "adjacent_subsystems": ["<name>", "<name>", "..."],
  "evidence_headlines": ["<3-6 verbatim or near-verbatim headlines>"]
}

HEADLINES:
<<HEADLINES>>
"""

_DEFAULT_STEP2 = """Step 2 — Core Variable Set (the systemic backbone).

System under study (from Step 1):
Title: <<NARRATIVE_TITLE>>
Summary: <<NARRATIVE_SUMMARY>>
Scope: <<SCOPE>>
Super-system: <<SUPER_SYSTEM>>

Extract EXACTLY <<CORE_N>> abstract core variables that constitute the
"gene-pool" of this system.

HARD RULES (critical):
- Each variable must be a STATE or a PROCESS that can vary continuously.
- NEVER include named entities (no firms, no countries, no people, no institutions).
- NEVER include dated events or one-off announcements.
- All variables must be at the SAME level of aggregation.
- Avoid synonyms; each variable must be distinct.

For each variable return:
  id           snake_case, <= 24 chars, globally unique
  name         human-readable display name
  description  1 sentence, what the variable measures
  kind         one of: driver, flow, stock, perception, institution_field, environment
  ontology_type  must be "state" or "process"

Return strict JSON: { "variables": [ {...}, ... ] }

HEADLINES:
<<HEADLINES>>
"""

_DEFAULT_STEP3 = """Step 3 of the Sensitivity Model — Criteria Matrix.

For EACH variable below, give a relevance score 0..3 against EACH of the 18 criteria:
  0 = no relevance, 1 = slight, 2 = clear, 3 = central

Return strict JSON:
{
  "scores": {
    "<variable_id>": {"<criterion_id>": <int 0..3>, ...},
    ...
  }
}

VARIABLES:
<<VAR_BLOCK>>

CRITERIA:
<<CRIT_BLOCK>>
"""

_DEFAULT_STEP4 = """Step 4 of the Sensitivity Model — Impact Matrix.

For the system "<<NARRATIVE_TITLE>>", build the cross-impact matrix.
For every ordered pair (row -> column), score the DIRECT effect of `row` on `column`:
  0 = no direct effect, 1 = weak, 2 = medium, 3 = strong
Self-impact (i,i) MUST be 0. Be conservative: only direct effects, not chains.

Return strict JSON:
{
  "matrix": {
    "<row_id>": {"<col_id>": <int 0..3>, ...},
    ...
  },
  "rationale": {
    "<row_id>__<col_id>": "<one short reason>",
    ...
  }
}

VARIABLES:
<<VAR_BLOCK>>
"""

_DEFAULT_STEP6 = """Step 6 of the Sensitivity Model — sign each direct effect of the Effect System.

For each ordered pair, give a sign:
  "+" = amplifying (an increase in source raises the target)
  "-" = dampening (an increase in source lowers the target)
  "?" = ambiguous / context-dependent

Return strict JSON:
{ "signs": { "<src>__<dst>": "+" | "-" | "?" } }

VARIABLES:
<<VAR_BLOCK>>

EDGES (only those with strength >= <<EDGE_THRESHOLD>>):
<<PAIR_BLOCK>>
"""

_DEFAULT_STEP7 = """Step 7 of the Sensitivity Model — Upcoming catalysts.

As of <<AS_OF>>, you are given the most recent news headlines for the system
"<<NARRATIVE_TITLE>>". Identify UPCOMING catalysts: concrete events or ongoing
processes, grounded in these headlines, likely to move the system in the near
future. Do NOT restate purely past events unless they imply a scheduled or
again-likely future occurrence.

For each catalyst return:
  id           snake_case, <= 32 chars, unique
  title        short human-readable title
  description  1-2 sentences
  type         "event" (a dated, one-off occurrence) or "process" (an ongoing dynamic)
  timeline     object with:
                 kind   "date" (a specific expected date) or "horizon" (a fuzzy window)
                 date   "YYYY-MM-DD" if kind=="date", else null
                 start  "YYYY-MM-DD" or null   (horizon lower bound, optional)
                 end    "YYYY-MM-DD" or null   (horizon upper bound, optional)
                 label  short human phrase, e.g. "2-4 weeks", "by end of month"
  probability  0.0-1.0 chance it materialises within its timeline
  affected     list of { "variable_id": <id from the catalog>, "intensity": 0-3, "sign": "+"|"-"|"?" }
                 intensity: 0=none, 1=weak, 2=medium, 3=strong direct effect on that variable
                 sign: direction the catalyst pushes the variable
                 ONLY use variable_ids from the catalog below; never invent ids.
  rationale    <= 25 words, cite the driving headline(s)

Return strict JSON: { "catalysts": [ {...}, ... ] }

VARIABLE CATALOG (allowed variable_ids):
<<VAR_CATALOG>>

RECENT HEADLINES (most recent <<LOOKBACK_DAYS>> days, as of <<AS_OF>>):
<<RECENT_BLOCK>>
"""


@dataclass
class PromptSet:
    """Editable prompt templates for a Vester run.

    Override `sys_msg`, `step2` and `step1` (and `core_n`) to change what counts
    as a "variable" — e.g. abstract state/process variables (default) vs.
    issue-level fault lines for a top-level core.

    Templates use <<TOKEN>> placeholders; raw JSON braces are fine.
    """

    sys_msg: str = _DEFAULT_SYS_MSG
    step1: str = _DEFAULT_STEP1
    step2: str = _DEFAULT_STEP2
    step3: str = _DEFAULT_STEP3
    step4: str = _DEFAULT_STEP4
    step6: str = _DEFAULT_STEP6
    step7: str = _DEFAULT_STEP7
    core_n: int = 16


DEFAULT_PROMPTS = PromptSet()


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #
@dataclass
class CoreOntology:
    """The output of one Vester run."""

    label: str
    as_of: Optional[date]
    model: str
    system_description: dict
    variables: list[dict]
    var_ids: list[str]
    var_names: dict[str, str]
    crit_matrix: pd.DataFrame
    M: np.ndarray
    impact_df: pd.DataFrame
    AS: np.ndarray
    PS: np.ndarray
    med_AS: float
    med_PS: float
    roles: pd.DataFrame
    graph: nx.DiGraph
    catalysts: list[dict] = field(default_factory=list)
    catalysts_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def N(self) -> int:
        return len(self.var_ids)

    @property
    def narrative_title(self) -> str:
        return self.system_description.get("narrative_title", self.label)

    def to_dict(self) -> dict:
        """JSON-serializable snapshot of the whole core."""
        graph_payload = {
            "nodes": [
                {"id": n, **{k: _jsonable(v) for k, v in d.items()}}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **{k: _jsonable(w) for k, w in d.items()}}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        return {
            "label": self.label,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "model": self.model,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "system_description": self.system_description,
            "variables": self.variables,
            "var_ids": self.var_ids,
            "var_names": self.var_names,
            "impact_matrix": {
                ri: {cj: int(self.M[i, j]) for j, cj in enumerate(self.var_ids)}
                for i, ri in enumerate(self.var_ids)
            },
            "AS": {vid: int(self.AS[i]) for i, vid in enumerate(self.var_ids)},
            "PS": {vid: int(self.PS[i]) for i, vid in enumerate(self.var_ids)},
            "med_AS": self.med_AS,
            "med_PS": self.med_PS,
            "roles": self.roles.reset_index().to_dict(orient="records"),
            "graph": graph_payload,
            "catalysts": self.catalysts,
            "catalysts_scored": (
                self.catalysts_df.to_dict(orient="records")
                if not self.catalysts_df.empty else []
            ),
        }

    def save(self, out_dir: Path | str = KG_DIR, stamp: Optional[str] = None) -> Path:
        """Write the core to `<out_dir>/<label>_core_<stamp>.json`. Returns the path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if stamp is None:
            stamp = (self.as_of or date.today()).isoformat()
        path = out_dir / f"{_slugify(self.label)}_core_{stamp}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not self.catalysts_df.empty:
            csv_path = out_dir / f"{_slugify(self.label)}_catalysts_{stamp}.csv"
            self.catalysts_df.to_csv(csv_path, index=False)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CoreOntology":
        """Rehydrate a core from a `*_core_*.json` snapshot (for plots / email)."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        var_ids = list(data["var_ids"])
        N = len(var_ids)
        M = np.zeros((N, N), dtype=int)
        for i, ri in enumerate(var_ids):
            row = (data.get("impact_matrix") or {}).get(ri, {})
            for j, cj in enumerate(var_ids):
                M[i, j] = int(row.get(cj, 0))
        AS = np.array([int((data.get("AS") or {}).get(v, 0)) for v in var_ids], dtype=float)
        PS = np.array([int((data.get("PS") or {}).get(v, 0)) for v in var_ids], dtype=float)
        roles = pd.DataFrame(data.get("roles") or [])
        if not roles.empty:
            for key in ("id", "variable", "var_id"):
                if key in roles.columns:
                    roles = roles.set_index(key)
                    break
        # plot_compass looks up roles.loc[vid, "role"] for each var_id
        if not roles.empty and "role" not in roles.columns and len(roles.columns):
            pass  # leave as-is; callers will surface KeyError if malformed
        graph = nx.DiGraph()
        for n in (data.get("graph") or {}).get("nodes") or []:
            nid = n.get("id")
            if nid is None:
                continue
            attrs = {k: v for k, v in n.items() if k != "id"}
            graph.add_node(nid, **attrs)
        for e in (data.get("graph") or {}).get("edges") or []:
            u, v = e.get("source"), e.get("target")
            if u is None or v is None:
                continue
            attrs = {k: val for k, val in e.items() if k not in {"source", "target"}}
            graph.add_edge(u, v, **attrs)
        as_of_raw = data.get("as_of")
        as_of = date.fromisoformat(as_of_raw) if as_of_raw else None
        catalysts = list(data.get("catalysts") or [])
        scored = data.get("catalysts_scored") or []
        catalysts_df = pd.DataFrame(scored) if scored else pd.DataFrame()
        crit_matrix = pd.DataFrame(M, index=var_ids, columns=var_ids)
        impact_df = crit_matrix.copy()
        return cls(
            label=str(data.get("label") or path.stem),
            as_of=as_of,
            model=str(data.get("model") or ""),
            system_description=dict(data.get("system_description") or {}),
            variables=list(data.get("variables") or []),
            var_ids=var_ids,
            var_names=dict(data.get("var_names") or {}),
            crit_matrix=crit_matrix,
            M=M,
            impact_df=impact_df,
            AS=AS,
            PS=PS,
            med_AS=float(data.get("med_AS", float(np.median(AS)) if N else 0.0)),
            med_PS=float(data.get("med_PS", float(np.median(PS)) if N else 0.0)),
            roles=roles,
            graph=graph,
            catalysts=catalysts,
            catalysts_df=catalysts_df,
        )


def _jsonable(v: object) -> object:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# --------------------------------------------------------------------------- #
# LLM plumbing.
# --------------------------------------------------------------------------- #
_LLM_RETRY_EXC = (APIConnectionError, APITimeoutError)


def _make_client(api_key: Optional[str] = None) -> OpenAI:
    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "600"))
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        timeout=timeout,
        max_retries=0,  # explicit backoff in _call_with_retry
    )


def _call_with_retry(
    fn: Callable[[], object],
    *,
    attempts: int = 8,
    base_delay: float = 5.0,
    label: str = "OpenAI API",
):
    """Retry transient connection/timeouts/rate limits with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except RateLimitError as exc:
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
        except _LLM_RETRY_EXC as exc:
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
        except Exception:
            raise
        if attempt == attempts:
            break
        print(f"  {label}: {type(last_exc).__name__}, retry {attempt}/{attempts - 1} "
              f"in {delay:.0f}s...")
        time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    *,
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """Embedding helper with the same retry policy as llm_json."""
    def _go():
        resp = client.embeddings.create(model=model, input=list(texts))
        arr = np.array([d.embedding for d in resp.data], dtype=float)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    return _call_with_retry(_go, label=f"embeddings ({model})")


def llm_json(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
) -> dict:
    """Call the model and parse the reply as JSON, tolerating markdown fences."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    def _go():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )

    resp = _call_with_retry(_go, label=f"chat ({model})")
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if m:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+\}", raw)
        if not m:
            raise
        return json.loads(m.group(0))


def _slugify(s: str, max_len: int = 28) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (s or "x").lower()).strip("_")
    return s[:max_len] or "x"


# --------------------------------------------------------------------------- #
# Individual pipeline stages.
# --------------------------------------------------------------------------- #
def _step1_system_description(client, model, prompts, headline_block, window_label):
    wf, wt = (window_label or ("the start", "the end of the window"))
    prompt = _fill(prompts.step1, WINDOW_FROM=wf, WINDOW_TO=wt, HEADLINES=headline_block)
    return llm_json(client, model, prompt, system=prompts.sys_msg, temperature=0.2)


def _normalize_variables(raw_vars, core_n=None):
    """Slugify + dedupe variable ids and apply defaults. Shared by Step 2 and
    seeded (externally supplied) variable lists."""
    if core_n is not None:
        raw_vars = raw_vars[:core_n]
    seen: set[str] = set()
    variables: list[dict] = []
    for v in raw_vars:
        v = dict(v)  # never mutate the caller's dicts
        vid = _slugify(v.get("id") or v.get("name") or "core", max_len=24)
        cand, k = vid, 2
        while cand in seen:
            cand = f"{vid}_{k}"
            k += 1
        seen.add(cand)
        v["id"] = cand
        v["name"] = v.get("name") or cand
        v["ontology_type"] = (v.get("ontology_type") or "state").lower()
        v["layer"] = "core"
        variables.append(v)
    return variables


def _step2_core_variables(client, model, prompts, system_description, headline_block, core_n):
    prompt = _fill(
        prompts.step2,
        CORE_N=core_n,
        NARRATIVE_TITLE=system_description.get("narrative_title", ""),
        NARRATIVE_SUMMARY=system_description.get("narrative_summary", ""),
        SCOPE=system_description.get("scope", ""),
        SUPER_SYSTEM=system_description.get("super_system", ""),
        HEADLINES=headline_block,
    )
    resp = llm_json(client, model, prompt, system=prompts.sys_msg, temperature=0.2)
    return _normalize_variables(resp.get("variables", []), core_n)


def _var_block(variables: Sequence[dict]) -> str:
    return "\n".join(
        f"- {v['id']}: {v['name']} — {v.get('description', '')}" for v in variables
    )


def _step3_criteria(client, model, prompts, variables):
    crit_keys = [k for k, _ in CRITERIA_18]
    crit_block = "\n".join(f"- {k}: {l}" for k, l in CRITERIA_18)
    prompt = _fill(prompts.step3, VAR_BLOCK=_var_block(variables), CRIT_BLOCK=crit_block)
    resp = llm_json(client, model, prompt, system=prompts.sys_msg, temperature=0.1)
    scores = resp.get("scores", {})

    crit_matrix = pd.DataFrame(
        [[int(scores.get(v["id"], {}).get(c, 0)) for c in crit_keys] for v in variables],
        index=[v["id"] for v in variables],
        columns=crit_keys,
    )
    crit_matrix["TOTAL"] = crit_matrix.sum(axis=1)

    def _retain(row: pd.Series) -> bool:
        total = int(row.drop("TOTAL", errors="ignore").sum())
        high = int((row.drop("TOTAL", errors="ignore") >= 2).sum())
        return not (total < 6 and high <= 2)

    retained_mask = crit_matrix.drop(columns=["TOTAL"]).apply(_retain, axis=1)
    kept = [v for v in variables if retained_mask.get(v["id"], True)]
    return crit_matrix, kept


def _step4_impact_matrix(client, model, prompts, variables, system_description):
    var_ids = [v["id"] for v in variables]
    var_names = {v["id"]: v["name"] for v in variables}
    N = len(variables)
    prompt = _fill(
        prompts.step4,
        NARRATIVE_TITLE=system_description.get("narrative_title", ""),
        VAR_BLOCK=_var_block(variables),
    )
    resp = llm_json(client, model, prompt, system=prompts.sys_msg, temperature=0.1)
    matrix = resp.get("matrix", {})

    M = np.zeros((N, N), dtype=int)
    for i, ri in enumerate(var_ids):
        row = matrix.get(ri, {})
        for j, cj in enumerate(var_ids):
            if i == j:
                continue
            try:
                M[i, j] = int(row.get(cj, 0))
            except (TypeError, ValueError):
                M[i, j] = 0
    M = np.clip(M, 0, 3)
    impact_df = pd.DataFrame(M, index=var_ids, columns=var_ids)
    return M, impact_df, var_ids, var_names


def _classify(as_, ps_, med_AS, med_PS):
    if as_ >= med_AS and ps_ < med_PS:
        return "active (lever)"
    if as_ < med_AS and ps_ >= med_PS:
        return "reactive (sensor)"
    if as_ >= med_AS and ps_ >= med_PS:
        return "critical (risk)"
    return "buffering (inert)"


def _roles(M, var_ids, var_names):
    AS = M.sum(axis=1)
    PS = M.sum(axis=0)
    eps = 1e-6
    Q = (AS + eps) / (PS + eps)
    P = AS * PS
    med_AS = float(np.median(AS))
    med_PS = float(np.median(PS))
    roles = pd.DataFrame({
        "variable": var_ids,
        "name": [var_names[v] for v in var_ids],
        "AS": AS,
        "PS": PS,
        "Q=AS/PS": Q.round(2),
        "P=AS*PS": P,
        "role": [_classify(a, p, med_AS, med_PS) for a, p in zip(AS, PS)],
    }).set_index("variable").sort_values("P=AS*PS", ascending=False)
    return AS, PS, med_AS, med_PS, roles


def _step6_effect_graph(client, model, prompts, variables, var_ids, var_names, M, AS, PS, edge_threshold):
    N = len(var_ids)
    pairs = [(var_ids[i], var_ids[j], int(M[i, j]))
             for i in range(N) for j in range(N)
             if i != j and M[i, j] >= edge_threshold]
    pair_block = "\n".join(f"- {a} -> {b} (strength {s})" for a, b, s in pairs)
    edge_signs = {}
    if pairs:
        prompt = _fill(
            prompts.step6,
            VAR_BLOCK=_var_block(variables),
            EDGE_THRESHOLD=edge_threshold,
            PAIR_BLOCK=pair_block,
        )
        resp = llm_json(client, model, prompt, system=prompts.sys_msg, temperature=0.1)
        edge_signs = resp.get("signs", {}) or {}

    idx = {vid: i for i, vid in enumerate(var_ids)}
    G = nx.DiGraph()
    for v in variables:
        i = idx[v["id"]]
        G.add_node(v["id"], name=v["name"], kind=v.get("kind", ""),
                   AS=int(AS[i]), PS=int(PS[i]))
    for a, b, s in pairs:
        G.add_edge(a, b, strength=s, sign=edge_signs.get(f"{a}__{b}", "?"))
    return G


def _recent_block(dated_headlines, headlines, as_of, lookback_days):
    """Build the Step-7 recent-headline block and resolve as_of."""
    if dated_headlines:
        parsed = []
        for d, h in dated_headlines:
            try:
                dd = pd.to_datetime(d).date()
            except Exception:
                continue
            parsed.append((dd, str(h)))
        if not parsed:
            return "", (as_of or date.today())
        if as_of is None:
            as_of = max(d for d, _ in parsed)
        cutoff = as_of - timedelta(days=lookback_days)
        seen: set[str] = set()
        lines: list[str] = []
        for d, h in sorted(parsed, key=lambda x: x[0]):
            if d < cutoff or h in seen:
                continue
            seen.add(h)
            lines.append(f"[{d}] {h}")
        return "\n".join(lines), as_of

    if as_of is None:
        as_of = date.today()
    seen = set()
    lines = []
    for h in headlines:
        if h in seen:
            continue
        seen.add(h)
        lines.append(str(h))
    return "\n".join(lines), as_of


def _step7_catalysts(client, model, prompts, variables, var_ids, system_description,
                     recent_block, as_of, lookback_days):
    var_catalog = "\n".join(
        f"- {v['id']}: {v['name']} — {v.get('description', '')}" for v in variables
    )
    prompt = _fill(
        prompts.step7,
        AS_OF=as_of,
        NARRATIVE_TITLE=system_description.get("narrative_title", ""),
        VAR_CATALOG=var_catalog,
        LOOKBACK_DAYS=lookback_days,
        RECENT_BLOCK=recent_block,
    )
    resp = llm_json(client, model, prompt, system=prompts.sys_msg, temperature=0.2)

    valid_ids = set(var_ids)
    catalysts: list[dict] = []
    for c in resp.get("catalysts", []) or []:
        affected = []
        for a in c.get("affected", []) or []:
            vid = str(a.get("variable_id", "")).strip()
            if vid not in valid_ids:
                continue
            try:
                intensity = int(round(float(a.get("intensity", 1))))
            except (TypeError, ValueError):
                intensity = 1
            intensity = max(0, min(3, intensity))
            sign = str(a.get("sign", "?")).strip()
            if sign not in {"+", "-", "?"}:
                sign = "?"
            affected.append({"variable_id": vid, "intensity": intensity, "sign": sign})
        if not affected:
            continue
        try:
            prob = max(0.0, min(1.0, float(c.get("probability", 0.0))))
        except (TypeError, ValueError):
            prob = 0.0
        tl = c.get("timeline", {}) or {}
        catalysts.append({
            "id": str(c.get("id", "")).strip() or f"catalyst_{len(catalysts) + 1}",
            "title": str(c.get("title", "")).strip(),
            "description": str(c.get("description", "")).strip(),
            "type": (str(c.get("type", "")).strip().lower() or "event"),
            "timeline": {
                "kind": (str(tl.get("kind", "horizon")).strip().lower() or "horizon"),
                "date": tl.get("date"),
                "start": tl.get("start"),
                "end": tl.get("end"),
                "label": str(tl.get("label", "")).strip(),
            },
            "probability": prob,
            "affected": affected,
            "rationale": str(c.get("rationale", "")).strip(),
        })
    return catalysts


def _score_catalysts(catalysts, var_ids, AS, PS):
    """Intensity-weighted first-order impact scoring (mirrors notebook cell 35)."""
    lev = {vid: float(AS[i]) for i, vid in enumerate(var_ids)}
    idx = {vid: i for i, vid in enumerate(var_ids)}
    rows = []
    for c in catalysts:
        aff = {a["variable_id"]: a["intensity"] for a in c["affected"]}
        int_sum = sum(aff.values())
        impact_score = float(sum(inten * lev[v] for v, inten in aff.items()))
        expected_impact = float(c["probability"]) * impact_score
        if int_sum > 0:
            event_PS = sum(inten * float(PS[idx[v]]) for v, inten in aff.items()) / int_sum
            event_AS = sum(inten * float(AS[idx[v]]) for v, inten in aff.items()) / int_sum
        else:
            event_PS = event_AS = float("nan")
        tl = c["timeline"]
        rows.append({
            "id": c["id"],
            "title": c["title"],
            "type": c["type"],
            "timeline_kind": tl.get("kind", ""),
            "timeline_label": tl.get("label", ""),
            "date": tl.get("date"),
            "start": tl.get("start"),
            "end": tl.get("end"),
            "probability": round(float(c["probability"]), 3),
            "impact_score": round(impact_score, 3),
            "expected_impact": round(expected_impact, 3),
            "event_PS": round(event_PS, 3),
            "event_AS": round(event_AS, 3),
            "n_vars": len(aff),
            "affected_ids": ", ".join(aff.keys()),
            "rationale": c["rationale"],
        })
    catalysts_df = pd.DataFrame(rows, columns=[
        "id", "title", "type", "timeline_kind", "timeline_label", "date", "start", "end",
        "probability", "impact_score", "expected_impact", "event_PS", "event_AS",
        "n_vars", "affected_ids", "rationale",
    ])
    if not catalysts_df.empty:
        catalysts_df = catalysts_df.sort_values("impact_score", ascending=False).reset_index(drop=True)
    return catalysts_df


# --------------------------------------------------------------------------- #
# Main entry point.
# --------------------------------------------------------------------------- #
def build_core_ontology(
    headlines: Sequence[str],
    *,
    dated_headlines: Optional[Sequence[tuple]] = None,
    window_label: Optional[tuple[str, str]] = None,
    core_n: Optional[int] = None,
    edge_threshold: int = 2,
    with_catalysts: bool = True,
    as_of: Optional[date] = None,
    lookback_days: int = 7,
    model: str = DEFAULT_MODEL,
    prompts: PromptSet = DEFAULT_PROMPTS,
    label: str = "core",
    client: Optional[OpenAI] = None,
    show_progress: bool = True,
    seed_variables: Optional[Sequence[dict]] = None,
    run_retention: bool = True,
    max_headlines: Optional[int] = None,
) -> CoreOntology:
    """Build a Vester core ontology from a list of headlines.

    Parameters
    ----------
    headlines : list[str]
        Cleaned headlines describing the system (used for Steps 1-4).
    dated_headlines : list[(date_like, str)], optional
        Headlines with dates, used for the Step-7 catalyst recency window. If
        omitted, `headlines` are used verbatim as the recent block.
    window_label : (str, str), optional
        Human-readable (from, to) date labels injected into the Step-1 prompt.
    core_n : int, optional
        Number of core variables to extract (defaults to `prompts.core_n`).
    edge_threshold : int
        Minimum impact strength for an edge to enter the signed effect graph.
    with_catalysts : bool
        Whether to run Step 7 (catalyst extraction + scoring).
    as_of : date, optional
        "Now" for catalyst forecasting (defaults to newest dated headline / today).
    lookback_days : int
        Recency window (days) for the catalyst headline block.
    model, prompts, label, client, show_progress
        Model id, prompt set, output label, optional OpenAI client, progress bar.
    seed_variables : list[dict], optional
        Pre-made variable list (each `{id?, name, description, kind?,
        ontology_type?}`). When given, Step 2 extraction is skipped and this list
        is used verbatim (ids slugified/deduped). Use to take over a curated list
        produced elsewhere.
    run_retention : bool
        When True (default) Step 3 scores criteria and may drop weak variables.
        Set False to keep the variable list intact (e.g. a curated `seed_variables`
        list) and skip the Step 3 LLM call.
    """
    headlines = [str(h) for h in headlines if str(h).strip()]
    if max_headlines is not None and len(headlines) > max_headlines:
        print(f"[{label}] Truncating {len(headlines)} headlines → {max_headlines} (most recent)")
        headlines = headlines[-max_headlines:]
    if not headlines:
        raise ValueError("build_core_ontology: `headlines` is empty.")
    if client is None:
        client = _make_client()
    if core_n is None:
        core_n = prompts.core_n

    headline_block = "\n\n".join(headlines)

    stages = ["system description", "core variables"]
    if run_retention:
        stages.append("criteria matrix")
    stages += ["impact matrix", "effect graph (signs)"]
    if with_catalysts:
        stages.append("catalysts")
    n_stages = len(stages)
    bar = _tqdm(total=n_stages, disable=not show_progress, desc=f"[{label}] starting")
    _stage_num = 0

    def _begin_stage(name: str) -> None:
        nonlocal _stage_num
        _stage_num += 1
        bar.set_description(f"[{label}] step {_stage_num}/{n_stages}: {name}")

    # Step 1
    _begin_stage("system description")
    system_description = _step1_system_description(
        client, model, prompts, headline_block, window_label)
    bar.update(1)

    # Step 2 (or take over a supplied list)
    _begin_stage("core variables")
    if seed_variables is not None:
        variables = _normalize_variables(seed_variables, core_n)
    else:
        variables = _step2_core_variables(
            client, model, prompts, system_description, headline_block, core_n)
    if not variables:
        bar.close()
        raise RuntimeError("Step 2 produced no variables.")
    bar.update(1)

    # Step 3 (criteria + retention) — optional
    if run_retention:
        _begin_stage("criteria matrix")
        crit_matrix, variables = _step3_criteria(client, model, prompts, variables)
        if not variables:
            bar.close()
            raise RuntimeError("Step 3 retention dropped all variables.")
        bar.update(1)
    else:
        crit_matrix = pd.DataFrame(index=[v["id"] for v in variables])

    # Step 4 (impact matrix)
    _begin_stage("impact matrix")
    M, impact_df, var_ids, var_names = _step4_impact_matrix(
        client, model, prompts, variables, system_description)
    bar.update(1)

    # Roles (local, no LLM)
    AS, PS, med_AS, med_PS, roles = _roles(M, var_ids, var_names)

    # Step 6 (signed effect graph)
    _begin_stage("effect graph (signs)")
    graph = _step6_effect_graph(
        client, model, prompts, variables, var_ids, var_names, M, AS, PS, edge_threshold)
    bar.update(1)

    catalysts: list[dict] = []
    catalysts_df = pd.DataFrame()
    resolved_as_of = as_of
    if with_catalysts:
        _begin_stage("catalysts")
        recent_block, resolved_as_of = _recent_block(
            dated_headlines, headlines, as_of, lookback_days)
        catalysts = _step7_catalysts(
            client, model, prompts, variables, var_ids, system_description,
            recent_block, resolved_as_of, lookback_days)
        catalysts_df = _score_catalysts(catalysts, var_ids, AS, PS)
        bar.update(1)

    bar.set_description(f"[{label}] done ({n_stages}/{n_stages} stages)")
    bar.close()

    return CoreOntology(
        label=label,
        as_of=resolved_as_of,
        model=model,
        system_description=system_description,
        variables=variables,
        var_ids=var_ids,
        var_names=var_names,
        crit_matrix=crit_matrix,
        M=M,
        impact_df=impact_df,
        AS=AS,
        PS=PS,
        med_AS=med_AS,
        med_PS=med_PS,
        roles=roles,
        graph=graph,
        catalysts=catalysts,
        catalysts_df=catalysts_df,
    )


def refresh_catalysts(
    core: CoreOntology,
    *,
    dated_headlines: Optional[Sequence[tuple]] = None,
    headlines: Optional[Sequence[str]] = None,
    as_of: Optional[date] = None,
    lookback_days: int = 7,
    model: Optional[str] = None,
    prompts: PromptSet = DEFAULT_PROMPTS,
    client: Optional[OpenAI] = None,
) -> CoreOntology:
    """Re-run Step 7 on a saved core; variables, roles, and effect graph unchanged."""
    if client is None:
        client = _make_client()
    model = model or core.model or DEFAULT_MODEL
    hl = [str(h) for h in (headlines or []) if str(h).strip()]
    recent_block, resolved_as_of = _recent_block(dated_headlines, hl, as_of, lookback_days)
    if not recent_block.strip():
        print(f"[{core.label}] No recent headlines for catalyst refresh; keeping existing.")
        return core
    catalysts = _step7_catalysts(
        client,
        model,
        prompts,
        core.variables,
        core.var_ids,
        core.system_description,
        recent_block,
        resolved_as_of,
        lookback_days,
    )
    catalysts_df = _score_catalysts(catalysts, core.var_ids, core.AS, core.PS)
    return replace(
        core,
        as_of=resolved_as_of,
        model=model,
        catalysts=catalysts,
        catalysts_df=catalysts_df,
    )


# --------------------------------------------------------------------------- #
# Optional plotting helpers (import-safe: matplotlib imported lazily).
# --------------------------------------------------------------------------- #
_ROLE_COLOR = {
    "active (lever)":     "#2ca02c",
    "critical (risk)":    "#d62728",
    "reactive (sensor)":  "#17becf",
    "buffering (inert)":  "#cccccc",
}
_SIGN_COLOR = {"+": "#1a9850", "-": "#d73027", "?": "#888888"}

# Shared edge-filter defaults for effect-graph plots (geo core + sub-cores).
SPARSE_EFFECT_GRAPH_KW = dict(
    max_edges_per_node=1,
    weak_edges_per_node=1,
    show_weak_edges=True,
    max_catalysts=None,              # None = all above threshold
    max_catalyst_links=2,            # top affected vars per catalyst
    catalyst_impact_threshold=None,  # None = median impact_score
)


def plot_impact_matrix(core: CoreOntology, ax=None):
    """Heatmap of the cross-impact matrix (mirrors notebook cell 30)."""
    import matplotlib.pyplot as plt

    M, var_ids, var_names = core.M, core.var_ids, core.var_names
    N = core.N
    labels = [var_names.get(v, v) for v in var_ids]
    if ax is None:
        fig_w, fig_h = max(10, 0.55 * N + 2), max(8, 0.55 * N + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.figure
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=3)
    ax.set_xticks(range(N), labels=labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(N), labels=labels, fontsize=8)
    ax.set_xlabel("affected (column)")
    ax.set_ylabel("influencer (row)")
    ax.set_title(f"Impact Matrix — {core.narrative_title}")
    for i in range(N):
        for j in range(N):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] >= 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="0..3")
    fig.tight_layout()
    return fig


def _spread_label_offsets(
    anchor_xy: Sequence[tuple[float, float]],
    origin: tuple[float, float],
    *,
    base_radius: float = 44.0,
    min_sep: float = 32.0,
    iterations: int = 40,
) -> list[tuple[float, float]]:
    """Fan catalyst labels apart in offset-point space to reduce overlap."""
    ox0, oy0 = origin

    def _outward(px: float, py: float, base: float = base_radius) -> tuple[float, float]:
        dx, dy = px - ox0, py - oy0
        norm = (dx * dx + dy * dy) ** 0.5
        if norm < 1e-6:
            return 18.0, -18.0
        return dx * base / norm, dy * base / norm

    offsets = [list(_outward(px, py)) for px, py in anchor_xy]
    n = len(offsets)
    if n <= 1:
        return [(float(o[0]), float(o[1])) for o in offsets]

    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = offsets[i][0] - offsets[j][0]
                dy = offsets[i][1] - offsets[j][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < min_sep:
                    if dist < 1e-3:
                        ang = 2 * np.pi * (i - j) / n
                        dx, dy, dist = float(np.cos(ang)), float(np.sin(ang)), 1.0
                    push = (min_sep - dist) * 0.55
                    ux, uy = dx / dist, dy / dist
                    offsets[i][0] += ux * push
                    offsets[i][1] += uy * push
                    offsets[j][0] -= ux * push
                    offsets[j][1] -= uy * push
                    moved = True
        if not moved:
            break
    return [(float(o[0]), float(o[1])) for o in offsets]


def _short_catalyst_label(title: str, timeline: str = "", *, max_len: int = 42) -> str:
    t = (title or "").strip()
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    tl = (timeline or "").strip()
    return t if not tl else f"{t}\n({tl})"


def plot_compass(core: CoreOntology, *, show_catalysts: bool = True,
                 impact_threshold: Optional[float] = None, ax=None):
    """Systemic-role compass with optional catalyst overlay (cells 33 + 36)."""
    import matplotlib.pyplot as plt

    PS, AS, var_ids = core.PS, core.AS, core.var_ids
    med_PS, med_AS = core.med_PS, core.med_AS

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = ax.figure

    def _inward(px, py, base=12.0):
        dx, dy = med_PS - px, med_AS - py
        norm = (dx ** 2 + dy ** 2) ** 0.5
        if norm < 1e-6:
            return 4.0, 4.0
        return dx * base / norm, dy * base / norm

    ax.scatter(PS, AS, s=120,
               c=[_ROLE_COLOR[core.roles.loc[vid, "role"]] for vid in var_ids],
               edgecolor="black", linewidths=0.8, zorder=3, label="variables")
    for i, vid in enumerate(var_ids):
        ox, oy = _inward(float(PS[i]), float(AS[i]))
        ax.annotate(vid, (PS[i], AS[i]), xytext=(ox, oy), textcoords="offset points",
                    fontsize=8, color="#33557a", ha="center", va="center", zorder=4)
    ax.axvline(med_PS, color="grey", linestyle="--", linewidth=1)
    ax.axhline(med_AS, color="grey", linestyle="--", linewidth=1)

    xlim_max = max(PS.max(), 1) * 1.15 + 1
    ylim_max = max(AS.max(), 1) * 1.15 + 1
    ax.set_xlim(-0.5, xlim_max)
    ax.set_ylim(-0.5, ylim_max)

    ax.text(med_PS / 2, ylim_max * 0.95, "ACTIVE\n(lever)",
            ha="center", va="top", fontsize=12, color="#2ca02c", weight="bold")
    ax.text((med_PS + xlim_max) / 2, ylim_max * 0.95, "CRITICAL\n(risk)",
            ha="center", va="top", fontsize=12, color="#d62728", weight="bold")
    ax.text(med_PS / 2, ylim_max * 0.05, "BUFFERING\n(inert)",
            ha="center", va="bottom", fontsize=12, color="#7f7f7f", weight="bold")
    ax.text((med_PS + xlim_max) / 2, ylim_max * 0.05, "REACTIVE\n(sensor)",
            ha="center", va="bottom", fontsize=12, color="#17becf", weight="bold")

    cdf = core.catalysts_df
    if show_catalysts and cdf is not None and not cdf.empty:
        scores = cdf["impact_score"]
        thr = float(np.median(scores)) if impact_threshold is None else impact_threshold
        shown = cdf[(cdf["impact_score"] >= thr)
                    & cdf["event_PS"].notna() & cdf["event_AS"].notna()]
        if not shown.empty:
            s_lo, s_hi = float(shown["impact_score"].min()), float(shown["impact_score"].max())

            def _msize(s):
                if s_hi <= s_lo:
                    return 340.0
                return 160.0 + 700.0 * (s - s_lo) / (s_hi - s_lo)

            ax.scatter(shown["event_PS"], shown["event_AS"],
                       s=[_msize(s) for s in shown["impact_score"]],
                       marker="*", c="#d62728", edgecolor="black", linewidths=0.8,
                       zorder=5, label="catalysts")
            anchors = [(float(r.event_PS), float(r.event_AS)) for r in shown.itertuples()]
            min_sep = 36.0 if len(anchors) <= 4 else 30.0
            cat_offsets = _spread_label_offsets(
                anchors, (med_PS, med_AS), base_radius=46.0, min_sep=min_sep,
            )
            for j, (r, (ox, oy)) in enumerate(zip(shown.itertuples(), cat_offsets)):
                lbl = _short_catalyst_label(r.title, r.timeline_label)
                ax.annotate(
                    lbl, (r.event_PS, r.event_AS), xytext=(ox, oy),
                    textcoords="offset points", fontsize=7, color="#7a1416",
                    weight="bold", ha="left" if ox >= 0 else "right",
                    va="bottom" if oy >= 0 else "top", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728",
                              alpha=0.9, lw=0.5),
                    arrowprops=dict(arrowstyle="-", color="#7a1416", lw=0.5,
                                    shrinkA=6, shrinkB=2),
                )

    ax.set_xlabel("Passive Sum  PS  ->  influenced by others")
    ax.set_ylabel("Active Sum  AS  ->  influences others")
    ax.set_title(f"Systemic Role compass — {core.narrative_title}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def _select_effect_edges(
    G: nx.DiGraph,
    *,
    min_strength: int = 2,
    max_edges_per_node: Optional[int] = 3,
    weak_edges_per_node: int = 2,
    show_weak_edges: bool = True,
    max_edges: Optional[int] = None,
) -> set[tuple[str, str]]:
    """Pick a sparse edge set for plotting."""
    cand = [
        (u, v, int(d.get("strength", 0)))
        for u, v, d in G.edges(data=True)
        if int(d.get("strength", 0)) >= min_strength
    ]
    if max_edges is not None:
        cand.sort(key=lambda x: x[2], reverse=True)
        return {(u, v) for u, v, _ in cand[:max_edges]}
    if max_edges_per_node is not None:
        kept: set[tuple[str, str]] = set()
        for u in G.nodes():
            out = sorted(
                [(u, v, int(G[u][v].get("strength", 0))) for v in G.successors(u)
                 if int(G[u][v].get("strength", 0)) >= min_strength],
                key=lambda x: x[2], reverse=True,
            )
            strong = [e for e in out if e[2] >= 3][:max_edges_per_node]
            kept.update((a, b) for a, b, _ in strong)
            if show_weak_edges and weak_edges_per_node:
                weak = [e for e in out if e[2] == 2][:weak_edges_per_node]
                kept.update((a, b) for a, b, _ in weak)
        return kept
    return {(u, v) for u, v, _ in cand}


def plot_effect_graph(core: CoreOntology, *, max_edges_per_node: Optional[int] = 3,
                      weak_edges_per_node: int = 2, max_edges: Optional[int] = None,
                      min_strength: int = 2, show_weak_edges: bool = True,
                      show_catalysts: bool = True,
                      max_catalysts: Optional[int] = None,
                      max_catalyst_links: int = 3,
                      catalyst_impact_threshold: Optional[float] = None,
                      ax=None):
    """Signed effect-system graph with catalyst star nodes (mirrors cell 37)."""
    import matplotlib.pyplot as plt

    G = core.graph.copy()
    roles = core.roles
    catalyst_color = "#ffcc00"

    kept = _select_effect_edges(
        G, min_strength=min_strength, show_weak_edges=show_weak_edges,
        max_edges_per_node=max_edges_per_node, weak_edges_per_node=weak_edges_per_node,
        max_edges=max_edges,
    )

    connected = {n for edge in kept for n in edge}
    if connected:
        G = G.subgraph(connected).copy()
        kept = {(u, v) for u, v in kept if u in G and v in G}
    else:
        G = nx.DiGraph()
        kept = set()

    var_nodes = list(G.nodes())
    N = len(var_nodes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 10))
    else:
        fig = ax.figure

    if N == 0 and not (show_catalysts and core.catalysts):
        ax.text(0.5, 0.5, "No connected nodes in effect graph", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f"Effect System — {core.narrative_title}")
        ax.axis("off")
        fig.tight_layout()
        return fig

    node_colors = [_ROLE_COLOR[roles.loc[n, "role"]] for n in var_nodes] if var_nodes else []
    node_sizes = [min(380, 80 + 14 * (G.nodes[n]["AS"] + G.nodes[n]["PS"])) for n in var_nodes]

    Gc = G.copy()
    cat_nodes = []
    if show_catalysts and core.catalysts:
        cdf = core.catalysts_df
        score_by_id = (
            dict(zip(cdf["id"], cdf["impact_score"]))
            if cdf is not None and not cdf.empty else {}
        )
        show_cat_ids: set[str] = set()
        if cdf is not None and not cdf.empty:
            thr = catalyst_impact_threshold
            if thr is None:
                thr = float(np.median(cdf["impact_score"]))
            eligible = cdf[cdf["impact_score"] >= thr].sort_values(
                "impact_score", ascending=False)
            if max_catalysts is not None:
                eligible = eligible.head(max_catalysts)
            show_cat_ids = set(eligible["id"])
        else:
            show_cat_ids = {c["id"] for c in core.catalysts}

        for c in core.catalysts:
            if c["id"] not in show_cat_ids:
                continue
            nid = f"cat::{c['id']}"
            Gc.add_node(nid, name=c["title"], node_kind="catalyst",
                        impact=float(score_by_id.get(c["id"], 0.0)))
            cat_nodes.append(nid)
            affected = sorted(
                (a for a in c.get("affected", []) if a["variable_id"] in Gc),
                key=lambda a: int(a.get("intensity", 0)),
                reverse=True,
            )[:max_catalyst_links]
            for a in affected:
                Gc.add_edge(nid, a["variable_id"], strength=int(a["intensity"]),
                            sign=a["sign"], kind="catalyst")

    if N > 0:
        pos = nx.spring_layout(G, seed=7, k=1.6 / max(np.sqrt(N), 1))
        if cat_nodes:
            pos = nx.spring_layout(Gc, pos=pos, fixed=var_nodes, seed=7,
                                   k=1.9 / max(np.sqrt(Gc.number_of_nodes()), 1))
    elif cat_nodes:
        pos = nx.spring_layout(Gc, seed=7, k=1.9 / max(np.sqrt(Gc.number_of_nodes()), 1))
    else:
        pos = {}

    if var_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_color=node_colors,
                               node_size=node_sizes, edgecolors="black", linewidths=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    edge_kw = dict(arrowsize=20, min_source_margin=10, min_target_margin=18,
                   connectionstyle="arc3,rad=0.08", ax=ax)

    strength_bands = [(3, 3, 0.82, 1.0)]   # strong: full width
    if show_weak_edges:
        strength_bands.append((2, 2, 0.30, 0.55))  # weak: faded + thinner

    for min_s, max_s, alpha, width_scale in strength_bands:
        for sign, color, style in [("+", "#1a9850", "solid"),
                                   ("-", "#d73027", "solid"),
                                   ("?", "#888888", "dashed")]:
            edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d["sign"] == sign and (u, v) in kept and min_s <= d["strength"] <= max_s]
            if not edges:
                continue
            widths = [width_scale * (0.6 + 0.9 * G[u][v]["strength"]) for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, edge_color=color,
                                   style=style, alpha=alpha, **edge_kw)

    if cat_nodes:
        cat_edge_kw = dict(arrowsize=22, min_source_margin=8, min_target_margin=20,
                           connectionstyle="arc3,rad=0.08", ax=ax)
        for sgn, color in _SIGN_COLOR.items():
            c_edges = [(u, v) for u, v, d in Gc.edges(data=True)
                       if d.get("kind") == "catalyst" and d["sign"] == sgn]
            c_widths = [0.6 + 0.9 * Gc[u][v]["strength"] for u, v in c_edges]
            nx.draw_networkx_edges(Gc, pos, edgelist=c_edges, width=c_widths, edge_color=color,
                                   style="dotted", alpha=0.7, **cat_edge_kw)
        imp = np.array([Gc.nodes[n]["impact"] for n in cat_nodes], dtype=float)
        if imp.size and imp.max() > imp.min():
            c_sizes = 220 + 500 * (imp - imp.min()) / (imp.max() - imp.min())
        else:
            c_sizes = np.full(len(cat_nodes), 350.0)
        nx.draw_networkx_nodes(Gc, pos, nodelist=cat_nodes, node_color=catalyst_color,
                               node_shape="*", node_size=list(c_sizes),
                               edgecolors="black", linewidths=1.0, ax=ax)
        nx.draw_networkx_labels(Gc, pos, labels={n: Gc.nodes[n]["name"] for n in cat_nodes},
                                font_size=7, font_color="#6b5200", ax=ax)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor="black", markersize=12, label=r)
               for r, c in _ROLE_COLOR.items()]
    handles += [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=catalyst_color,
                   markeredgecolor="black", markersize=16, label="upcoming catalyst"),
        plt.Line2D([0], [0], color="#1a9850", lw=2, label="+ amplifying"),
        plt.Line2D([0], [0], color="#d73027", lw=2, label="- dampening"),
        plt.Line2D([0], [0], color="#888888", lw=2, linestyle="--", label="? ambiguous"),
        plt.Line2D([0], [0], color="#888888", lw=2, linestyle=":", label="catalyst link"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_title(f"Effect System — {core.narrative_title}")
    ax.axis("off")
    fig.tight_layout()
    return fig

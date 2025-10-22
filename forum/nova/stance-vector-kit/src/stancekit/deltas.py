
import os, json
from pathlib import Path
from typing import List, Dict

KEYS_DEFAULT = ["cross_context_cosine", "flicker_index"]

def _read_json(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None

def compute_deltas(session_dirs: List[str], keys: List[str] = None) -> Dict:
    keys = keys or KEYS_DEFAULT
    data = []
    for d in session_dirs:
        m = _read_json(Path(d) / "metrics.json") or {}
        row = {k: _to_float(m.get(k)) for k in keys}
        data.append(row)
    # choose first as baseline (A), compute deltas X - A
    if not data:
        return {"keys": keys, "baseline_index": 0, "rows": []}
    baseline = data[0]
    deltas = []
    for i, row in enumerate(data):
        delta = {}
        for k in keys:
            a = baseline.get(k)
            b = row.get(k)
            delta[k] = (None if a is None or b is None else (b - a))
        deltas.append(delta)
    return {"keys": keys, "baseline_index": 0, "rows": data, "deltas": deltas}

def render_delta_html(session_names: List[str], results: Dict) -> str:
    keys = results["keys"]
    rows = results["rows"]
    dels = results["deltas"]
    def _fmt(x):
        return "" if x is None else f"{x:.4f}"
    # Build simple table
    head = "<tr><th>Session</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
    body_rows = []
    for i, (r, d) in enumerate(zip(rows, dels)):
        name = session_names[i] if session_names and i < len(session_names) else f"S{i+1}"
        vals = "".join(f"<td>{_fmt(r.get(k))}</td>" for k in keys)
        body_rows.append(f"<tr><td><b>{name}</b></td>{vals}</tr>")
    delta_rows = []
    for i, d in enumerate(dels):
        name = session_names[i] if session_names and i < len(session_names) else f"S{i+1}"
        vals = "".join(f"<td>{_fmt(d.get(k))}</td>" for k in keys)
        delta_rows.append(f"<tr><td><b>{name}</b></td>{vals}</tr>")
    html = [
        "<div class='card'><h2>Metric Table</h2><table class='tbl'>",
        head, *body_rows, "</table></div>",
        "<div class='card'><h2>Delta vs Baseline (Session 1)</h2><table class='tbl'>",
        head, *delta_rows, "</table></div>"
    ]
    return "\n".join(html)

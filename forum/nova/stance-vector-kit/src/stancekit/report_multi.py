
import os, json, base64, datetime
from pathlib import Path
from typing import List, Optional

SESS_ARTIFACTS = [
    ("Metrics", "metrics.json"),
    ("Drift", "drift_events.json"),
    ("Clusters", "cluster_summary.json"),
    ("A/B Metrics", "ab_metrics.json"),
    ("Stance Windows", "stance_windows.csv"),
    ("Viz: PCA", "trajectory_pca.png"),
    ("Viz: UMAP", "trajectory_umap.png"),
    ("Viz: A/B Overlay", "ab_overlay.png"),
]

def _embed_img(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width: 100%; height: auto;"/>'

def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _table_from_dict(label: str, rows: List[dict]) -> str:
    if not rows:
        return "<div class='small'>No data.</div>"
    # collect keys
    keys = sorted({k for r in rows for k in r.keys()})
    thead = "<tr><th>Session</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
    body = []
    for r in rows:
        sid = r.pop("_session", "session")
        body.append("<tr><td><b>%s</b></td>%s</tr>" % (sid, "".join(f"<td>{r.get(k, '')}</td>" for k in keys)))
    return f"<div class='card'><h2>{label}</h2><table class='tbl'>{thead}{''.join(body)}</table></div>"

def _metrics_row(session_name: str, metrics: dict) -> dict:
    row = {"_session": session_name}
    if not isinstance(metrics, dict):
        return row
    for k,v in metrics.items():
        row[k] = v
    return row

def generate_multi_report(input_dirs: List[str], session_names: Optional[List[str]], output_html: str, title: str = "SVK Multi-Session Report"):
    tstamp = datetime.datetime.utcnow().isoformat() + "Z"
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title>",
             "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;}h1{margin-bottom:4px;} .card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0;} pre{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto;} .small{color:#666;font-size:12px} table.tbl{border-collapse:collapse;width:100%;} table.tbl th, table.tbl td{border:1px solid #eee;padding:6px 8px;text-align:left;} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;}</style>",
             "</head><body>",
             f"<h1>{title}</h1><div class='small'>Generated {tstamp}</div>"]

    # metrics table across sessions
    metric_rows = []
    for i, d in enumerate(input_dirs):
        name = session_names[i] if session_names and i < len(session_names) else f"S{i+1}"
        p = Path(d)
        m = _read_json(p / "metrics.json")
        metric_rows.append(_metrics_row(name, m if m else {}))
    parts.append(_table_from_dict("Session Metrics", metric_rows))

    # drift summary table
    drift_rows = []
    for i, d in enumerate(input_dirs):
        name = session_names[i] if session_names and i < len(session_names) else f"S{i+1}"
        p = Path(d)
        dj = _read_json(p / "drift_events.json")
        row = {"_session": name}
        if dj and isinstance(dj, dict):
            row["num_events"] = len(dj.get("events",[]))
            row["mean_dist"] = dj.get("mean","")
            row["std_dist"] = dj.get("std","")
            row["z_thresh"] = dj.get("z_thresh","")
        drift_rows.append(row)
    parts.append(_table_from_dict("Drift Summary", drift_rows))

    # cluster segments table (top 8 segments)
    cluster_rows = []
    for i, d in enumerate(input_dirs):
        name = session_names[i] if session_names and i < len(session_names) else f"S{i+1}"
        p = Path(d)
        cj = _read_json(p / "cluster_summary.json")
        row = {"_session": name}
        if cj and isinstance(cj, dict):
            segs = cj.get("segments", [])
            row["k"] = cj.get("k","")
            row["pca_dims"] = cj.get("pca_dims","")
            row["segments"] = "; ".join([f"[c{int(s['cluster'])}:{int(s['start'])}-{int(s['end'])}]" for s in segs[:8]])
        cluster_rows.append(row)
    parts.append(_table_from_dict("Cluster Segments", cluster_rows))

    # image grids: pca/umap per session
    for label, fname in [("Trajectory PCA (per session)", "trajectory_pca.png"),
                         ("Trajectory UMAP (per session)", "trajectory_umap.png")]:
        parts.append(f"<div class='card'><h2>{label}</h2><div class='grid'>")
        for i, d in enumerate(input_dirs):
            name = session_names[i] if session_names and i < len(session_names) else f"S{i+1}"
            p = Path(d)
            img = _embed_img(p / fname)
            if img:
                parts.append(f"<div><div class='small'><b>{name}</b></div>{img}</div>")
        parts.append("</div></div>")

    parts.append("</body></html>")
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return output_html

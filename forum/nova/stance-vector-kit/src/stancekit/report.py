
import os, json, base64, datetime
from pathlib import Path

SECTIONS = [
    ("Metrics", "metrics.json"),
    ("Drift", "drift_events.json"),
    ("Clusters", "cluster_summary.json"),
    ("A/B Metrics", "ab_metrics.json"),
    ("Viz: PCA", "trajectory_pca.png"),
    ("Viz: UMAP", "trajectory_umap.png"),
    ("Viz: A/B Overlay", "ab_overlay.png"),
]

def _embed_img(path):
    if not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width: 100%; height: auto;"/>'

def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def generate_report(inputs_dir: str, output_html: str, title: str = "SVK Session Report"):
    p = Path(inputs_dir)
    tstamp = datetime.datetime.utcnow().isoformat() + "Z"
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title>",
             "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;}h1{margin-bottom:4px;} .card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0;} pre{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto;} .small{color:#666;font-size:12px}</style>",
             "</head><body>",
             f"<h1>{title}</h1><div class='small'>Generated {tstamp}</div>"]

    for label, fname in SECTIONS:
        path = p / fname
        parts.append(f"<div class='card'><h2>{label}</h2>")
        if fname.endswith(".json"):
            data = _read_json(path)
            if data is None:
                parts.append("<div class='small'>No data.</div>")
            else:
                pretty = json.dumps(data, indent=2)
                parts.append(f"<pre>{pretty}</pre>")
        elif fname.endswith(".png"):
            if path.exists():
                parts.append(_embed_img(str(path)))
            else:
                parts.append("<div class='small'>No image available.</div>")
        parts.append("</div>")

    # Also include stance_windows.csv preview if present
    sw = p / "stance_windows.csv"
    if sw.exists():
        try:
            import pandas as pd
            df = pd.read_csv(sw).head(20)
            parts.append("<div class='card'><h2>Stance Windows (preview)</h2>")
            parts.append(df.to_html(index=False))
            parts.append("</div>")
        except Exception:
            pass

    parts.append("</body></html>")
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return output_html

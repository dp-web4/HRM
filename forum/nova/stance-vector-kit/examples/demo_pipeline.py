
import os, json, argparse
import numpy as np
import pandas as pd
from stancekit.feature_extraction import compile_lexicons, extract_features, window_iter
from stancekit.stance_classifier import StanceHead, AXES
from stancekit.fuse import ema_vector
from stancekit.eval import cosine_similarity, flicker_index
from stancekit.config import WINDOW_SIZE, WINDOW_STEP

def load_jsonl(path):
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            turns.append(json.loads(ln))
    return turns

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    kit_root = os.path.dirname(os.path.dirname(__file__))
    lex = compile_lexicons(kit_root)
    turns = load_jsonl(args.input)

    feats = extract_features(turns, lex)

    # Bootstrap labels heuristically; replace with your annotations for higher fidelity.
    labels = {ax: np.zeros(len(feats), dtype=int) for ax in AXES}
    for i,f in enumerate(feats):
        labels["EH"][i] = int(f["hedges"] > 0.01)
        labels["DC"][i] = int(f["modals"] > 0.01)
        labels["EX"][i] = int(f["q_ratio"] > 0.15)
        labels["MA"][i] = int(f["meta"] > 0.0)
        labels["RR"][i] = int(f["backtrack"] > 0.0)
        labels["AG"][i] = int(f["action"] > 0.0)
        labels["SV"][i] = int(f["verify"] > 0.0)
        labels["VA"][i] = 1 if (f["pos"]-f["neg"])>0 else 0
        labels["AR"][i] = int(f["exclaim"]>0.0)
        labels["IF"][i] = 1 if f["action"]>0.0 else 0
        labels["ED"][i] = int(f["verify"]>0.0)

    head = StanceHead()
    head.fit(feats, labels)
    probs = head.predict_proba(feats)  # per-turn axis probabilities

    rows = []
    s_prev = None
    series = []
    for idx, window in window_iter(turns, size=WINDOW_SIZE, step=WINDOW_STEP):
        agg = {ax: float(np.mean(probs[ax][idx:idx+WINDOW_SIZE])) for ax in probs}
        s = np.array([agg[ax] for ax in head.axes], dtype=float)
        s_smooth = ema_vector(s_prev, s)
        s_prev = s_smooth
        series.append(s_smooth)
        rows.append({"start_idx": idx, **{f"s_{ax}": float(s_smooth[i]) for i,ax in enumerate(head.axes)}})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir,"stance_windows.csv"), index=False)

    metrics = {"flicker_index": float(flicker_index(series))}

    if args.baseline:
        turns_b = load_jsonl(args.baseline)
        feats_b = extract_features(turns_b, lex)
        probs_b = head.predict_proba(feats_b)
        rows_b, s_prev_b, series_b = [], None, []
        for idx, window in window_iter(turns_b, size=WINDOW_SIZE, step=WINDOW_STEP):
            agg = {ax: float(np.mean(probs_b[ax][idx:idx+WINDOW_SIZE])) for ax in probs_b}
            s = np.array([agg[ax] for ax in head.axes], dtype=float)
            s_smooth = ema_vector(s_prev_b, s); s_prev_b = s_smooth
            series_b.append(s_smooth)
        n = min(len(series), len(series_b))
        if n>0:
            cos = np.mean([cosine_similarity(series[i], series_b[i]) for i in range(n)])
            metrics["cross_context_cosine"] = float(cos)

    with open(os.path.join(args.out_dir,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote:", os.path.join(args.out_dir,"stance_windows.csv"))
    print("Wrote:", os.path.join(args.out_dir,"metrics.json"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl transcript")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--baseline", default=None, help="optional jsonl for cross-context comparison")
    args = ap.parse_args()
    main(args)

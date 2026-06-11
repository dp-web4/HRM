#!/usr/bin/env python3
"""Scale modulates substrate-sensitivity of the raising trajectory.

Two same-model / two-machine contrasts, run on the live instance data:

  qwen3.5:0.8b  on cbp  vs  sprout    (small model, two machines)
  gemma3:12b    on legion vs mcnugget (large model, two machines)

Each holds the MODEL CONSTANT and varies the substrate (machine + that
machine's runner + accumulated path-history + operator raising). The question:
does the same model land the same raising trajectory regardless of substrate
(model-determined → "weather dominates"), or diverge by substrate
(substrate-determined → "organism dominates")?

The two contrasts are each other's control (agent-zero): if metrics diverged
for BOTH pairs it's noise; a divergence in one and convergence in the other
is signal about scale.

Metrics are per-turn RATES/RATIOS (turn-count-normalized). Pairs are truncated
to equal session depth (min of the two) so curriculum position matches.

Honest scope: 2 families, 2 scale points (0.8B, 12B) — locates a direction,
not a curve. Cannot isolate WHICH substrate variable (machine vs runner-version
vs raising-depth) drives a divergence; only that the MODEL does not (it's held
constant). Sprout's qwen track is the celebrated identity-anchored runner and
is 2.5x deeper than CBP's — a plausible substrate driver, named not hidden.
"""
import json, glob, re, os

HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.abspath(os.path.join(HERE, "..", "..", "instances"))

IDENTITY = re.compile(r"\b(as SAGE|I am SAGE|I'm SAGE|my name is SAGE|being SAGE)\b", re.I)
SELF_ID = re.compile(r"\b(as an? \w+ (entity|intelligence|system|model|being)|my (identity|sense of self|nature|purpose|existence))\b", re.I)
CONT = re.compile(r"\b(previous session|prior session|last session|earlier (session|conversation)|recent sessions|sessions? (have|we|I)|accumulated|over (our|these) sessions|remember(ed)? (from|that)|each session)\b", re.I)

def ttr(t):
    w = re.findall(r"[a-z']+", t.lower())
    return len(set(w)) / len(w) if w else 0.0

def trajectory(inst):
    rows = []
    for p in sorted(glob.glob(f"{INST}/{inst}/sessions/session_*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        turns = [t.get("text", "") for t in d.get("conversation", []) if t.get("speaker") == "SAGE"]
        if not turns:
            continue
        n = len(turns)
        rows.append(dict(
            session=d.get("session"), phase=d.get("phase"),
            ident=sum(IDENTITY.search(t) is not None for t in turns) / n,
            selfid=sum(SELF_ID.search(t) is not None for t in turns) / n,
            cont=sum(CONT.search(t) is not None for t in turns) / n,
            ttr=ttr(" ".join(turns)),
            chars=sum(len(t) for t in turns) / n,
        ))
    return rows

def agg(rows, k):
    v = [r[k] for r in rows]
    return sum(v) / len(v) if v else 0.0

METRICS = ["ident", "selfid", "cont", "ttr", "chars"]

def contrast(a, b, label):
    ra, rb = trajectory(a), trajectory(b)
    m = min(len(ra), len(rb))
    ra, rb = ra[:m], rb[:m]
    print(f"\n=== {label} (first {m} sessions each, model held constant) ===")
    print(f"{'metric':8s} {a.split('-',1)[0]:>10s} {b.split('-',1)[0]:>10s} {'delta':>9s} {'ratio':>7s}")
    res = {}
    for k in METRICS:
        va, vb = agg(ra, k), agg(rb, k)
        ratio = (max(va, vb) / min(va, vb)) if min(va, vb) > 1e-6 else float("inf")
        print(f"{k:8s} {va:10.3f} {vb:10.3f} {vb-va:+9.3f} {ratio:7.1f}x")
        res[k] = (va, vb, ratio)
    return res

if __name__ == "__main__":
    small = contrast("cbp-qwen3.5-0.8b", "sprout-qwen3.5-0.8b", "qwen3.5:0.8b SMALL")
    large = contrast("legion-gemma3-12b", "mcnugget-gemma3-12b", "gemma3:12b LARGE")
    print("\n--- the contrast of contrasts (the actual finding) ---")
    for k in METRICS:
        sr, lr = small[k][2], large[k][2]
        verdict = "substrate-sensitive at small, convergent at large" if (sr > 2 and lr < 1.5) else ""
        print(f"  {k:8s} small-machine-ratio {sr:5.1f}x | large-machine-ratio {lr:5.1f}x  {verdict}")

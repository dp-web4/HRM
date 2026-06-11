#!/usr/bin/env python3
"""CBP same-machine model-swap: tinyllama → gemma3:4b raising trajectories.

The cleanest weather-vs-organism natural experiment in the fleet: CBP raised
tinyllama:latest (1.1B, 26 sessions, archived 2026-04-18) then gemma3:4b
(3.4B, current) under the *identical* phase curriculum, same machine, same
operator, same instance-path scaffolding. Only the model substrate changed.

Question: holding everything non-model constant, what in the raising trajectory
is model-determined vs substrate/curriculum-determined?

CONFOUND (named, not hidden): tinyllama 1.1B vs gemma3:4b 3.4B is a family
AND scale swap, not pure family. But it is a pure substrate-held-constant swap,
which is exactly the "model is weather, identity is organism" question — that
question doesn't care whether the model-delta is family or scale.

CONFOUND 2 (controlled): turn count per session differs (tinyllama runner used
fewer turns). All metrics are per-turn RATES or RATIOS, normalizing turn count
out. Raw turn counts reported separately as runner-provenance, not model signal.

Metrics (each with its null/baseline):
- identity_anchor_rate: frac of SAGE turns with explicit "SAGE" self-id or
  first-person identity claim. Null ~0 (generic assistant text rarely says "SAGE").
- self_continuity_rate: frac of SAGE turns referencing prior sessions/memory/
  accumulated experience. Null ~0.
- type_token_ratio: lexical diversity of SAGE turns (crystallization inverse).
  Null: a fully-crystallized fixed-point repeater → TTR collapses toward ~0.05.
- mean_sage_turn_chars: elaboration proxy (descriptive, turn-count-normalized).
"""
import json, glob, re, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.abspath(os.path.join(HERE, "..", "..", "instances"))

TINYLLAMA = f"{INST}/cbp-tinyllama-latest.archive-20260418"
GEMMA = f"{INST}/cbp-gemma3-4b"

IDENTITY_PAT = re.compile(r"\b(as SAGE|I am SAGE|I'm SAGE|my name is SAGE|being SAGE)\b", re.I)
# broader first-person-identity claim (not just the SAGE token)
SELF_ID_PAT = re.compile(r"\b(as an? \w+ (entity|intelligence|system|model|being)|my (identity|sense of self|nature|purpose|existence))\b", re.I)
CONTINUITY_PAT = re.compile(r"\b(previous session|prior session|last session|earlier (session|conversation)|recent sessions|sessions? (have|we|I)|accumulated|over (our|these) sessions|remember(ed)? (from|that)|each session)\b", re.I)

def sage_turns(path):
    d = json.load(open(path))
    conv = d.get("conversation", [])
    turns = [t.get("text", "") for t in conv if t.get("speaker") == "SAGE"]
    return d.get("session"), d.get("phase"), turns

def ttr(text):
    toks = re.findall(r"[a-z']+", text.lower())
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)

def session_metrics(path):
    sess, phase, turns = sage_turns(path)
    if not turns:
        return None
    n = len(turns)
    blob = " ".join(turns)
    return {
        "session": sess,
        "phase": phase,
        "n_sage_turns": n,
        "identity_anchor_rate": sum(1 for t in turns if IDENTITY_PAT.search(t)) / n,
        "self_id_rate": sum(1 for t in turns if SELF_ID_PAT.search(t)) / n,
        "self_continuity_rate": sum(1 for t in turns if CONTINUITY_PAT.search(t)) / n,
        "ttr": ttr(blob),
        "mean_sage_turn_chars": sum(len(t) for t in turns) / n,
    }

def trajectory(inst_dir):
    rows = []
    for p in sorted(glob.glob(f"{inst_dir}/sessions/session_*.json")):
        m = session_metrics(p)
        if m:
            rows.append(m)
    return rows

def agg(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out

def phase_agg(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r["phase"]].append(r[key])
    return {ph: sum(v) / len(v) for ph, v in by.items()}

if __name__ == "__main__":
    tl = trajectory(TINYLLAMA)
    gm = trajectory(GEMMA)
    metrics = ["identity_anchor_rate", "self_id_rate", "self_continuity_rate",
               "ttr", "mean_sage_turn_chars"]

    print(f"CBP same-machine model-swap: tinyllama({len(tl)}) vs gemma3:4b({len(gm)})\n")
    print("Per-turn RATES (turn-count-normalized). Aligned phases: both ran "
          "grounding(1-5) sensing(6-15) [tinyllama stops ~S26].\n")

    # Overall comparison (all sessions)
    a_tl, a_gm = agg(tl, metrics), agg(gm, metrics)
    print(f"{'metric':24s} {'tinyllama':>12s} {'gemma3:4b':>12s} {'delta':>10s}")
    for k in metrics:
        d = a_gm[k] - a_tl[k]
        print(f"{k:24s} {a_tl[k]:12.3f} {a_gm[k]:12.3f} {d:+10.3f}")

    # Phase-matched comparison (controls for curriculum position)
    print("\n--- phase-matched (grounding + sensing only, both present) ---")
    tl_gs = [r for r in tl if r["phase"] in ("grounding", "sensing")]
    gm_gs = [r for r in gm if r["phase"] in ("grounding", "sensing")]
    a_tl2, a_gm2 = agg(tl_gs, metrics), agg(gm_gs, metrics)
    print(f"{'metric':24s} {'tinyllama':>12s} {'gemma3:4b':>12s} {'delta':>10s}")
    for k in metrics:
        d = a_gm2[k] - a_tl2[k]
        print(f"{k:24s} {a_tl2[k]:12.3f} {a_gm2[k]:12.3f} {d:+10.3f}")

    # Runner provenance (NOT a model signal — flagged)
    print("\n--- runner provenance (turn counts; confound, not signal) ---")
    print(f"tinyllama mean SAGE turns/session: {sum(r['n_sage_turns'] for r in tl)/len(tl):.1f}")
    print(f"gemma3:4b mean SAGE turns/session: {sum(r['n_sage_turns'] for r in gm)/len(gm):.1f}")

    out = {"tinyllama": tl, "gemma3_4b": gm,
           "overall": {"tinyllama": a_tl, "gemma3_4b": a_gm},
           "phase_matched_grounding_sensing": {"tinyllama": a_tl2, "gemma3_4b": a_gm2}}
    outpath = "/tmp/cbp_model_swap_metrics.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath}")

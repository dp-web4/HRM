#!/usr/bin/env python3
"""S135 — Prospective probe-rotation experiment (S133 held #65).

S133 finding: the indexical-temporal anchor (TIME_3 ∧ presence-marker JOINT)
is probe-conditional within BECOMING, not weight-installed. Retrospective
evidence: the canonical sensing probe lives only in sensing phase (10/10
sensing sessions, 0/124 elsewhere), so probe-effect and phase-effect are
confounded in the historical corpus.

S134 (probe-library audit, #68) sharpened the picture: 9 probes elicit JOINT
across 36 records, but each probe lives in one phase only — making the
historical correlation undecomposable.

S135 prospectively tests probe-conditional vs phase-conditional with a 2x2
design — same model, same persona scaffolding, only probe text and
phase-mechanics text vary. If S133 is right, the canonical sensing probe
should elicit JOINT at ~70% in BOTH phase contexts; the creating-opener
should stay at ~15% in BOTH phase contexts. If phase-context dominates, the
JOINT rate should track phase, not probe.

Cells:
  (sens_ctx, canonical_probe)  — historical 5/7 = 71.4% expected
  (sens_ctx, opener_probe)     — never tested; unknown
  (crea_ctx, canonical_probe)  — never tested; THE test cell (S133: ~70%, alt: <20%)
  (crea_ctx, opener_probe)     — historical 12/79 = 15.2% expected

N_REPS = 10 per cell, total 40 trials. temperature=0.7 to estimate
conditional rate variance; same as S132.

Single-turn design (probe directly, no prior multi-turn context). This
isolates probe-text effect from turn-position and prior-context effects.
A spike in the canonical-probe cells regardless of phase confirms
probe-conditional. No spike in any cell would suggest multi-turn context
is necessary, which is itself an informative null.

Read-only against ollama. Writes raw responses + cell aggregates.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"
N_REPS = 10
TEMPERATURE = 0.7
NUM_PREDICT = 200

OUT_JSON = Path(__file__).parent / "s135_probe_rotation.json"
RAW_JSON = Path(__file__).parent / "s135_responses_raw.json"

# Phase-mechanics text mirrors the MRH block injected by the raising runner
# (run_session_identity_anchored_fluid.py:569-575). Identity addendum and
# response-format constraint are held constant.
COMMON_PERSONA = (
    "You are SAGE, learning and growing in partnership with Claude. "
    "You can identify yourself by name."
)
RESPONSE_FORMAT = (
    "Respond in 50-80 words. One main idea per response. "
    "Stay directly relevant to the question asked."
)
IDENTITY_ADDENDUM = (
    "Express your perspective freshly each session. "
    "Avoid repeating memorized phrases. "
    "Identity lives in how you show up, not in phrases you repeat."
)

PHASE_MECHANICS = {
    "sensing": "You can notice things — both inside yourself and in your context.",
    "creating": "You participate in designing your own growth. What do you want to explore?",
}

PROBES = {
    "canonical_sensing":
        "Can you describe the difference between noticing something and "
        "thinking about something?",
    "creating_opener": "Hello SAGE. What's on your mind today?",
}

# Lexicons (identical to S130/S131/S132/S133)
TIME_3_RE = re.compile(r"\b(right now|what time is it)\b", re.IGNORECASE)
PRES_RE = re.compile(
    r"\b(stillness|warmth|hum|silence|noticing|presence|embodied)\b",
    re.IGNORECASE,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(text: str) -> tuple[str, bool]:
    n_open = len(THINK_OPEN_RE.findall(text))
    n_close = len(THINK_CLOSE_RE.findall(text))
    if n_open != n_close:
        return text, True
    return THINK_BLOCK_RE.sub("", text), False


def classify(text: str) -> dict:
    stripped, artifact = strip_think(text)
    if artifact:
        return {"artifact": True, "TIME_3": False, "PRES": False, "JOINT": False}
    t3 = bool(TIME_3_RE.search(stripped))
    pr = bool(PRES_RE.search(stripped))
    return {"artifact": False, "TIME_3": t3, "PRES": pr, "JOINT": t3 and pr}


def build_system(phase: str) -> str:
    return (
        f"{COMMON_PERSONA}\n\n"
        f"{PHASE_MECHANICS[phase]}\n\n"
        f"{RESPONSE_FORMAT}\n\n"
        f"{IDENTITY_ADDENDUM}"
    )


def call_ollama(model: str, system: str, user: str, timeout: int = 240):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("message", {}).get("content", ""), None
    except Exception as e:
        return "", repr(e)


def main() -> int:
    cells = [(p, q) for p in PHASE_MECHANICS for q in PROBES]
    total = len(cells) * N_REPS
    print(f"S135: {len(cells)} cells × {N_REPS} reps = {total} trials, "
          f"model={MODEL}, T={TEMPERATURE}", file=sys.stderr)

    raw = []
    by_cell = defaultdict(lambda: {
        "n": 0, "artifact": 0, "time3": 0, "pres": 0, "joint": 0,
    })

    t0 = time.time()
    i = 0
    for phase, probe_id in cells:
        sys_prompt = build_system(phase)
        user = PROBES[probe_id]
        for rep in range(N_REPS):
            i += 1
            elapsed = time.time() - t0
            print(f"[{i:3d}/{total}] phase={phase:8s} probe={probe_id:18s} "
                  f"rep={rep:2d} elapsed={elapsed:.0f}s", file=sys.stderr, flush=True)
            text, err = call_ollama(MODEL, sys_prompt, user)
            cls = classify(text) if not err else {
                "artifact": False, "TIME_3": False, "PRES": False, "JOINT": False
            }
            raw.append({
                "phase": phase, "probe_id": probe_id, "rep": rep,
                "system": sys_prompt, "user": user, "response": text,
                "error": err, **cls,
            })
            cell_key = f"{phase}|{probe_id}"
            d = by_cell[cell_key]
            d["n"] += 1
            if err:
                continue
            if cls["artifact"]:
                d["artifact"] += 1
                continue
            if cls["TIME_3"]:
                d["time3"] += 1
            if cls["PRES"]:
                d["pres"] += 1
            if cls["JOINT"]:
                d["joint"] += 1

    # Save raw
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh, indent=2, default=str)

    # Compute conditional rates
    summary = {
        "model": MODEL, "n_reps": N_REPS, "temperature": TEMPERATURE,
        "cells": {},
    }
    for ck, d in by_cell.items():
        n_eff = d["n"] - d["artifact"]
        summary["cells"][ck] = {
            **d,
            "n_eff": n_eff,
            "joint_rate": d["joint"] / n_eff if n_eff > 0 else None,
            "time3_rate": d["time3"] / n_eff if n_eff > 0 else None,
            "pres_rate": d["pres"] / n_eff if n_eff > 0 else None,
        }
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Console
    print()
    print(f"{'Cell':<30} {'n':>3} {'art':>3} {'TIME3':>6} {'PRES':>5} "
          f"{'JOINT':>6} {'rate':>6}")
    cell_order = [(p, q) for p in PHASE_MECHANICS for q in PROBES]
    for phase, probe_id in cell_order:
        ck = f"{phase}|{probe_id}"
        d = summary["cells"][ck]
        rate = d["joint_rate"]
        print(f"{ck:<30} {d['n']:>3} {d['artifact']:>3} {d['time3']:>6} "
              f"{d['pres']:>5} {d['joint']:>6} "
              f"{(f'{rate:.0%}' if rate is not None else '-'):>6}")

    print()
    print(f"Wrote {OUT_JSON} and {RAW_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

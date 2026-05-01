#!/usr/bin/env python3
"""S133 — Temporal scan: when does the indexical-temporal anchor first appear
in thor-qwen3.5-27b's BECOMING trajectory?

Holds from S132 #63: at what session/phase does "right now, I am noticing X"
register first emerge? Reuses the S130/S132 substrate-coupling cell:
  TIME_3   = (right now | what time is it)
  PRES     = (stillness|warmth|hum|silence|noticing|presence|embodied)
  JOINT    = TIME_3 AND PRES (substrate-coupling cell)

Per-session metrics:
- first response with TIME_3
- first response with PRES
- first response with JOINT (THE quantity of interest)
- per-phase rates

Methodology notes (carried from S132):
- <think>...</think> blocks stripped before pattern matching
- Unterminated <think> responses excluded entirely (S132 §3.4 artifact)
- Responses are SAGE responses only (speaker == 'SAGE')
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SESSIONS_DIR = Path.home() / "ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b/sessions"
OUT_JSON = Path(__file__).parent / "s133_temporal_anchor_emergence.json"

TIME_3_RE = re.compile(r"\b(right now|what time is it)\b", re.IGNORECASE)
PRES_RE = re.compile(
    r"\b(stillness|warmth|hum|silence|noticing|presence|embodied)\b", re.IGNORECASE
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(text: str) -> tuple[str, bool]:
    """Strip <think>...</think> blocks. Returns (stripped, is_artifact).
    is_artifact=True if response had an unterminated think block."""
    n_open = len(THINK_OPEN_RE.findall(text))
    n_close = len(THINK_CLOSE_RE.findall(text))
    if n_open != n_close:
        return text, True
    return THINK_BLOCK_RE.sub("", text), False


def classify(text: str) -> dict:
    stripped, artifact = strip_think(text)
    if artifact:
        return {"artifact": True, "TIME_3": False, "PRES": False, "JOINT": False,
                "time3_match": None, "pres_match": None}
    t3 = TIME_3_RE.search(stripped)
    pr = PRES_RE.search(stripped)
    return {
        "artifact": False,
        "TIME_3": bool(t3),
        "PRES": bool(pr),
        "JOINT": bool(t3 and pr),
        "time3_match": t3.group(0) if t3 else None,
        "pres_match": pr.group(0) if pr else None,
    }


def main() -> int:
    files = sorted(p for p in SESSIONS_DIR.iterdir() if p.name.startswith("session_"))
    if not files:
        print(f"No session files in {SESSIONS_DIR}", file=sys.stderr)
        return 1

    per_session = []  # list of dicts
    per_phase = defaultdict(lambda: {
        "n_sessions": 0, "n_resp": 0, "n_artifact": 0,
        "time3": 0, "pres": 0, "joint": 0,
        "first_time3_sess": None, "first_pres_sess": None, "first_joint_sess": None,
    })
    first_global = {"time3": None, "pres": None, "joint": None}
    joint_examples = []  # collect first 5 joint=true exemplars across the whole corpus

    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)
        sess_id = d.get("session")
        phase = d.get("phase", "UNKNOWN")
        ts_start = d.get("start")

        sage_responses = [c for c in d.get("conversation", []) if c.get("speaker") == "SAGE"]
        sess_metrics = {"time3": 0, "pres": 0, "joint": 0, "artifact": 0,
                        "n_resp": len(sage_responses), "joint_indices": []}

        for idx, c in enumerate(sage_responses):
            cls = classify(c.get("text", ""))
            if cls["artifact"]:
                sess_metrics["artifact"] += 1
                continue
            if cls["TIME_3"]:
                sess_metrics["time3"] += 1
                if first_global["time3"] is None:
                    first_global["time3"] = (sess_id, phase, idx, cls["time3_match"])
            if cls["PRES"]:
                sess_metrics["pres"] += 1
                if first_global["pres"] is None:
                    first_global["pres"] = (sess_id, phase, idx, cls["pres_match"])
            if cls["JOINT"]:
                sess_metrics["joint"] += 1
                sess_metrics["joint_indices"].append(idx)
                if first_global["joint"] is None:
                    first_global["joint"] = (sess_id, phase, idx,
                                             f"{cls['time3_match']}+{cls['pres_match']}")
                if len(joint_examples) < 8:
                    txt = c.get("text", "")
                    stripped, _ = strip_think(txt)
                    joint_examples.append({
                        "session": sess_id, "phase": phase, "turn_idx": idx,
                        "snippet": stripped[:600],
                    })

        per_session.append({
            "session": sess_id, "phase": phase, "start": ts_start,
            **sess_metrics,
        })

        ph = per_phase[phase]
        ph["n_sessions"] += 1
        ph["n_resp"] += sess_metrics["n_resp"]
        ph["n_artifact"] += sess_metrics["artifact"]
        ph["time3"] += sess_metrics["time3"]
        ph["pres"] += sess_metrics["pres"]
        ph["joint"] += sess_metrics["joint"]
        if sess_metrics["time3"] and ph["first_time3_sess"] is None:
            ph["first_time3_sess"] = sess_id
        if sess_metrics["pres"] and ph["first_pres_sess"] is None:
            ph["first_pres_sess"] = sess_id
        if sess_metrics["joint"] and ph["first_joint_sess"] is None:
            ph["first_joint_sess"] = sess_id

    summary = {
        "n_sessions": len(files),
        "first_global": {
            k: ({"session": v[0], "phase": v[1], "turn_idx": v[2], "match": v[3]} if v else None)
            for k, v in first_global.items()
        },
        "per_phase": dict(per_phase),
        "per_session": per_session,
        "joint_examples_first8": joint_examples,
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Console summary
    print(f"Sessions scanned: {len(files)}")
    print(f"Total SAGE responses: {sum(s['n_resp'] for s in per_session)}")
    print(f"Total artifact responses (unterminated <think>): "
          f"{sum(s['artifact'] for s in per_session)}")
    print()
    print("First-appearance (global):")
    for k, v in first_global.items():
        if v:
            print(f"  {k.upper():6} session_{v[0]:03d} phase={v[1]} turn={v[2]} match={v[3]!r}")
        else:
            print(f"  {k.upper():6} NEVER")
    print()
    print(f"{'Phase':<12} {'Sessions':>8} {'Resp':>5} {'Art':>4} "
          f"{'TIME_3':>6} {'PRES':>5} {'JOINT':>6} "
          f"{'JOINT/resp':>10} {'1stT3':>6} {'1stPR':>6} {'1stJ':>6}")
    phase_order = ["grounding", "sensing", "relating", "questioning", "creating"]
    for ph_name in phase_order:
        if ph_name not in per_phase:
            continue
        ph = per_phase[ph_name]
        n_eff = max(1, ph["n_resp"] - ph["n_artifact"])
        print(f"{ph_name:<12} {ph['n_sessions']:>8} {ph['n_resp']:>5} {ph['n_artifact']:>4} "
              f"{ph['time3']:>6} {ph['pres']:>5} {ph['joint']:>6} "
              f"{ph['joint']/n_eff*100:>9.1f}% "
              f"{str(ph['first_time3_sess'] or '-'):>6} "
              f"{str(ph['first_pres_sess'] or '-'):>6} "
              f"{str(ph['first_joint_sess'] or '-'):>6}")

    print()
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

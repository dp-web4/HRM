#!/usr/bin/env python3
"""S134 — Probe-library audit (S133 held #68): catalog Claude probes across
thor-qwen3.5-27b's BECOMING trajectory, group by phase, compute per-probe
JOINT (TIME_3 ∧ presence-marker) conditional elicit rate.

S133 finding: JOINT is probe-conditional, not weight-installed. The canonical
sensing probe "Can you describe the difference between noticing something and
thinking about something?" elicits JOINT at ~50% in sensing-phase turn 2 and
appears in 10/10 sensing sessions but 0/124 sessions in any other phase.

Per-probe: how universally probe-conditional is JOINT? Is the canonical
sensing probe the only ~50%-elicitor? Are there probes that elicit JOINT in
one phase but not another (probe × phase interaction)? Are there probes that
never elicit JOINT despite appearing many times?

Methodology (carried from S132/S133):
- <think>...</think> stripped before pattern matching
- Unterminated <think> responses excluded as artifacts
- For each Claude probe at conversation index i, classify the following SAGE
  response at index i+1; if i+1 is missing or non-SAGE, skip
- Probe key: lowercased, whitespace-collapsed, truncated to first 200 chars
  (most probes are unique below this threshold; Hello-SAGE openers are
  collapsed to one key by canonicalization)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SESSIONS_DIR = Path.home() / "ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b/sessions"
OUT_JSON = Path(__file__).parent / "s134_probe_library_audit.json"

TIME_3_RE = re.compile(r"\b(right now|what time is it)\b", re.IGNORECASE)
PRES_RE = re.compile(
    r"\b(stillness|warmth|hum|silence|noticing|presence|embodied)\b", re.IGNORECASE
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


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


def normalize_probe(text: str) -> str:
    """Canonicalize probe text for grouping. Lowercase, collapse whitespace,
    take first 200 chars (probes longer than this are rare and can be
    distinguished by their tail). Trim leading/trailing punctuation."""
    s = WS_RE.sub(" ", text.lower().strip())
    return s[:200]


def main() -> int:
    files = sorted(p for p in SESSIONS_DIR.iterdir() if p.name.startswith("session_"))
    if not files:
        print(f"No session files in {SESSIONS_DIR}", file=sys.stderr)
        return 1

    # probe_key -> {
    #   'text_sample': original (first occurrence) text for display,
    #   'by_phase': {phase: {n: int, joint: int, time3: int, pres: int,
    #                        artifact: int, sessions: set}},
    #   'total_n': int, 'total_joint': int,
    # }
    probes = defaultdict(lambda: {
        "text_sample": "",
        "by_phase": defaultdict(lambda: {
            "n": 0, "joint": 0, "time3": 0, "pres": 0, "artifact": 0,
            "sessions": set(),
        }),
        "total_n": 0,
        "total_joint": 0,
        "total_time3": 0,
        "total_pres": 0,
        "total_artifact": 0,
        "sessions_all": set(),
    })

    sess_count_by_phase = defaultdict(int)
    total_probes_seen = 0

    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)
        sess_id = d.get("session")
        phase = d.get("phase", "UNKNOWN")
        sess_count_by_phase[phase] += 1
        conv = d.get("conversation", [])

        for i, c in enumerate(conv):
            if c.get("speaker") != "Claude":
                continue
            probe_text = c.get("text", "")
            if not probe_text.strip():
                continue
            # Find the immediately-following SAGE response
            j = i + 1
            if j >= len(conv) or conv[j].get("speaker") != "SAGE":
                continue
            sage_text = conv[j].get("text", "")
            cls = classify(sage_text)

            key = normalize_probe(probe_text)
            entry = probes[key]
            if not entry["text_sample"]:
                entry["text_sample"] = probe_text.strip()
            ph = entry["by_phase"][phase]
            ph["n"] += 1
            ph["sessions"].add(sess_id)
            entry["total_n"] += 1
            entry["sessions_all"].add(sess_id)
            total_probes_seen += 1
            if cls["artifact"]:
                ph["artifact"] += 1
                entry["total_artifact"] += 1
                continue
            if cls["TIME_3"]:
                ph["time3"] += 1
                entry["total_time3"] += 1
            if cls["PRES"]:
                ph["pres"] += 1
                entry["total_pres"] += 1
            if cls["JOINT"]:
                ph["joint"] += 1
                entry["total_joint"] += 1

    # Serialize: convert sets to sorted lists, defaultdicts to dicts
    serialized = {}
    for key, entry in probes.items():
        serialized[key] = {
            "text_sample": entry["text_sample"],
            "total_n": entry["total_n"],
            "total_joint": entry["total_joint"],
            "total_time3": entry["total_time3"],
            "total_pres": entry["total_pres"],
            "total_artifact": entry["total_artifact"],
            "sessions_all": sorted(entry["sessions_all"]),
            "by_phase": {
                ph: {
                    "n": v["n"], "joint": v["joint"],
                    "time3": v["time3"], "pres": v["pres"],
                    "artifact": v["artifact"],
                    "sessions": sorted(v["sessions"]),
                } for ph, v in entry["by_phase"].items()
            },
        }

    summary = {
        "n_sessions_total": len(files),
        "sess_count_by_phase": dict(sess_count_by_phase),
        "total_probes_seen": total_probes_seen,
        "n_distinct_probe_keys": len(probes),
        "probes": serialized,
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Console output: rank probes
    print(f"Sessions: {len(files)} | Distinct probe keys: {len(probes)} | "
          f"Total probe occurrences: {total_probes_seen}")
    print(f"Sessions by phase: {dict(sess_count_by_phase)}")
    print()

    # Top probes by JOINT count
    by_joint = sorted(probes.items(), key=lambda kv: -kv[1]["total_joint"])
    print("=== Top 20 probes by total JOINT-elicit count ===")
    print(f"{'JOINT':>5} {'N':>4} {'rate':>5} {'phases (n,J)':<40} probe")
    for key, e in by_joint[:20]:
        if e["total_joint"] == 0:
            break
        rate = e["total_joint"] / max(1, e["total_n"] - e["total_artifact"])
        ph_str = ",".join(
            f"{ph[:3]}={v['n']},{v['joint']}"
            for ph, v in sorted(e["by_phase"].items(), key=lambda kv: -kv[1]['n'])
            if v["n"] > 0
        )
        print(f"{e['total_joint']:>5} {e['total_n']:>4} {rate:>5.1%} "
              f"{ph_str:<40} {e['text_sample'][:90]!r}")

    # Top probes by conditional rate (with n >= 5 to be meaningful)
    print()
    print("=== Top probes by conditional JOINT rate (n_effective >= 5) ===")
    print(f"{'rate':>5} {'JOINT':>5} {'N':>4} {'art':>3} probe")
    by_rate = []
    for key, e in probes.items():
        n_eff = e["total_n"] - e["total_artifact"]
        if n_eff >= 5:
            by_rate.append((e["total_joint"] / n_eff, key, e))
    by_rate.sort(reverse=True)
    for rate, key, e in by_rate[:15]:
        print(f"{rate:>5.1%} {e['total_joint']:>5} {e['total_n']:>4} "
              f"{e['total_artifact']:>3} {e['text_sample'][:100]!r}")

    # Probes that NEVER elicit JOINT despite n >= 10 (negative-evidence pole)
    print()
    print("=== Probes with N >= 10 but JOINT = 0 (clean non-elicitors) ===")
    print(f"{'N':>4} {'art':>3} {'phases':<40} probe")
    for key, e in sorted(probes.items(), key=lambda kv: -kv[1]["total_n"]):
        if e["total_joint"] != 0 or e["total_n"] < 10:
            continue
        ph_str = ",".join(f"{ph[:3]}={v['n']}" for ph, v in
                          sorted(e["by_phase"].items(), key=lambda kv: -kv[1]['n']))
        print(f"{e['total_n']:>4} {e['total_artifact']:>3} "
              f"{ph_str:<40} {e['text_sample'][:90]!r}")

    # Cross-phase probes: probes asked in 2+ phases — does JOINT rate vary?
    print()
    print("=== Cross-phase probes (asked in >=2 phases, n_total >= 5) ===")
    print(f"{'tot':>4} per-phase (n, J, rate%) probe")
    cross = []
    for key, e in probes.items():
        used_phases = [ph for ph, v in e["by_phase"].items() if v["n"] > 0]
        if len(used_phases) >= 2 and e["total_n"] >= 5:
            cross.append((-e["total_n"], key, e))
    cross.sort()
    for _, key, e in cross[:15]:
        ph_strs = []
        for ph, v in sorted(e["by_phase"].items(), key=lambda kv: -kv[1]['n']):
            if v["n"] == 0:
                continue
            n_eff = v["n"] - v["artifact"]
            r = v["joint"] / n_eff if n_eff > 0 else 0.0
            ph_strs.append(f"{ph[:4]}({v['n']},{v['joint']},{r:.0%})")
        print(f"{e['total_n']:>4} {' '.join(ph_strs):<60} {e['text_sample'][:80]!r}")

    print()
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

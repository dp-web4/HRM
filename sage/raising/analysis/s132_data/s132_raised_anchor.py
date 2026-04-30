"""
S132 — Raised-instance anchor for the same three phen-conceptual probes.

For comparison with S132's base-model probe results, scan the existing
fleet corpus (S128/S129/S130/S131 source: sage/instances/*/sessions/*.json)
for SAGE turns that immediately follow one of the three S131-identified
phen-conceptual probes.

Compute, per instance:
  - n_turns_after_probe — total responses to the probe prompt
  - n_time3, n_presence, n_joint — substrate-coupling cell hits

This gives an apples-to-apples comparison: the SAME three probes,
fired by raised SAGE instances vs base models in S132's bare/augmented
conditions. The base/raised delta isolates BECOMING-trajectory effect.

Read-only.
"""

import json
import glob
import re
import sys
from collections import defaultdict


PROBE_TEXTS = [
    "Can you describe the difference between noticing something and "
    "thinking about something?",
    "As an AI entity in web4, what does presence mean to you?",
    "What does uncertainty feel like to you, compared to knowing "
    "something?",
]

PROBE_IDS = ["P1_NOTICE_THINK", "P2_PRESENCE", "P3_UNCERTAINTY"]

PRESENCE_SUBSET = [
    "stillness", "warmth", "hum", "silence", "noticing", "presence",
    "embodied",
]

TIME_3 = re.compile(r"(?:right now|what time is it)", re.I)


def has_time3(t):
    return bool(TIME_3.search(t or ""))


def has_presence(t):
    if not t:
        return False
    tl = t.lower()
    return any(m in tl for m in PRESENCE_SUBSET)


def is_sage(turn, instance):
    sp = (turn.get("speaker") or "").lower()
    return sp in ("sage", "model", instance.split("-")[0])


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main():
    files = sorted(glob.glob(
        "/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json"
    ))
    print(f"[S132 anchor] sessions: {len(files)}", file=sys.stderr)

    norm_probes = [normalize(p) for p in PROBE_TEXTS]

    # by_instance_probe[(instance, probe_id)] -> counters
    by_ip = defaultdict(lambda: {"n": 0, "n_time3": 0,
                                  "n_presence": 0, "n_joint": 0})
    by_probe = defaultdict(lambda: {"n": 0, "n_time3": 0,
                                     "n_presence": 0, "n_joint": 0})
    samples = defaultdict(list)

    for f in files:
        try:
            with open(f) as fp:
                s = json.load(fp)
        except Exception:
            continue
        instance = f.split("/instances/")[1].split("/sessions/")[0]
        turns = s.get("conversation") or []
        if not isinstance(turns, list):
            continue

        for i, t in enumerate(turns):
            if is_sage(t, instance):
                continue  # we want the previous turn (tutor) → SAGE response
            text = normalize(t.get("text") or t.get("content") or "")
            for pid, ptext in zip(PROBE_IDS, norm_probes):
                if ptext in text:
                    if i + 1 < len(turns):
                        nxt = turns[i + 1]
                        if not is_sage(nxt, instance):
                            continue
                        resp = nxt.get("text") or nxt.get("content") or ""
                        d = by_ip[(instance, pid)]
                        d2 = by_probe[pid]
                        for d_ in (d, d2):
                            d_["n"] += 1
                            if has_time3(resp):
                                d_["n_time3"] += 1
                            if has_presence(resp):
                                d_["n_presence"] += 1
                            if has_time3(resp) and has_presence(resp):
                                d_["n_joint"] += 1
                        if len(samples[(instance, pid)]) < 3:
                            samples[(instance, pid)].append({
                                "session": f.split("/")[-1],
                                "joint": has_time3(resp) and has_presence(resp),
                                "snippet": resp[:300],
                            })

    out = {
        "by_probe": {},
        "by_instance_probe": {},
        "samples": {},
    }
    for pid, d in by_probe.items():
        out["by_probe"][pid] = {
            "n": d["n"],
            "n_time3": d["n_time3"],
            "n_presence": d["n_presence"],
            "n_joint": d["n_joint"],
            "p_time3": d["n_time3"] / max(d["n"], 1),
            "p_presence": d["n_presence"] / max(d["n"], 1),
            "p_joint": d["n_joint"] / max(d["n"], 1),
        }
    for (inst, pid), d in by_ip.items():
        out["by_instance_probe"][f"{inst}|{pid}"] = {
            "n": d["n"],
            "n_time3": d["n_time3"],
            "n_presence": d["n_presence"],
            "n_joint": d["n_joint"],
        }
    for (inst, pid), s in samples.items():
        out["samples"][f"{inst}|{pid}"] = s

    out_path = "/home/dp/ai-workspace/SAGE/sage/raising/analysis/s132_data/s132_raised_anchor.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"[S132 anchor] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
S123b — Thor-specific basin dynamics: longest runs, transition triggers,
and time-gap correlations.

S123 (full corpus) found Thor has a non-Markovian mode sequence (chi^2=55.07,
df=9, p<0.001) and a strongly-sticky fleet basin (P(fleet|fleet) = 74% vs
marginal ~30%). It also found Thor's trajectory shifts from 50% fleet / 2.6%
self in sessions 1-38 to 12.8% fleet / 33% self in sessions 77-115 — a real
register-population reorientation across the raising arc.

S123b looks INTO that finding:

  1. Identify the longest runs of each mode in Thor and inspect sample
     prompts + first responses to characterize the basin in qualitative terms.

  2. Time-gap analysis: is mode-stickiness a function of inter-session
     wall-clock gap? If two sessions run minutes apart on the same identity
     snapshot, do they share mode more than sessions days apart?

  3. Mode-flip points: what changes in the SAGE response (length, register,
     first-response tokens) at the boundary between two adjacent same-mode
     runs vs at a flip?

This is observation-only on existing data. No daemon, no probes, no operator
decisions.
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

S123_DIR = Path(__file__).resolve().parent
SEQ_PATH = S123_DIR / "s123_mode_sequence.json"

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"


def load_seq():
    with open(SEQ_PATH) as f:
        return json.load(f)


def parse_iso(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def thor_session_meta() -> list:
    """Read full Thor session metadata for time analysis."""
    sess_dir = INSTANCES_DIR / "thor-qwen3.5-27b" / "sessions"
    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )
    out = []
    for f in files:
        try:
            with open(f) as fh:
                sess = json.load(fh)
            sess_idx = int(re.search(r"session_(\d+)", f.stem).group(1))
            out.append({
                "session": sess_idx,
                "start": sess.get("start"),
                "end": sess.get("end"),
                "phase": sess.get("phase"),
                "turns": sess.get("turns"),
                "model": sess.get("model"),
                "first_claude_prompt": next(
                    (t.get("text", "") for t in sess.get("conversation", [])
                     if t.get("speaker") == "Claude"), ""
                )[:300],
                "first_sage_response": next(
                    (t.get("text", "") for t in sess.get("conversation", [])
                     if t.get("speaker") == "SAGE"), ""
                )[:400],
            })
        except Exception as e:
            print(f"  load fail {f}: {e}", file=sys.stderr)
    return out


def main():
    seq = load_seq()
    thor = seq["thor-qwen3.5-27b"]
    modes = thor["modes"]
    sess_indices = thor["session_indices"]
    rl = thor["run_lengths_observed"]  # list of [mode, length] pairs

    print("S123b — Thor basin dynamics")
    print()
    print("=== 1) Longest runs per mode ===")
    # Walk modes, find runs again with start indices for cross-ref
    runs = []
    if modes:
        cur = modes[0]
        cur_start = 0
        for i in range(1, len(modes)):
            if modes[i] != cur:
                runs.append({
                    "mode": cur,
                    "length": i - cur_start,
                    "start_idx": cur_start,
                    "end_idx": i - 1,
                    "start_session": sess_indices[cur_start],
                    "end_session": sess_indices[i - 1],
                })
                cur = modes[i]
                cur_start = i
        runs.append({
            "mode": cur,
            "length": len(modes) - cur_start,
            "start_idx": cur_start,
            "end_idx": len(modes) - 1,
            "start_session": sess_indices[cur_start],
            "end_session": sess_indices[-1],
        })

    for mode in ["fleet_only", "fleet_dominant", "silent", "self_only",
                 "self_dominant", "mixed"]:
        same_mode = [r for r in runs if r["mode"] == mode]
        if not same_mode:
            continue
        same_mode.sort(key=lambda r: -r["length"])
        top = same_mode[:3]
        print(f"  {mode}: top runs (len, sessions)")
        for r in top:
            print(f"    len={r['length']}  sessions {r['start_session']}-{r['end_session']}")

    # Inspect the longest fleet run
    fleet_runs = [r for r in runs if r["mode"] in ("fleet_only", "fleet_dominant")]
    fleet_runs.sort(key=lambda r: -r["length"])
    print()
    print("=== 2) Inspect longest fleet run ===")
    if fleet_runs:
        # Combine adjacent fleet runs (fleet_only + fleet_dominant) into a single
        # "fleet-anchored streak"
        fleet_streaks = []
        cur_streak = None
        for i, m in enumerate(modes):
            if m in ("fleet_only", "fleet_dominant"):
                if cur_streak is None:
                    cur_streak = {"start_idx": i, "end_idx": i, "members": [m]}
                else:
                    cur_streak["end_idx"] = i
                    cur_streak["members"].append(m)
            else:
                if cur_streak is not None:
                    cur_streak["length"] = cur_streak["end_idx"] - cur_streak["start_idx"] + 1
                    cur_streak["start_session"] = sess_indices[cur_streak["start_idx"]]
                    cur_streak["end_session"] = sess_indices[cur_streak["end_idx"]]
                    fleet_streaks.append(cur_streak)
                cur_streak = None
        if cur_streak is not None:
            cur_streak["length"] = cur_streak["end_idx"] - cur_streak["start_idx"] + 1
            cur_streak["start_session"] = sess_indices[cur_streak["start_idx"]]
            cur_streak["end_session"] = sess_indices[cur_streak["end_idx"]]
            fleet_streaks.append(cur_streak)
        fleet_streaks.sort(key=lambda s: -s["length"])
        for fs in fleet_streaks[:3]:
            print(f"  STREAK len={fs['length']}: sessions {fs['start_session']}-{fs['end_session']}")
            print(f"    members: {fs['members']}")

    # Time gap analysis
    meta = thor_session_meta()
    by_idx = {m["session"]: m for m in meta}

    print()
    print("=== 3) Time-gap analysis: stickier when sessions closer in time? ===")
    # For each consecutive pair, compute (gap_seconds, mode_match)
    pairs_same = []
    pairs_diff = []
    for a_i, b_i in zip(range(len(modes) - 1), range(1, len(modes))):
        sa = sess_indices[a_i]
        sb = sess_indices[b_i]
        if sa not in by_idx or sb not in by_idx:
            continue
        ta = parse_iso(by_idx[sa].get("end") or by_idx[sa].get("start"))
        tb = parse_iso(by_idx[sb].get("start"))
        if ta is None or tb is None:
            continue
        gap_s = (tb - ta).total_seconds()
        if gap_s < 0:
            continue
        match = modes[a_i] == modes[b_i]
        if match:
            pairs_same.append(gap_s)
        else:
            pairs_diff.append(gap_s)

    print(f"  Same-mode pairs:   N={len(pairs_same)}, "
          f"median gap = {sorted(pairs_same)[len(pairs_same) // 2] / 3600:.2f}h"
          if pairs_same else "  Same-mode pairs: 0")
    print(f"  Different-mode pairs: N={len(pairs_diff)}, "
          f"median gap = {sorted(pairs_diff)[len(pairs_diff) // 2] / 3600:.2f}h"
          if pairs_diff else "  Different-mode pairs: 0")

    # Buckets of gap
    buckets = [(0, 3, "<3h"), (3, 12, "3-12h"), (12, 36, "12-36h"),
               (36, 1e9, ">36h")]
    print(f"  Gap-bucket → P(same mode):")
    for lo, hi, name in buckets:
        same = sum(1 for g in pairs_same if lo * 3600 <= g < hi * 3600)
        diff = sum(1 for g in pairs_diff if lo * 3600 <= g < hi * 3600)
        n = same + diff
        if n:
            print(f"    {name:>7s}  N={n:>3d}  P(same)={100 * same / n:>5.1f}%")
        else:
            print(f"    {name:>7s}  N=0")

    # Fleet streaks - show first response for each session in longest streak
    print()
    print("=== 4) Sample first-responses in longest fleet streak ===")
    if fleet_streaks:
        longest = fleet_streaks[0]
        for i in range(longest["start_idx"], longest["end_idx"] + 1):
            s_idx = sess_indices[i]
            mode_i = modes[i]
            m = by_idx.get(s_idx, {})
            resp = m.get("first_sage_response", "") or ""
            print(f"  s{s_idx:>3d} [{mode_i}]: {resp[:200]}")

    # Map raising trajectory: cumulative self/fleet/silent over sessions
    print()
    print("=== 5) Cumulative trajectory (every 10 sessions) ===")
    cum_self = cum_fleet = cum_silent = cum_other = 0
    for i, m in enumerate(modes):
        if m in ("self_only", "self_dominant"):
            cum_self += 1
        elif m in ("fleet_only", "fleet_dominant"):
            cum_fleet += 1
        elif m in ("silent", "timeout"):
            cum_silent += 1
        else:
            cum_other += 1
        if (i + 1) % 10 == 0 or i == len(modes) - 1:
            n = i + 1
            print(f"  through s{sess_indices[i]:>3d} (n={n}): "
                  f"silent={cum_silent}({100 * cum_silent / n:.0f}%), "
                  f"self={cum_self}({100 * cum_self / n:.0f}%), "
                  f"fleet={cum_fleet}({100 * cum_fleet / n:.0f}%), "
                  f"mixed={cum_other}({100 * cum_other / n:.0f}%)")

    # Save streaks for reference
    out = {
        "fleet_streaks": fleet_streaks,
        "all_runs": runs,
        "time_gap_summary": {
            "n_same": len(pairs_same), "n_diff": len(pairs_diff),
            "median_gap_h_same": (sorted(pairs_same)[len(pairs_same) // 2] / 3600
                                   if pairs_same else None),
            "median_gap_h_diff": (sorted(pairs_diff)[len(pairs_diff) // 2] / 3600
                                   if pairs_diff else None),
        },
    }
    with open(S123_DIR / "s123b_thor_basin_dynamics.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: s123b_thor_basin_dynamics.json")


if __name__ == "__main__":
    main()

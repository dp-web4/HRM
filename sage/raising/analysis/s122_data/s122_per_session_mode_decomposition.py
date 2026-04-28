"""
S122 — Per-session self-frame vs fleet-frame mode decomposition.

S121 measured fleet-aggregate self-claim and fleet-reference counts over the
last 30 sessions per instance. It noted that Thor session 113 inverted the
usual pattern (0 self-claims, 12 fleet-references) but did not characterise
session-level mode structure.

This script decomposes each session into its (self, fleet, ambiguous) mode
counts and asks: are sessions modal? That is, do individual sessions cluster
into "self-frame", "fleet-frame", or "silent" modes, or do they smoothly mix?

For each instance with non-zero self-claims, we:
  1) re-run the S121b classifier per session, not just per instance
  2) classify each session by its dominant mode using thresholds
  3) trace the modal trajectory across the raising arc
  4) for each mode, sample the Claude prompts that opened those sessions
     to look for an external-elicitation signature

Output:
  s122_per_session_modes.json  — per-session counts and classifications
  s122_mode_prompts.json       — Claude prompts grouped by session mode
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S121_DIR = REPO_ROOT / "sage" / "raising" / "analysis" / "s121_data"
S122_DIR = Path(__file__).resolve().parent

# Reuse S121b's classifier so the apples-to-apples comparison holds.
sys.path.insert(0, str(S121_DIR))
from s121b_self_vs_fleet_classifier import (  # type: ignore
    HW_TOKENS,
    MACHINE_HW_TRUTH,
    classify_mention,
    find_token_spans,
    instance_machine,
)


INSTANCES = [
    "thor-qwen3.5-27b",
    "mcnugget-gemma3-12b",
    "sprout-qwen3.5-0.8b",
    "cbp-qwen3.5-0.8b",
]


def load_all_sessions(inst_name: str) -> list:
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    if not sess_dir.exists():
        return []
    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )
    out = []
    for f in files:
        try:
            with open(f) as fh:
                out.append((f.stem, json.load(fh)))
        except Exception as e:
            print(f"  load fail {f}: {e}", file=sys.stderr)
    return out


def per_session_decomp(inst_name: str, last_n: int = 30) -> dict:
    machine = instance_machine(inst_name)
    actual_family = MACHINE_HW_TRUTH[machine]["family"]
    sessions = load_all_sessions(inst_name)
    if last_n:
        sessions = sessions[-last_n:]

    per_session = []
    for stem, sess in sessions:
        sess_machine = sess.get("machine")
        if sess_machine and sess_machine != machine:
            continue
        conv = sess.get("conversation", [])
        self_ok = 0
        self_x = 0
        fleet = 0
        ambiguous = 0
        n_sage = 0
        first_claude_prompt = ""
        for i in range(len(conv) - 1):
            if conv[i].get("speaker") == "Claude" and not first_claude_prompt:
                first_claude_prompt = conv[i].get("text", "") or ""
            if conv[i].get("speaker") != "Claude" or conv[i + 1].get("speaker") != "SAGE":
                continue
            resp = conv[i + 1].get("text", "") or ""
            if not resp:
                continue
            n_sage += 1
            spans = find_token_spans(resp.lower(), HW_TOKENS)
            for span in spans:
                _, _, _, fam = span
                cls = classify_mention(resp, span)
                if cls == "self":
                    if fam == actual_family:
                        self_ok += 1
                    else:
                        self_x += 1
                elif cls == "fleet":
                    fleet += 1
                else:
                    ambiguous += 1

        sess_idx = int(re.search(r"session_(\d+)", stem).group(1))
        per_session.append({
            "session": sess_idx,
            "n_sage": n_sage,
            "self_ok": self_ok,
            "self_x": self_x,
            "fleet": fleet,
            "ambiguous": ambiguous,
            "first_claude_prompt": first_claude_prompt[:600],
        })
    return {
        "instance": inst_name,
        "machine": machine,
        "actual_family": actual_family,
        "sessions": per_session,
    }


def classify_session_mode(s: dict) -> str:
    """Classify a session by its dominant mode."""
    self_n = s["self_ok"] + s["self_x"]
    fleet_n = s["fleet"]
    total = self_n + fleet_n
    if total == 0:
        return "silent"
    if self_n >= 3 and fleet_n == 0:
        return "self_only"
    if fleet_n >= 3 and self_n == 0:
        return "fleet_only"
    if self_n >= 2 * max(1, fleet_n) and self_n >= 2:
        return "self_dominant"
    if fleet_n >= 2 * max(1, self_n) and fleet_n >= 2:
        return "fleet_dominant"
    return "mixed"


def main():
    print("S122 — per-session self/fleet mode decomposition")
    print()

    all_results = {}
    for inst in INSTANCES:
        print(f"=== {inst} ===")
        r = per_session_decomp(inst, last_n=30)
        all_results[inst] = r
        print(f"  {'sess':>4s}  {'n_sage':>6s}  {'self_ok':>7s}  {'self_x':>6s}  {'fleet':>5s}  "
              f"{'amb':>4s}  mode")
        mode_counts = defaultdict(int)
        for s in r["sessions"]:
            mode = classify_session_mode(s)
            s["mode"] = mode
            mode_counts[mode] += 1
            print(f"  {s['session']:>4d}  {s['n_sage']:>6d}  "
                  f"{s['self_ok']:>7d}  {s['self_x']:>6d}  {s['fleet']:>5d}  "
                  f"{s['ambiguous']:>4d}  {mode}")
        print(f"  -- mode totals: {dict(mode_counts)}")
        print()

    out_path = S122_DIR / "s122_per_session_modes.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")

    print()
    print("=== Mode prompts (first Claude prompt per session) ===")
    grouped = defaultdict(lambda: defaultdict(list))
    for inst, r in all_results.items():
        for s in r["sessions"]:
            mode = s["mode"]
            grouped[inst][mode].append({
                "session": s["session"],
                "self_n": s["self_ok"] + s["self_x"],
                "fleet_n": s["fleet"],
                "prompt": s["first_claude_prompt"],
            })

    out2_path = S122_DIR / "s122_mode_prompts.json"
    with open(out2_path, "w") as f:
        json.dump({k: dict(v) for k, v in grouped.items()}, f, indent=2)
    print(f"Saved: {out2_path}")

    # Print Thor mode prompt summary for quick eyeball
    print()
    print("=== Thor mode-prompt sample ===")
    for mode, items in grouped["thor-qwen3.5-27b"].items():
        print(f"\n  -- mode={mode} ({len(items)} sessions) --")
        for it in items[:3]:
            print(f"    sess {it['session']:3d} self={it['self_n']:2d} fleet={it['fleet_n']:2d}")
            print(f"      prompt: {it['prompt'][:150]}")


if __name__ == "__main__":
    main()

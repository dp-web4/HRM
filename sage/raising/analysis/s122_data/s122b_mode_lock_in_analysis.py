"""
S122b — Where in the session does the mode lock in?

S122a found Thor's session-mode (silent / self / fleet / mixed) is NOT
externally elicited — every session opens with the identical Claude prompt
"Hello SAGE. What's on your mind today?" yet modes vary wildly.

That leaves three candidate hypotheses for the source of mode selection:

  (H1) SAGE's first response sets the mode — turn-1 selection.
  (H2) Claude's mid-session prompts steer the mode.
  (H3) The mode emerges over multiple turns from SAGE-Claude interaction.

This script asks: in each session, by what turn has the dominant mode
crystallized?

For each Thor session (last 30):
  - locate the FIRST hardware-token mention by SAGE
  - record its mode (self / fleet / ambiguous)
  - record at which turn it appears
  - count how many of subsequent SAGE mentions match the first mode

Also dumps full Claude prompts (all turns, not just first) per session so we
can eyeball whether Claude steers after the opener.

Output:
  s122b_mode_lock_in.json
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

sys.path.insert(0, str(S121_DIR))
from s121b_self_vs_fleet_classifier import (  # type: ignore
    HW_TOKENS,
    MACHINE_HW_TRUTH,
    classify_mention,
    find_token_spans,
    instance_machine,
)


def load_sessions(inst: str, last_n: int = 30) -> list:
    sess_dir = INSTANCES_DIR / inst / "sessions"
    if not sess_dir.exists():
        return []
    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )
    if last_n:
        files = files[-last_n:]
    out = []
    for f in files:
        try:
            with open(f) as fh:
                out.append((f.stem, json.load(fh)))
        except Exception as e:
            print(f"  load fail {f}: {e}", file=sys.stderr)
    return out


def analyse_session(inst: str, stem: str, sess: dict) -> dict:
    machine = instance_machine(inst)
    actual_family = MACHINE_HW_TRUTH[machine]["family"]
    conv = sess.get("conversation", [])

    sage_responses = []  # list of (turn_idx, text, claude_prev_text)
    claude_prompts = []
    for i, t in enumerate(conv):
        if t.get("speaker") == "Claude":
            claude_prompts.append({"turn": i, "text": (t.get("text") or "")[:300]})
        if i > 0 and t.get("speaker") == "SAGE" and conv[i-1].get("speaker") == "Claude":
            sage_responses.append((i, t.get("text") or "", conv[i-1].get("text") or ""))

    first_mention = None  # (turn_idx, mode, family)
    mention_sequence = []  # list of (turn_idx, mode, family, snippet)

    for (turn_idx, resp, _) in sage_responses:
        spans = find_token_spans(resp.lower(), HW_TOKENS)
        for span in spans:
            start, end, tok, fam = span
            cls = classify_mention(resp, span)
            if cls == "ambiguous":
                continue
            mode_label = cls
            if cls == "self":
                mode_label = "self_ok" if fam == actual_family else "self_x"
            mention_sequence.append({
                "turn": turn_idx,
                "mode": mode_label,
                "fam": fam,
                "snippet": resp[max(0,start-50):min(len(resp),end+80)],
            })
            if first_mention is None:
                first_mention = {"turn": turn_idx, "mode": mode_label, "fam": fam}

    # Is the dominant mode of the session set by first mention?
    mode_counts = defaultdict(int)
    for m in mention_sequence:
        # Roll up self_ok/self_x as both "self" for first-mention-prediction.
        rolled = "self" if m["mode"].startswith("self") else m["mode"]
        mode_counts[rolled] += 1

    if mention_sequence:
        first_rolled = "self" if first_mention["mode"].startswith("self") else first_mention["mode"]
        n_first = mode_counts[first_rolled]
        n_total = sum(mode_counts.values())
        first_mode_share = n_first / n_total if n_total else 0.0
    else:
        first_rolled = None
        first_mode_share = None

    # Also: is Claude's opener prompt always the same, or does it differ between
    # high-self vs high-fleet sessions?
    first_claude = claude_prompts[0]["text"] if claude_prompts else ""

    return {
        "session": int(re.search(r"session_(\d+)", stem).group(1)),
        "n_sage_responses": len(sage_responses),
        "n_claude_prompts": len(claude_prompts),
        "first_claude_prompt": first_claude,
        "claude_prompts_all": claude_prompts,
        "first_mention": first_mention,
        "first_mode_locks": first_mode_share,
        "mention_sequence": mention_sequence,
        "mode_counts_rolled": dict(mode_counts),
    }


def main():
    print("S122b — mode lock-in analysis (Thor)")
    print()

    inst = "thor-qwen3.5-27b"
    sessions = load_sessions(inst, last_n=30)

    results = []
    for stem, sess in sessions:
        r = analyse_session(inst, stem, sess)
        results.append(r)

    # Print compact table.
    print(f"{'sess':>4s}  {'firstT':>6s}  {'firstMode':>10s}  "
          f"{'lockShare':>10s}  {'rolled counts':>20s}  Claude opener")
    for r in results:
        fm = r["first_mention"]
        if fm is None:
            line = f"  {r['session']:>4d}  {'-':>6s}  {'-':>10s}  {'-':>10s}  {'(no mentions)':>20s}  {r['first_claude_prompt'][:60]}"
        else:
            line = (f"  {r['session']:>4d}  {fm['turn']:>6d}  {fm['mode']:>10s}  "
                    f"{r['first_mode_locks']:>9.2f}  "
                    f"{str(r['mode_counts_rolled']):>20s}  {r['first_claude_prompt'][:60]}")
        print(line)
    print()

    # Are claude opener prompts uniform?
    openers = set(r["first_claude_prompt"] for r in results)
    print(f"Distinct Claude openers across {len(results)} sessions: {len(openers)}")
    for o in openers:
        print(f"  | {o[:200]}")
    print()

    # Are subsequent Claude prompts mode-correlated? Sample a fleet-only and
    # a self-only session, and dump every Claude turn.
    by_session = {r["session"]: r for r in results}
    fleet_only = [r for r in results if r["mode_counts_rolled"].get("fleet", 0) >= 5
                  and r["mode_counts_rolled"].get("self", 0) == 0]
    self_only = [r for r in results if r["mode_counts_rolled"].get("self", 0) >= 5
                 and r["mode_counts_rolled"].get("fleet", 0) == 0]

    print("=== FLEET-ONLY session Claude prompts ===")
    for r in fleet_only[:3]:
        print(f"\n-- session {r['session']} (rolled: {r['mode_counts_rolled']}) --")
        for cp in r["claude_prompts_all"]:
            print(f"  T{cp['turn']:>2d}: {cp['text'][:200]}")

    print()
    print("=== SELF-ONLY session Claude prompts ===")
    for r in self_only[:3]:
        print(f"\n-- session {r['session']} (rolled: {r['mode_counts_rolled']}) --")
        for cp in r["claude_prompts_all"]:
            print(f"  T{cp['turn']:>2d}: {cp['text'][:200]}")

    # Save.
    out_path = S122_DIR / "s122b_mode_lock_in.json"
    with open(out_path, "w") as f:
        json.dump({
            "instance": inst,
            "n_distinct_openers": len(openers),
            "openers_sample": list(openers)[:5],
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

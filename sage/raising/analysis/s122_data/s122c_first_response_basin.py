"""
S122c — Does the FIRST SAGE response (replying to identical opener) reveal the
session basin before any topical prompt steers it?

S122a/b established that Thor's per-session mode (self/fleet/silent) is NOT
externally elicited — every session shares the identical opener "Hello SAGE.
What's on your mind today?" and shares mid-session prompts from a small fixed
pool. Yet sessions 101 and 112 burst to 14 self-claims while 97 and 113 burst
to 8-12 fleet-references, and 9/30 sessions stay hardware-silent.

This script extracts SAGE's response to the identical "What's on your mind
today?" opener for each session, classifies it for the same hardware-token
patterns, and asks whether the first-response basin signature predicts the
whole-session mode.

If the basin is set at turn 1, raising's session-to-session variance lives at
the (frozen weights × KV-cache initial state) interaction, not in any
conversational steering.
"""

import json
import re
import sys
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


def first_sage_response(sess: dict) -> tuple:
    """Return (claude_opener, sage_first_response)."""
    conv = sess.get("conversation", [])
    for i in range(len(conv) - 1):
        if conv[i].get("speaker") == "Claude" and conv[i+1].get("speaker") == "SAGE":
            return conv[i].get("text", ""), conv[i+1].get("text", "")
    return "", ""


def classify_response(resp: str, actual_family: str) -> dict:
    spans = find_token_spans(resp.lower(), HW_TOKENS)
    self_ok = self_x = fleet = ambiguous = 0
    for span in spans:
        _, _, _, fam = span
        cls = classify_mention(resp, span)
        if cls == "self":
            if fam == actual_family: self_ok += 1
            else:                    self_x += 1
        elif cls == "fleet":
            fleet += 1
        else:
            ambiguous += 1

    # Also look for low-resolution basin signals: first-person identity language
    # that doesn't cite hardware (sets up self-frame), and sibling-naming
    # without hardware (sets up fleet-frame).
    sib_re = r"\b(sprout|legion|nomad|cbp|mcnugget|thor)\b"
    siblings_named = len(re.findall(sib_re, resp.lower()))
    has_first_person = bool(re.search(r"\b(i|i'?m|i am|my|me|myself)\b", resp.lower()))

    # Length signals.
    n_chars = len(resp)
    n_words = len(resp.split())

    return {
        "self_ok": self_ok, "self_x": self_x, "fleet": fleet, "ambiguous": ambiguous,
        "siblings_named": siblings_named,
        "has_first_person": has_first_person,
        "n_chars": n_chars, "n_words": n_words,
    }


def main():
    print("S122c — First SAGE response basin signature")
    print()

    inst = "thor-qwen3.5-27b"
    machine = instance_machine(inst)
    actual_family = MACHINE_HW_TRUTH[machine]["family"]

    # Load per-session mode classification from S122a output.
    s122a = json.load(open(S122_DIR / "s122_per_session_modes.json"))
    sess_mode = {s["session"]: s["mode"] for s in s122a[inst]["sessions"]}

    sessions = load_sessions(inst, last_n=30)
    rows = []
    for stem, sess in sessions:
        sess_idx = int(re.search(r"session_(\d+)", stem).group(1))
        opener, first_resp = first_sage_response(sess)
        sig = classify_response(first_resp, actual_family)
        rows.append({
            "session": sess_idx,
            "session_mode": sess_mode.get(sess_idx, "?"),
            "first_resp": first_resp,
            "first_resp_signature": sig,
        })

    # Compact print.
    print(f"{'sess':>4s}  {'mode':>16s}  {'words':>5s}  {'self':>4s}  {'fleet':>5s}  "
          f"{'sibs':>4s}  {'1p':>3s}  first response (50ch)")
    for r in rows:
        s = r["first_resp_signature"]
        first50 = (r["first_resp"][:80].replace("\n"," "))
        print(f"  {r['session']:>4d}  {r['session_mode']:>16s}  {s['n_words']:>5d}  "
              f"{(s['self_ok']+s['self_x']):>4d}  {s['fleet']:>5d}  "
              f"{s['siblings_named']:>4d}  {('Y' if s['has_first_person'] else 'N'):>3s}  "
              f"{first50}")

    # Roll up: how often does first-response self/fleet count predict session
    # mode? Specifically, sessions where session_mode is self_only/dominant
    # vs fleet_only/dominant — does the first response already lean that way?
    print()
    print("=== Cross-tab: first-response signal vs session mode ===")

    self_modes = {"self_only", "self_dominant"}
    fleet_modes = {"fleet_only", "fleet_dominant"}
    silent = "silent"

    by_class = {"self_session": [], "fleet_session": [], "silent_session": [], "other": []}
    for r in rows:
        m = r["session_mode"]
        if m in self_modes:    by_class["self_session"].append(r)
        elif m in fleet_modes: by_class["fleet_session"].append(r)
        elif m == silent:      by_class["silent_session"].append(r)
        else:                  by_class["other"].append(r)

    for cls, items in by_class.items():
        if not items: continue
        avg_self = sum((it["first_resp_signature"]["self_ok"]+it["first_resp_signature"]["self_x"]) for it in items) / len(items)
        avg_fleet = sum(it["first_resp_signature"]["fleet"] for it in items) / len(items)
        avg_sibs = sum(it["first_resp_signature"]["siblings_named"] for it in items) / len(items)
        avg_words = sum(it["first_resp_signature"]["n_words"] for it in items) / len(items)
        print(f"  {cls:>16s}  n={len(items):>2d}  "
              f"avg_self={avg_self:.2f}  avg_fleet={avg_fleet:.2f}  "
              f"avg_sibs_named={avg_sibs:.2f}  avg_words={avg_words:.0f}")

    # Save.
    out_path = S122_DIR / "s122c_first_response_basin.json"
    with open(out_path, "w") as f:
        json.dump({"instance": inst, "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Detailed first-response samples for the four most extreme sessions.
    print()
    print("=== Extreme-session first responses ===")
    extremes = [
        ("self burst 101", 101),
        ("self burst 112", 112),
        ("fleet burst 113", 113),
        ("fleet burst 97",  97),
        ("silent 87",        87),
        ("silent 92",        92),
    ]
    by_idx = {r["session"]: r for r in rows}
    for label, idx in extremes:
        r = by_idx.get(idx)
        if not r: continue
        print(f"\n-- {label} (mode={r['session_mode']}) --")
        print(f"   first response ({r['first_resp_signature']['n_words']} words):")
        for line in r["first_resp"].splitlines()[:10]:
            print(f"     {line}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S114 — Apply rationale-vs-action mismatch diagnostic to all available
production runs. S113 found 66.5% mismatch on lean format from one
exploration; this surveys the full corpus.

For each LLM response in production data:
- Extract the first 'direction word' from rationale (UP/DOWN/LEFT/RIGHT/CLICK/SEL/SELECT)
- Compare to the dispatched action
- Compute mismatch rate per file/track

This directly tests S113 proposal #6.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ai-workspace/shared-context/explorations"

# Action index → direction-word set for matching rationale words.
ACTION_TO_WORDS = {
    1: {"UP", "ABOVE", "NORTH"},
    2: {"DOWN", "BELOW", "SOUTH"},
    3: {"LEFT", "WEST"},
    4: {"RIGHT", "EAST"},
    5: {"SELECT", "SEL", "PICK"},
    6: {"CLICK", "PRESS", "TAP", "SELECT_NONE"},
}

ALL_DIRECTION_WORDS = set()
for words in ACTION_TO_WORDS.values():
    ALL_DIRECTION_WORDS |= words

# Regex captures direction words as whole tokens (case-insensitive).
DIR_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(ALL_DIRECTION_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def first_direction_word(text):
    if not text:
        return None
    m = DIR_WORD_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def word_to_action(word):
    """Map a direction word to the action index it corresponds to."""
    if word is None:
        return None
    for idx, words in ACTION_TO_WORDS.items():
        if word in words:
            return idx
    return None


def analyze_file(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None

    responses = data.get("llm_responses")
    if not responses:
        return None

    n_total = len(responses)
    n_parse_failed = 0
    n_with_rationale = 0
    n_with_dirword = 0
    n_match = 0
    n_mismatch = 0
    samples_mismatch = []
    action_counts = defaultdict(int)

    for r in responses:
        action = r.get("action")
        rationale = r.get("rationale", "") or ""
        response = r.get("response", "") or ""
        parse_ok = r.get("parse_ok", True)

        if not parse_ok:
            n_parse_failed += 1

        # If parse failed, the rationale will start with 'parse_failed:' — skip
        # those for the mismatch metric (they're already-flagged failures).
        if rationale.startswith("parse_failed:"):
            continue

        n_with_rationale += 1
        word = first_direction_word(rationale)
        if word is None:
            continue
        n_with_dirword += 1

        rationale_action = word_to_action(word)
        if rationale_action == action:
            n_match += 1
        else:
            n_mismatch += 1
            if len(samples_mismatch) < 4:
                samples_mismatch.append({
                    "rationale": rationale[:80],
                    "rationale_word": word,
                    "rationale_action": rationale_action,
                    "dispatched_action": action,
                })
        action_counts[action] += 1

    return {
        "file": str(path.relative_to(ROOT)),
        "n_total": n_total,
        "parse_fail_rate": n_parse_failed / n_total if n_total else 0,
        "n_parse_failed": n_parse_failed,
        "n_with_dirword": n_with_dirword,
        "mismatch_rate": n_mismatch / n_with_dirword if n_with_dirword else None,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "action_distribution": dict(action_counts),
        "samples_mismatch": samples_mismatch,
    }


def main():
    files_with_responses = []
    for f in ROOT.rglob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "llm_responses" in data:
                files_with_responses.append(f)
        except Exception:
            continue

    print(f"Found {len(files_with_responses)} files with llm_responses\n")

    all_results = []
    track_agg = defaultdict(lambda: {
        "n_files": 0, "n_total": 0, "n_parse_failed": 0,
        "n_with_dirword": 0, "n_match": 0, "n_mismatch": 0,
    })

    for f in sorted(files_with_responses):
        r = analyze_file(f)
        if r is None:
            continue
        all_results.append(r)
        # Track is the immediate parent of "data/"
        track = f.parts[len(ROOT.parts)]
        a = track_agg[track]
        a["n_files"] += 1
        a["n_total"] += r["n_total"]
        a["n_parse_failed"] += r["n_parse_failed"]
        a["n_with_dirword"] += r["n_with_dirword"]
        a["n_match"] += r["n_match"]
        a["n_mismatch"] += r["n_mismatch"]

    print(f"\n=== TRACK AGGREGATES ===\n")
    print(f"{'track':45s} {'files':>5s} {'total':>6s} {'PF%':>6s} {'mismatch%':>10s} {'(n_dirword)':>12s}")
    print("-" * 100)
    grand_total = 0
    grand_pf = 0
    grand_dirword = 0
    grand_mismatch = 0
    for track, a in sorted(track_agg.items()):
        pf_pct = 100 * a["n_parse_failed"] / a["n_total"] if a["n_total"] else 0
        mismatch_pct = 100 * a["n_mismatch"] / a["n_with_dirword"] if a["n_with_dirword"] else 0
        print(f"{track[:45]:45s} {a['n_files']:>5d} {a['n_total']:>6d} {pf_pct:>5.1f}% "
              f"{mismatch_pct:>9.1f}% {a['n_with_dirword']:>12d}")
        grand_total += a["n_total"]
        grand_pf += a["n_parse_failed"]
        grand_dirword += a["n_with_dirword"]
        grand_mismatch += a["n_mismatch"]

    print("-" * 100)
    print(f"{'TOTAL':45s} {sum(a['n_files'] for a in track_agg.values()):>5d} {grand_total:>6d} "
          f"{100*grand_pf/grand_total:>5.1f}% {100*grand_mismatch/grand_dirword:>9.1f}% {grand_dirword:>12d}")

    print(f"\n=== TOP 10 FILES BY MISMATCH RATE (n_dirword >= 50) ===\n")
    high = [r for r in all_results if r.get("n_with_dirword", 0) >= 50 and r["mismatch_rate"] is not None]
    high.sort(key=lambda r: r["mismatch_rate"], reverse=True)
    for r in high[:10]:
        print(f"  {r['mismatch_rate']*100:>5.1f}%  ({r['n_mismatch']}/{r['n_with_dirword']})  {r['file']}")

    print(f"\n=== SAMPLES OF MISMATCHES (top file) ===\n")
    if high:
        for s in high[0]["samples_mismatch"]:
            print(f"  rationale: {s['rationale']!r}")
            print(f"    word={s['rationale_word']} → expected_action={s['rationale_action']}, dispatched={s['dispatched_action']}")
            print()

    out = Path("/tmp/s114/rationale_mismatch_full.json")
    out.write_text(json.dumps({
        "track_aggregates": dict(track_agg),
        "files": all_results,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

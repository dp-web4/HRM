"""Cross-capacity validation of the prev-summary schema-fragment filter.

S92 shipped `sage/raising/prev_summary_filter.py` calibrated on Sprout 0.5B
sessions 62-121 (11/11 bursts caught, 0/93 false positives). The filter rule
`(qmarks >= 5) OR (schema_phrases >= 1)` was tuned against a dataset where
the LoRA cycle_001 basin produces specific "what's the next X?" self-
interrogation templates.

**Open question (carried from S90, S91, S92):** does the rule generalize to
higher capacity? Two distinct worries —
  (a) false-positives: 4B/12B reflective memory-asks may legitimately
      contain multiple question marks or a phrase like "what's the next step"
      as part of thoughtful articulation. If so, the filter would suppress
      healthy prior-session continuity on non-LoRA runs.
  (b) false-negatives-at-capacity: a basin at 4B or 12B might exist with
      different surface forms. If so, the filter protects 0.5B but not the
      larger instances, and wiring it into every runner would create a
      false sense of coverage.

Both matter for Phase 2. If (a) is real the filter needs capacity-aware
thresholds or a separate detector. If (b) is real, 4B/12B still need a
distinct filter before Phase 2 can claim to close the surface.

This script runs the filter against every session JSON in the fleet-
relevant instance directories, extracting the same memory-ask that
`_get_previous_session_summary` extracts (last SAGE turn following a user
turn containing "remember"), and reports counts + samples per capacity.

Run from repo root:
    python3 -m sage.raising.analysis.cross_capacity_filter_scan
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from sage.raising.prev_summary_filter import is_schema_fragment

INSTANCES = [
    ("sprout-qwen2.5-0.5b", "0.5B"),
    ("sprout-qwen3.5-0.8b", "0.8B"),
    ("sprout-qwen3.5-2b", "2B"),
    ("nomad-gemma3-4b", "4B"),
    ("legion-gemma3-12b", "12B-legion"),
    ("mcnugget-gemma3-12b", "12B-mcnugget"),
    ("thor-qwen2.5-14b", "14B"),
]

SESS_RE = re.compile(r"session_(\d+)(?:[_.]|$)")


def extract_memory_ask(session_data: dict) -> str | None:
    """Return last SAGE turn following a user turn containing 'remember'."""
    conv = session_data.get("conversation") or session_data.get("turns") or []
    for i in range(len(conv) - 1, -1, -1):
        turn = conv[i]
        if turn.get("speaker") != "SAGE":
            continue
        prev = conv[i - 1] if i > 0 else {}
        prev_text = (prev.get("text") or "").lower()
        if "remember" in prev_text:
            return turn.get("text", "")
    return None


def simulate_prev_summary(prev_session_data: dict) -> tuple[str, str]:
    """Replicate `_get_previous_session_summary`'s extraction logic.

    Returns ("remember_fired", response) when the function would splice a
    prior SAGE response verbatim into the next system prompt, or
    ("generic_fallback", "") when it falls through to the phase-only
    string.
    """
    conv = prev_session_data.get("conversation", [])
    for i in range(len(conv) - 1, -1, -1):
        if conv[i].get("speaker") != "SAGE":
            continue
        response = conv[i].get("text", "")
        if i > 0 and "remember" in conv[i - 1].get("text", "").lower():
            return "remember_fired", response
    return "generic_fallback", ""


def scan_instance(inst_dir: Path) -> dict:
    """Collect filter statistics and samples for one instance.

    Two views:
      - memory-ask view: filter applied to every SAGE-after-remember turn
        found anywhere in a session (shows the rule's prevalence across
        all outputs that could be extracted by the current function).
      - prev-summary-sim view: replay _get_previous_session_summary on
        each session using the previous session's JSON as the source.
        This counts the specific spliced strings the runners would inject.
    """
    stats = {
        "n_sessions": 0,
        "n_with_memory_ask": 0,
        "n_flagged": 0,
        "n_clean": 0,
        "flagged_samples": [],
        "clean_samples": [],
        "qmark_counts": [],
        "sim_remember_fired": 0,
        "sim_generic_fallback": 0,
        "sim_flagged_if_fired": 0,
        "sim_flagged_samples": [],
    }
    sess_map: dict[int, dict] = {}
    for sf in sorted(inst_dir.glob("session_*.json")):
        m = SESS_RE.search(sf.name)
        if not m:
            continue
        n = int(m.group(1))
        try:
            d = json.loads(sf.read_text())
        except Exception:
            continue
        sess_map[n] = d
        stats["n_sessions"] += 1
        ma = extract_memory_ask(d)
        if not ma:
            continue
        stats["n_with_memory_ask"] += 1
        qmarks = ma.count("?")
        stats["qmark_counts"].append(qmarks)
        flagged = is_schema_fragment(ma)
        if flagged:
            stats["n_flagged"] += 1
            if len(stats["flagged_samples"]) < 8:
                stats["flagged_samples"].append(
                    (n, qmarks, ma[:200].replace("\n", " "))
                )
        else:
            stats["n_clean"] += 1
            if len(stats["clean_samples"]) < 3:
                stats["clean_samples"].append(
                    (n, qmarks, ma[:200].replace("\n", " "))
                )

    # Prev-summary simulation: for each session N, run logic against N-1.
    for n in sorted(sess_map):
        if (n - 1) not in sess_map:
            continue
        kind, resp = simulate_prev_summary(sess_map[n - 1])
        if kind == "remember_fired":
            stats["sim_remember_fired"] += 1
            if is_schema_fragment(resp):
                stats["sim_flagged_if_fired"] += 1
                if len(stats["sim_flagged_samples"]) < 5:
                    stats["sim_flagged_samples"].append(
                        (n, resp[:200].replace("\n", " "))
                    )
        else:
            stats["sim_generic_fallback"] += 1
    return stats


def main() -> None:
    instances_root = Path.home() / "ai-workspace/SAGE/sage/instances"
    all_results: dict[str, dict] = {}

    for inst_name, label in INSTANCES:
        inst_dir = instances_root / inst_name / "sessions"
        if not inst_dir.exists():
            print(f"[{label}] {inst_name}: no sessions dir — skipped")
            continue
        stats = scan_instance(inst_dir)
        all_results[inst_name] = {"label": label, **stats}

    # Summary table — memory-ask view
    print()
    print("== MEMORY-ASK VIEW (every SAGE turn after a 'remember' user turn) ==")
    print(f"{'Instance':<28} {'Label':<14} {'N':>5} {'MA':>5} {'Flag':>5} "
          f"{'Clean':>6} {'Flag%':>7} {'avgQ':>6} {'maxQ':>5}")
    print("-" * 90)
    for inst, r in all_results.items():
        qs = r["qmark_counts"] or [0]
        n_ma = r["n_with_memory_ask"]
        pct = (100 * r["n_flagged"] / n_ma) if n_ma else 0
        print(
            f"{inst:<28} {r['label']:<14} {r['n_sessions']:>5} "
            f"{n_ma:>5} {r['n_flagged']:>5} {r['n_clean']:>6} "
            f"{pct:>6.1f}% {sum(qs)/len(qs):>6.2f} {max(qs):>5}"
        )

    # Summary table — prev-summary simulation view
    print()
    print("== PREV-SUMMARY SIM VIEW (what the runner would splice into next "
          "session's system prompt) ==")
    print(f"{'Instance':<28} {'Label':<14} {'Seq':>5} {'Fire':>5} "
          f"{'Generic':>8} {'Flag':>5} {'FireFlag%':>10}")
    print("-" * 85)
    for inst, r in all_results.items():
        fired = r["sim_remember_fired"]
        flagged = r["sim_flagged_if_fired"]
        seq = r["n_sessions"]
        pct = (100 * flagged / fired) if fired else 0
        print(
            f"{inst:<28} {r['label']:<14} {seq:>5} {fired:>5} "
            f"{r['sim_generic_fallback']:>8} {flagged:>5} {pct:>9.1f}%"
        )

    # Detail: flagged samples per instance
    for inst, r in all_results.items():
        if not r["flagged_samples"]:
            continue
        print()
        print(f"=== FLAGGED at {r['label']} ({inst}) — {r['n_flagged']} total ===")
        for n, q, text in r["flagged_samples"]:
            print(f"  S{n} q={q}: {text[:160]}")

    # Detail: one clean sample per capacity-tier for ground-truth sanity
    print()
    print("=== CLEAN MEMORY-ASK SAMPLES (one per instance) ===")
    for inst, r in all_results.items():
        if r["clean_samples"]:
            n, q, text = r["clean_samples"][0]
            print(f"  [{r['label']}] S{n} q={q}: {text[:160]}")

    # Write machine-readable results
    out = Path.home() / (
        "ai-workspace/SAGE/sage/raising/analysis/"
        "cross_capacity_filter_scan_results.json"
    )
    summary = {
        inst: {
            "label": r["label"],
            "n_sessions": r["n_sessions"],
            "n_with_memory_ask": r["n_with_memory_ask"],
            "n_flagged": r["n_flagged"],
            "n_clean": r["n_clean"],
            "avg_qmarks": (
                sum(r["qmark_counts"]) / len(r["qmark_counts"])
                if r["qmark_counts"] else 0
            ),
            "max_qmarks": max(r["qmark_counts"]) if r["qmark_counts"] else 0,
            "flagged_samples": [
                {"session": n, "qmarks": q, "text": t}
                for (n, q, t) in r["flagged_samples"]
            ],
            "sim_remember_fired": r["sim_remember_fired"],
            "sim_generic_fallback": r["sim_generic_fallback"],
            "sim_flagged_if_fired": r["sim_flagged_if_fired"],
            "sim_flagged_samples": [
                {"into_session": n, "text": t}
                for (n, t) in r["sim_flagged_samples"]
            ],
        }
        for inst, r in all_results.items()
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()

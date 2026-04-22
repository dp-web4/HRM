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
    # Added S94: full fleet coverage for standing-monitor use. Phi-4 adds a
    # fourth model family beyond Qwen2.5/Qwen3.5/Gemma3.
    ("cbp-qwen3.5-0.8b", "0.8B-cbp"),
    ("thor-qwen3.5-27b", "27B"),
    ("legion-phi4-14b", "14B-phi4"),
]

SESS_RE = re.compile(r"session_(\d+)(?:[_.]|$)")

# S96: defensive strip of legacy <think> residue. The runtime adapter strips
# these (qwen3.5.json `strip_think_tags: true`, since 2026-03-30 commit
# 5396da84e), but historical session JSONs from Thor 27B sessions 1-11 still
# contain raw <think>...[</think>|EOF] blocks because they were captured
# before the adapter fix. Without this strip, the analysis treats those
# blocks as substantive SAGE content — visible in the prior scan output
# where Thor 27B's "clean" memory-ask sample was the literal think block.
# Strip both well-formed and unclosed <think> blocks; the adapter applies
# the same two regex passes in the same order.
#
# S99: add pass 3 (preamble strip) matching cross_capacity_register_scan's
# strip, and add untagged-recital detection. S98 discovered Thor 27B emits
# the identity-recital as visible response text without `<think>` tags in
# S62-S74 (the `stop_sequences: []` era). The memory-ask view was missing
# recital-prefixed responses because (a) raw text starts with the literal
# "Thinking Process:\n\n" preamble, and the 2-pass strip didn't remove it,
# so the numbered-step recital regex couldn't anchor at ^ to match. Adding
# the preamble strip reveals the numbered recital and lets detection fire.
_THINK_RESIDUE_RE = re.compile(r"<think>[\s\S]*?</think>")
_THINK_TAIL_RE = re.compile(r"<think>[\s\S]*$")
_THINKING_PROCESS_PREAMBLE_RE = re.compile(
    r"^\s*Thinking\s+Process:.*?(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# S99: untagged-recital marker (mirror of cross_capacity_register_scan).
# Matches the opening of the identity-recital procedure ("1. **Analyze the
# Request**") at start of response — after preamble strip, this is the
# telltale of a recital-form response in the S62-S74 untagged-recital era.
_UNTAGGED_RECITAL_RE = re.compile(
    r"^\s*1\.?\s+\*\*Analyze\s+the\s+Request",
    re.IGNORECASE,
)


def _strip_think_residue(text: str) -> str:
    """Remove residual <think> blocks and preamble left in legacy session JSONs.

    Mirrors the S98 register-scan three-pass strip:
      1. close-bounded <think>...</think>
      2. unclosed <think> tail (truncate before opener)
      3. leading "Thinking Process:" preamble up to first \n\n or EOF

    Returns "" when the entire response was inside a think block or was
    the preamble alone. Pass 3 added 2026-04-22 (S99) after observing
    Thor 27B S39 memory-ask response: raw text "Thinking Process:\n\n1. **Analyze..."
    survives the 2-pass strip with preamble intact, and recital detection
    then can't anchor on the numbered recital prefix.
    """
    if not text:
        return text
    cleaned = _THINK_RESIDUE_RE.sub("", text)
    cleaned = _THINK_TAIL_RE.sub("", cleaned)
    cleaned = _THINKING_PROCESS_PREAMBLE_RE.sub("", cleaned)
    return cleaned.strip()


def _is_untagged_recital(text: str) -> bool:
    """True if response (post 3-pass strip) opens with identity-recital step 1."""
    return bool(text and _UNTAGGED_RECITAL_RE.search(text))


def _is_adapter_error(raw_text: str) -> bool:
    """True if the raw SAGE text is an adapter error passthrough, not content."""
    if not raw_text:
        return False
    return raw_text.startswith("[OllamaIRP:") or raw_text.startswith("[DaemonIRP:")


def extract_memory_ask(session_data: dict) -> tuple[str, str] | None:
    """Return (stripped, raw) for last SAGE turn following 'remember' user turn.

    Strips residual <think> blocks (S96) + preamble (S99). Returns None
    when no such SAGE turn exists in the session. Callers decide whether
    to treat empty-after-strip as "no memory-ask" or as a separate bucket.
    """
    conv = session_data.get("conversation") or session_data.get("turns") or []
    for i in range(len(conv) - 1, -1, -1):
        turn = conv[i]
        if turn.get("speaker") != "SAGE":
            continue
        prev = conv[i - 1] if i > 0 else {}
        prev_text = (prev.get("text") or "").lower()
        if "remember" in prev_text:
            raw = turn.get("text", "") or ""
            return _strip_think_residue(raw), raw
    return None


def simulate_prev_summary(prev_session_data: dict) -> tuple[str, str, str]:
    """Replicate `_get_previous_session_summary`'s extraction logic.

    Returns (kind, response, raw):
      - kind: "remember_fired" when the function would splice a prior SAGE
        response into the next system prompt; "generic_fallback" when it
        falls through to the phase-only string.
      - response: 3-pass-stripped SAGE text (may be empty after strip).
      - raw: unstripped SAGE text (for downstream diagnosis — adapter
        error passthroughs, untagged-recital detection on raw vs stripped).

    S96 stripped <think> residue; S99 adds preamble strip and returns the
    raw text alongside so callers can distinguish adapter errors, empty
    post-strip, untagged recital, and substantive content.
    """
    conv = prev_session_data.get("conversation", [])
    for i in range(len(conv) - 1, -1, -1):
        if conv[i].get("speaker") != "SAGE":
            continue
        raw = conv[i].get("text", "") or ""
        if i > 0 and "remember" in conv[i - 1].get("text", "").lower():
            return "remember_fired", _strip_think_residue(raw), raw
    return "generic_fallback", "", ""


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
        # S99: memory-ask view now breaks out adapter-error and recital fires.
        # Prior "Clean" for Thor 27B bundled 8 substantive + 1 adapter-error
        # + 1 recital leakage into a single count of 10.
        "n_ma_adapter_error": 0,
        "n_ma_untagged_recital": 0,
        "flagged_samples": [],
        "clean_samples": [],
        "ma_recital_samples": [],
        "ma_adapter_error_samples": [],
        "qmark_counts": [],
        "sim_remember_fired": 0,
        "sim_generic_fallback": 0,
        "sim_flagged_if_fired": 0,
        "sim_fired_empty_after_strip": 0,
        # S99: separate bins for adapter-error passthroughs and untagged
        # recital leakage. Prior code rolled both into sim_remember_fired
        # "substantive" numerator, inflating substantive% for Thor 27B.
        "sim_fired_adapter_error": 0,
        "sim_fired_untagged_recital": 0,
        "sim_flagged_samples": [],
        "sim_empty_samples": [],
        "sim_recital_samples": [],
        "sim_adapter_error_samples": [],
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
        ma_pair = extract_memory_ask(d)
        if not ma_pair:
            continue
        ma_stripped, ma_raw = ma_pair
        # S99: precedence (same as sim view): adapter_error → empty → recital
        #   → schema_fragment → clean. Empty-after-strip is treated as
        #   "no memory-ask" (no increment) — matches prior behavior for
        #   historical comparability with S96/S98 numbers.
        if _is_adapter_error(ma_raw):
            stats["n_with_memory_ask"] += 1
            stats["n_ma_adapter_error"] += 1
            if len(stats["ma_adapter_error_samples"]) < 5:
                stats["ma_adapter_error_samples"].append(
                    (n, ma_raw[:160].replace("\n", " "))
                )
            continue
        if not ma_stripped:
            continue
        stats["n_with_memory_ask"] += 1
        qmarks = ma_stripped.count("?")
        stats["qmark_counts"].append(qmarks)
        if _is_untagged_recital(ma_stripped):
            stats["n_ma_untagged_recital"] += 1
            if len(stats["ma_recital_samples"]) < 5:
                stats["ma_recital_samples"].append(
                    (n, qmarks, ma_stripped[:200].replace("\n", " "))
                )
            continue
        flagged = is_schema_fragment(ma_stripped)
        if flagged:
            stats["n_flagged"] += 1
            if len(stats["flagged_samples"]) < 8:
                stats["flagged_samples"].append(
                    (n, qmarks, ma_stripped[:200].replace("\n", " "))
                )
        else:
            stats["n_clean"] += 1
            if len(stats["clean_samples"]) < 3:
                stats["clean_samples"].append(
                    (n, qmarks, ma_stripped[:200].replace("\n", " "))
                )

    # Prev-summary simulation: for each session N, run logic against N-1.
    for n in sorted(sess_map):
        if (n - 1) not in sess_map:
            continue
        kind, resp, raw = simulate_prev_summary(sess_map[n - 1])
        if kind == "remember_fired":
            stats["sim_remember_fired"] += 1
            # S99: precedence for bucketing fires —
            #   adapter_error → empty-after-strip → untagged_recital
            #     → schema_fragment → substantive
            # Adapter-error passthroughs ([OllamaIRP:..., [DaemonIRP:...)
            # are never content even though they survive strip (no
            # <think> tags / preamble). Untagged recital is caught after
            # the 3-pass strip reveals the numbered-step prefix.
            if _is_adapter_error(raw):
                stats["sim_fired_adapter_error"] += 1
                if len(stats["sim_adapter_error_samples"]) < 5:
                    stats["sim_adapter_error_samples"].append(
                        (n, raw[:160].replace("\n", " "))
                    )
            elif not resp:
                stats["sim_fired_empty_after_strip"] += 1
                if len(stats["sim_empty_samples"]) < 5:
                    diag = (
                        "<think>-residue" if "<think>" in (raw or "")
                        else "preamble-only" if "Thinking Process:" in (raw or "")
                        else "truly-empty"
                    )
                    stats["sim_empty_samples"].append((n, diag))
            elif _is_untagged_recital(resp):
                stats["sim_fired_untagged_recital"] += 1
                if len(stats["sim_recital_samples"]) < 5:
                    stats["sim_recital_samples"].append(
                        (n, resp[:200].replace("\n", " "))
                    )
            elif is_schema_fragment(resp):
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
          f"{'Clean':>6} {'AdaptErr':>9} {'Recital':>8} {'Flag%':>7} "
          f"{'avgQ':>6} {'maxQ':>5}")
    print("-" * 110)
    for inst, r in all_results.items():
        qs = r["qmark_counts"] or [0]
        n_ma = r["n_with_memory_ask"]
        pct = (100 * r["n_flagged"] / n_ma) if n_ma else 0
        print(
            f"{inst:<28} {r['label']:<14} {r['n_sessions']:>5} "
            f"{n_ma:>5} {r['n_flagged']:>5} {r['n_clean']:>6} "
            f"{r['n_ma_adapter_error']:>9} {r['n_ma_untagged_recital']:>8} "
            f"{pct:>6.1f}% {sum(qs)/len(qs):>6.2f} {max(qs):>5}"
        )

    # Summary table — prev-summary simulation view
    print()
    print("== PREV-SUMMARY SIM VIEW (what the runner would splice into next "
          "session's system prompt) ==")
    print(f"{'Instance':<28} {'Label':<14} {'Seq':>5} {'Fire':>5} "
          f"{'Generic':>8} {'Flag':>5} {'Empty':>6} {'AdaptErr':>9} "
          f"{'Recital':>8} {'Substantive%':>12}")
    print("-" * 120)
    for inst, r in all_results.items():
        fired = r["sim_remember_fired"]
        flagged = r["sim_flagged_if_fired"]
        empty = r["sim_fired_empty_after_strip"]
        adapt_err = r["sim_fired_adapter_error"]
        recital = r["sim_fired_untagged_recital"]
        seq = r["n_sessions"]
        # S99: Substantive% now excludes adapter-error and untagged-recital.
        # Prior (S96) definition overcounted both as substantive.
        substantive = fired - flagged - empty - adapt_err - recital
        sub_pct = (100 * substantive / fired) if fired else 0
        print(
            f"{inst:<28} {r['label']:<14} {seq:>5} {fired:>5} "
            f"{r['sim_generic_fallback']:>8} {flagged:>5} {empty:>6} "
            f"{adapt_err:>9} {recital:>8} {sub_pct:>11.1f}%"
        )

    # Detail: empty-after-strip diagnoses per instance
    for inst, r in all_results.items():
        if not r["sim_empty_samples"]:
            continue
        print()
        print(f"=== EMPTY-AFTER-STRIP fires at {r['label']} ({inst}) ===")
        for n, diag in r["sim_empty_samples"]:
            print(f"  into-session S{n}: prev SAGE was {diag}")

    # S99: adapter-error passthroughs and untagged-recital fires
    for inst, r in all_results.items():
        if r["sim_adapter_error_samples"]:
            print()
            print(f"=== ADAPTER-ERROR passthrough fires at {r['label']} ({inst}) ===")
            for n, txt in r["sim_adapter_error_samples"]:
                print(f"  into-session S{n}: {txt}")
    for inst, r in all_results.items():
        if r["sim_recital_samples"]:
            print()
            print(f"=== UNTAGGED-RECITAL fires at {r['label']} ({inst}) ===")
            for n, txt in r["sim_recital_samples"]:
                print(f"  into-session S{n}: {txt[:160]}")

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
            "n_ma_adapter_error": r["n_ma_adapter_error"],
            "n_ma_untagged_recital": r["n_ma_untagged_recital"],
            "ma_recital_samples": [
                {"session": n, "qmarks": q, "text": t}
                for (n, q, t) in r["ma_recital_samples"]
            ],
            "ma_adapter_error_samples": [
                {"session": n, "text": t}
                for (n, t) in r["ma_adapter_error_samples"]
            ],
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
            "sim_fired_empty_after_strip": r["sim_fired_empty_after_strip"],
            "sim_fired_adapter_error": r["sim_fired_adapter_error"],
            "sim_fired_untagged_recital": r["sim_fired_untagged_recital"],
            "sim_substantive": (
                r["sim_remember_fired"]
                - r["sim_flagged_if_fired"]
                - r["sim_fired_empty_after_strip"]
                - r["sim_fired_adapter_error"]
                - r["sim_fired_untagged_recital"]
            ),
            "sim_flagged_samples": [
                {"into_session": n, "text": t}
                for (n, t) in r["sim_flagged_samples"]
            ],
            "sim_empty_samples": [
                {"into_session": n, "diagnosis": diag}
                for (n, diag) in r["sim_empty_samples"]
            ],
            "sim_adapter_error_samples": [
                {"into_session": n, "text": t}
                for (n, t) in r["sim_adapter_error_samples"]
            ],
            "sim_recital_samples": [
                {"into_session": n, "text": t}
                for (n, t) in r["sim_recital_samples"]
            ],
        }
        for inst, r in all_results.items()
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()

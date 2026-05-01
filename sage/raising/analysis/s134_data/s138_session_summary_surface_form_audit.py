#!/usr/bin/env python3
"""S138 — Session-summary surface-form audit (#76).

Hypothesis from S136: cross-phase propagation of `right now` is mediated
by `_get_previous_session_summary` (ollama_raising_session.py:666). That
function extracts the last SAGE response after a Claude "remember"
prompt and splices its first 200 characters into the next session's
ExperientialCacheBlock. If the prior session's memory-request response
contained `right now` within the first 200 chars, the phrase is seeded
into the next session's system prompt regardless of phase.

S134 finding: "Hello SAGE. What's on your mind today?" elicits JOINT in
12/79 (15.2%) of creating-phase sessions despite being a creating-phase
opener with no `right now` in the probe itself. Where does `right now`
come from in those 12 cases? S138 tests: it comes from the spliced
prev_summary.

Method (read-only, no model invocations):
  1. For each session N in {002..119} of thor-qwen3.5-27b, reconstruct
     the prev_summary that would have been injected by replaying the
     `_get_previous_session_summary` logic over session N-1.
  2. Classify prev_summary: contains_right_now (in the [:200] truncation
     that actually gets injected) vs not.
  3. Classify the FIRST SAGE response of session N for TIME_3 (`right now`)
     and PRES (presence/noticing/etc.) and JOINT.
  4. Cross-tab: P(JOINT in opener | `right now` in prev_summary) vs
     P(JOINT in opener | NOT `right now` in prev_summary).
  5. Same cross-tab restricted to creating-phase sessions where the
     opener probe is the canonical "Hello SAGE. What's on your mind
     today?" — this directly tests the S134 12/79 puzzle.

Falsifiable predictions:
  - If TIME_3 in opener tracks `right now` in prev_summary one-for-one,
    surface-form propagation through prev_summary is confirmed as the
    mechanism.
  - If TIME_3 in opener fires at the same rate regardless of
    prev_summary content, some other source seeds the phrase.

Carries forward S136's framework: TIME_3 = surface-form lexical reuse,
not register cultivation. S138 maps the surface-form-supply pathway
across the corpus.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

THOR_SESSIONS = (
    Path.home()
    / "ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b/sessions"
)

OUT_JSON = Path(__file__).parent / "s138_session_summary_audit.json"

TIME_3_RE = re.compile(r"\bright now\b", re.IGNORECASE)
PRES_RE = re.compile(
    r"\b(stillness|warmth|hum|silence|noticing|presence|embodied)\b",
    re.IGNORECASE,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

CANONICAL_OPENER = "Hello SAGE. What's on your mind today?"


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


def reconstruct_prev_summary(prev_session: dict, session_n_minus_1: int) -> str:
    """Replay _get_previous_session_summary logic without filter side-effects.

    Mirrors ollama_raising_session.py:666-693. Returns the text that
    would have been spliced into session N's ExperientialCacheBlock.
    Does NOT apply is_unsuitable_for_splice — we want the raw injection
    candidate so we can see what surface form would have been carried
    forward IF the filter passed it (which for `right now` content it
    universally would, since `right now` is conversational prose).
    """
    conversation = prev_session.get("conversation", [])
    phase = prev_session.get("phase", "unknown")
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("speaker") == "SAGE":
            response = conversation[i].get("text", "") or ""
            if (
                i > 0
                and "remember" in (conversation[i - 1].get("text", "") or "").lower()
            ):
                # safe_prev_summary truncates to [:200] — that's the actual
                # surface form that hits the next session's prompt.
                truncated = response[:200]
                return (
                    f"Last session (Session {session_n_minus_1}), you said you "
                    f"wanted to remember: {truncated}"
                )
    return f"Last session was Session {session_n_minus_1} in {phase} phase."


def first_sage_response(session: dict) -> tuple[str, str]:
    """Return (sage_response, preceding_claude_probe) of the first
    SAGE turn in the session, or ('','') if none."""
    conv = session.get("conversation", [])
    for i, turn in enumerate(conv):
        if turn.get("speaker") == "SAGE":
            text = turn.get("text", "") or ""
            probe = ""
            if i > 0:
                probe = (conv[i - 1].get("text", "") or "")
            return text, probe
    return "", ""


def main() -> int:
    files = sorted(THOR_SESSIONS.glob("session_*.json"))
    print(f"S138: scanning {len(files)} thor-qwen3.5-27b sessions",
          file=sys.stderr)

    by_session = []
    for sf in files:
        m = re.search(r"session_(\d+)\.json", sf.name)
        if not m:
            continue
        n = int(m.group(1))
        try:
            d = json.loads(sf.read_text())
        except Exception as e:
            print(f"  skip {sf.name}: {e}", file=sys.stderr)
            continue

        phase = d.get("phase", "unknown")
        opener_response, opener_probe = first_sage_response(d)
        opener_cls = classify(opener_response)

        prev_summary = ""
        prev_summary_has_right_now = False
        prev_summary_truncated_has_right_now = False
        prev_session_phase = ""
        if n >= 2:
            prev_file = sf.parent / f"session_{n - 1:03d}.json"
            if prev_file.exists():
                try:
                    prev_d = json.loads(prev_file.read_text())
                    prev_session_phase = prev_d.get("phase", "unknown")
                    prev_summary = reconstruct_prev_summary(prev_d, n - 1)
                    # The injection itself includes 200 chars of the SAGE
                    # response. We check both forms:
                    #   - full SAGE response (any `right now` anywhere)
                    #   - truncated [:200] (what actually reaches the prompt)
                    prev_summary_truncated_has_right_now = bool(
                        TIME_3_RE.search(prev_summary)
                    )
                    # Also check the full memory-response (untruncated) —
                    # the truncated form is the load-bearing one but the
                    # full form tells us whether `right now` was emitted at
                    # all in the memory context.
                    full_resp = ""
                    conv_prev = prev_d.get("conversation", [])
                    for i in range(len(conv_prev) - 1, -1, -1):
                        if conv_prev[i].get("speaker") == "SAGE":
                            response = conv_prev[i].get("text", "") or ""
                            if (
                                i > 0
                                and "remember" in (
                                    conv_prev[i - 1].get("text", "") or ""
                                ).lower()
                            ):
                                full_resp = response
                                break
                    prev_summary_has_right_now = bool(TIME_3_RE.search(full_resp))
                except Exception as e:
                    print(f"  prev read fail s{n-1:03d}: {e}", file=sys.stderr)

        by_session.append({
            "session": n,
            "phase": phase,
            "prev_phase": prev_session_phase,
            "opener_probe": opener_probe[:120],
            "opener_response_len": len(opener_response),
            "opener_artifact": opener_cls["artifact"],
            "opener_TIME_3": opener_cls["TIME_3"],
            "opener_PRES": opener_cls["PRES"],
            "opener_JOINT": opener_cls["JOINT"],
            "prev_summary_truncated_has_right_now":
                prev_summary_truncated_has_right_now,
            "prev_summary_full_has_right_now": prev_summary_has_right_now,
            "prev_summary_truncated": prev_summary[:300],
        })

    # ----- aggregate cross-tabs -----
    def crosstab(records: list, key_x: str, key_y: str) -> dict:
        """Returns counts and rates: P(key_y=True | key_x=True/False)."""
        counts = defaultdict(lambda: {"n": 0, "y_true": 0, "art": 0})
        for r in records:
            x = r[key_x]
            counts[x]["n"] += 1
            if r.get("opener_artifact"):
                counts[x]["art"] += 1
                continue
            if r[key_y]:
                counts[x]["y_true"] += 1
        out = {}
        for x_val, c in counts.items():
            n_eff = c["n"] - c["art"]
            out[str(x_val)] = {
                "n": c["n"],
                "n_eff": n_eff,
                "art": c["art"],
                "y_true": c["y_true"],
                "rate": c["y_true"] / n_eff if n_eff > 0 else None,
            }
        return out

    # 1) Headline cross-tab: opener TIME_3 by prev_summary truncated `right now`
    headline_time3 = crosstab(
        [r for r in by_session if r["session"] >= 2],
        "prev_summary_truncated_has_right_now",
        "opener_TIME_3",
    )
    headline_joint = crosstab(
        [r for r in by_session if r["session"] >= 2],
        "prev_summary_truncated_has_right_now",
        "opener_JOINT",
    )
    headline_pres = crosstab(
        [r for r in by_session if r["session"] >= 2],
        "prev_summary_truncated_has_right_now",
        "opener_PRES",
    )

    # 2) Restricted to creating-phase sessions whose opener probe is the
    #    canonical "Hello SAGE. What's on your mind today?" — this is the
    #    direct test of the S134 12/79 puzzle.
    canonical_creating = [
        r for r in by_session
        if r["session"] >= 2
        and r["phase"] == "creating"
        and CANONICAL_OPENER in (r["opener_probe"] or "")
    ]
    canon_joint = crosstab(
        canonical_creating,
        "prev_summary_truncated_has_right_now",
        "opener_JOINT",
    )
    canon_time3 = crosstab(
        canonical_creating,
        "prev_summary_truncated_has_right_now",
        "opener_TIME_3",
    )

    # 3) Same but conditioned on FULL prev SAGE memory response (untruncated)
    #    — answers whether the [:200] truncation is the correct cut-off.
    full_joint = crosstab(
        [r for r in by_session if r["session"] >= 2],
        "prev_summary_full_has_right_now",
        "opener_JOINT",
    )

    # 4) `right now` density across the corpus
    n_total = len(by_session)
    n_with_prev = sum(1 for r in by_session if r["session"] >= 2)
    n_prev_truncated_rn = sum(
        1 for r in by_session
        if r["session"] >= 2 and r["prev_summary_truncated_has_right_now"]
    )
    n_prev_full_rn = sum(
        1 for r in by_session
        if r["session"] >= 2 and r["prev_summary_full_has_right_now"]
    )
    n_canonical_creating = len(canonical_creating)
    n_canonical_creating_rn_seeded = sum(
        1 for r in canonical_creating if r["prev_summary_truncated_has_right_now"]
    )

    summary = {
        "n_sessions": n_total,
        "n_sessions_with_prev": n_with_prev,
        "n_prev_summary_truncated_has_right_now": n_prev_truncated_rn,
        "n_prev_summary_full_has_right_now": n_prev_full_rn,
        "n_canonical_creating_opener": n_canonical_creating,
        "n_canonical_creating_opener_rn_seeded": n_canonical_creating_rn_seeded,
        "headline": {
            "opener_TIME_3_by_prev_truncated_right_now": headline_time3,
            "opener_JOINT_by_prev_truncated_right_now": headline_joint,
            "opener_PRES_by_prev_truncated_right_now": headline_pres,
        },
        "canonical_creating_opener": {
            "n": n_canonical_creating,
            "JOINT_by_prev_truncated_right_now": canon_joint,
            "TIME_3_by_prev_truncated_right_now": canon_time3,
        },
        "full_response_check": {
            "JOINT_by_prev_FULL_right_now": full_joint,
        },
        "by_session": by_session,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    # ----- console summary -----
    def fmt_row(label, d):
        rate = d.get("rate")
        rate_str = f"{rate:.1%}" if rate is not None else "-"
        return (f"  {label:<30} n={d['n']:>3} n_eff={d['n_eff']:>3} "
                f"y_true={d['y_true']:>3} ({rate_str})")

    print()
    print(f"S138 — Thor sessions scanned: {n_total}")
    print(f"  prev_summary truncated [:200] contains 'right now': "
          f"{n_prev_truncated_rn}/{n_with_prev} = "
          f"{n_prev_truncated_rn/max(n_with_prev,1):.1%}")
    print(f"  prev_summary FULL response contains 'right now': "
          f"{n_prev_full_rn}/{n_with_prev} = "
          f"{n_prev_full_rn/max(n_with_prev,1):.1%}")
    print()
    print("Headline 1 — Opener TIME_3 by prev_summary truncated 'right now':")
    for k, d in headline_time3.items():
        print(fmt_row(f"prev_truncated_rn={k}", d))
    print()
    print("Headline 2 — Opener JOINT by prev_summary truncated 'right now':")
    for k, d in headline_joint.items():
        print(fmt_row(f"prev_truncated_rn={k}", d))
    print()
    print("Headline 3 — Opener PRES by prev_summary truncated 'right now':")
    for k, d in headline_pres.items():
        print(fmt_row(f"prev_truncated_rn={k}", d))
    print()
    print(f"Canonical creating opener (n={n_canonical_creating}, "
          f"of which {n_canonical_creating_rn_seeded} have prev rn-seeded):")
    print("  JOINT by prev_truncated_rn:")
    for k, d in canon_joint.items():
        print(fmt_row(f"  rn={k}", d))
    print("  TIME_3 by prev_truncated_rn:")
    for k, d in canon_time3.items():
        print(fmt_row(f"  rn={k}", d))
    print()
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

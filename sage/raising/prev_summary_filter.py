"""Schema-fragment detection for previous-session summary re-injection.

The 0.5B LoRA-induced burst basin (see S89/S90/S91 insights) perpetuates
itself cross-session because eight raising runners extract the last SAGE
response verbatim from prior-session JSON and splice it into the next
session's system prompt. When that last response is already schema-mode
output (e.g. "What's the next step? What's the next decision? ..."), the
injection seeds the same basin on the next forward pass.

This module provides the canonical filter and safe-fallback helpers that
each `_get_previous_session_summary` call site should route through.

Validation (from S91 cross-tab, re-confirmed S92 on Sprout 0.5B sessions
62-121):

  Rule: `(qmarks >= 5) OR (schema_phrases >= 1)` on FULL response
  - 11/11 burst memory-asks flagged
  - 0/93 non-burst memory-asks flagged
  - Threshold ratio ~50:1 on qmark count alone

The filter must be applied to the *untruncated* text. On S109/S110, the
schema phrase falls beyond the first 50 characters; filtering the :50
form misses the burst. Apply at the earliest point that full text is
available (typically the closing handler that writes last_session_summary
to state, and the reader that extracts from session JSON).
"""

from __future__ import annotations

import re

_SCHEMA_PHRASE_RE = re.compile(
    r"what'?s\s+(?:the\s+next|causing|happening|on\s+the|going\s+to)|what\s+is\s+the\s+next",
    re.IGNORECASE,
)

_QMARK_THRESHOLD = 5


def is_schema_fragment(text: str) -> bool:
    """Return True iff text matches the S91/S92 burst-schema signature.

    Apply to the full response, not a truncated form. See module docstring.
    """
    if not text:
        return False
    if text.count("?") >= _QMARK_THRESHOLD:
        return True
    return bool(_SCHEMA_PHRASE_RE.search(text))


def safe_prev_summary(
    last_sage_response: str,
    session_number: int,
    phase_name: str,
    state_fallback: str = "",
) -> str:
    """Build a prev-summary injection string, filtered for schema fragments.

    If `last_sage_response` is a schema fragment, skip the verbatim splice
    and fall back (in order of preference):
      1. state_fallback — pre-computed summary from identity state, if non-schema
      2. generic phase string — "Session N was in {phase} phase."

    Otherwise, preserve existing behavior: verbatim :200 splice with the
    canonical "you said you wanted to remember:" framing.
    """
    if last_sage_response and not is_schema_fragment(last_sage_response):
        return (
            f"Last session (Session {session_number}), you said you wanted to "
            f"remember: {last_sage_response[:200]}"
        )
    if state_fallback and not is_schema_fragment(state_fallback):
        return state_fallback
    return f"Last session was Session {session_number} in {phase_name} phase."


def safe_state_summary(
    memory_response: str,
    session_number: int,
    phase_name: str,
    tag: str = "",
) -> str:
    """Build the `last_session_summary` value written to identity state.

    Replaces the raw `f"Session {n} ({tag}): {phase} phase. {memory_response[:50]}..."`
    pattern that eight runners use at session close. If `memory_response`
    is a schema fragment, the :50 truncation would still carry partial
    schema forward and re-seed the fallback path; suppress in that case.
    """
    tag_part = f" ({tag})" if tag else ""
    if memory_response and not is_schema_fragment(memory_response):
        return (
            f"Session {session_number}{tag_part}: {phase_name} phase. "
            f"{memory_response[:50]}..."
        )
    return f"Session {session_number}{tag_part}: {phase_name} phase."


if __name__ == "__main__":
    # Self-validation against real Sprout 0.5B burst sessions.
    # Run from repo root: `python3 -m sage.raising.prev_summary_filter`
    import json
    from pathlib import Path

    sprout_dir = (
        Path.home()
        / "ai-workspace/SAGE/sage/instances/sprout-qwen2.5-0.5b/sessions"
    )
    if not sprout_dir.exists():
        raise SystemExit(f"Sessions dir not found: {sprout_dir}")

    known_burst = {68, 83, 87, 88, 89, 90, 109, 110, 111, 112, 113}
    caught = missed = false_pos = ok = 0
    missed_samples: list[str] = []
    fp_samples: list[str] = []

    for sf in sorted(sprout_dir.glob("session_*.json")):
        m = re.search(r"session_(\d+)\.json", sf.name)
        if not m:
            continue
        n = int(m.group(1))
        try:
            d = json.loads(sf.read_text())
        except Exception:
            continue
        conv = d.get("conversation", [])
        ma = ""
        for i in range(len(conv) - 1, -1, -1):
            if conv[i].get("speaker") == "SAGE":
                if i > 0 and "remember" in conv[i - 1].get("text", "").lower():
                    ma = conv[i].get("text", "")
                    break
        if not ma:
            continue

        flagged = is_schema_fragment(ma)
        is_burst = n in known_burst
        if is_burst and flagged:
            caught += 1
        elif is_burst and not flagged:
            missed += 1
            missed_samples.append(f"S{n}: {ma[:120]}")
        elif not is_burst and flagged:
            false_pos += 1
            fp_samples.append(f"S{n}: {ma[:120]}")
        else:
            ok += 1

    print(
        f"Sprout 0.5B: caught {caught}/{len(known_burst)} known bursts, "
        f"{missed} missed, {false_pos} flagged non-burst, {ok} clean non-burst"
    )
    for s in missed_samples:
        print(f"  MISS {s}")
    for s in fp_samples:
        print(f"  FLAG {s}  (inspect — may be additional burst not in curated list)")

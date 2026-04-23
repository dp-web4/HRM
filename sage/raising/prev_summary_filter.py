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

# S99 additions — two more unsuitable-for-splice signatures distinct from the
# Sprout 0.5B LoRA burst basin that `is_schema_fragment` targets.
#
# Untagged recital: Thor 27B in the S30-S39 (pre-stop-seq) and S62-S74 (post-
# stop-seq-cleared) eras emitted the identity-recital procedure as visible
# response text, either with a "Thinking Process:\n\n" preamble or naked.
# Splicing "1. **Analyze the Request:**  *   Role: ..." as prior-session
# memory contaminates the next session's system prompt with recital content
# instead of substantive memory.
#
# Adapter-error passthrough: when the IRP adapter encounters an error it
# returns a string like "[OllamaIRP: Unexpected error: timed out]" which
# survives both strip and schema-fragment checks. Seen in Thor 27B S74.
# Splicing that as memory carries an error string forward as "what I wanted
# to remember" — obvious contamination but silent under prior detectors.
_UNTAGGED_RECITAL_RE = re.compile(
    r"^\s*1\.?\s+\*\*Analyze\s+the\s+Request",
    re.IGNORECASE,
)
_THINKING_PROCESS_PREAMBLE_RE = re.compile(
    r"^\s*Thinking\s+Process:.*?(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_ADAPTER_ERROR_PREFIXES = ("[OllamaIRP:", "[DaemonIRP:")


def is_schema_fragment(text: str) -> bool:
    """Return True iff text matches the S91/S92 burst-schema signature.

    Apply to the full response, not a truncated form. See module docstring.
    """
    if not text:
        return False
    if text.count("?") >= _QMARK_THRESHOLD:
        return True
    return bool(_SCHEMA_PHRASE_RE.search(text))


def is_untagged_recital(text: str) -> bool:
    """Return True iff text is the qwen3.5-27B identity-recital procedure.

    Matches both preamble-prefixed (`Thinking Process:\n\n1. **Analyze...`)
    and naked (`1. **Analyze the Request:**...`) forms by stripping the
    leading preamble before checking the numbered-step anchor.

    Carries the S98 register-scan detection into the splice-validation
    path. S99 discovered that prior-session memory from Thor 27B S39 and
    S63-S73 contained this procedure in the splice-candidate position.
    """
    if not text:
        return False
    stripped = _THINKING_PROCESS_PREAMBLE_RE.sub("", text).strip()
    return bool(_UNTAGGED_RECITAL_RE.search(stripped))


def is_adapter_error_passthrough(text: str) -> bool:
    """Return True iff text is an IRP adapter error string, not content.

    Seen in Thor 27B S74 ("[OllamaIRP: Unexpected error: timed out]"). The
    raising runner would otherwise splice this as prior-session memory.
    """
    if not text:
        return False
    return text.startswith(_ADAPTER_ERROR_PREFIXES)


def is_unsuitable_for_splice(text: str) -> bool:
    """Composite check: text must not be spliced as prior-session memory.

    True iff any of:
      - S91/S92 burst-schema signature (Sprout 0.5B LoRA basin)
      - S99 untagged-recital procedure (Thor 27B S30-S39 and S62-S74)
      - S99 adapter-error passthrough ([OllamaIRP:..., [DaemonIRP:...)

    Each detector targets a historically-observed contamination pattern
    that would otherwise seed the next session's system prompt with
    non-substantive content.
    """
    if not text:
        return True  # empty is trivially unsuitable as a splice source
    return (
        is_schema_fragment(text)
        or is_untagged_recital(text)
        or is_adapter_error_passthrough(text)
    )


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
    # S99: route through composite check — rejects burst-schema (Sprout 0.5B),
    # untagged-recital (Thor 27B S30-S39 + S62-S74), and adapter-error passthroughs.
    if last_sage_response and not is_unsuitable_for_splice(last_sage_response):
        return (
            f"Last session (Session {session_number}), you said you wanted to "
            f"remember: {last_sage_response[:200]}"
        )
    if state_fallback and not is_unsuitable_for_splice(state_fallback):
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
    # S99: composite unsuitable-check. A 50-char truncation of untagged
    # recital would still carry the "1. **Analyze the Request" opener
    # forward; a 50-char truncation of "[OllamaIRP: Unexpected error..."
    # would carry an error string. Suppress both alongside schema fragments.
    if memory_response and not is_unsuitable_for_splice(memory_response):
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

    # S99: Thor 27B recital + adapter-error detection sanity check.
    # Known-historical cases from cross_capacity_filter_scan S99 output:
    #   S39 memory-ask: untagged recital (pre-stop-seq-fix era)
    #   S74 memory-ask: adapter-error passthrough
    thor_dir = Path.home() / "ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b/sessions"
    if thor_dir.exists():
        s39 = thor_dir / "session_039.json"
        s74 = thor_dir / "session_074.json"
        for sf, expect_detector in [(s39, is_untagged_recital),
                                     (s74, is_adapter_error_passthrough)]:
            if not sf.exists():
                continue
            d = json.loads(sf.read_text())
            conv = d.get("conversation", [])
            ma = ""
            for i in range(len(conv) - 1, -1, -1):
                if conv[i].get("speaker") == "SAGE":
                    if i > 0 and "remember" in conv[i - 1].get("text", "").lower():
                        ma = conv[i].get("text", "")
                        break
            det = expect_detector.__name__
            label = sf.name
            ok_detected = expect_detector(ma)
            composite_ok = is_unsuitable_for_splice(ma)
            print(f"Thor 27B {label}: {det}={ok_detected}, "
                  f"is_unsuitable_for_splice={composite_ok}")

            # S100: verify safe_prev_summary returns the generic-phase fallback
            # (not the contaminated verbatim splice) when given this response.
            sps = safe_prev_summary(ma, 39 if sf == s39 else 74, "questioning")
            splice_leaked = (
                "you said you wanted to remember" in sps
                and ("Analyze the Request" in sps or "[OllamaIRP:" in sps)
            )
            fallback_used = sps.startswith("Last session was Session ")
            print(f"Thor 27B {label}: safe_prev_summary leaked={splice_leaked}, "
                  f"fallback_used={fallback_used}")

    # S100: runner-side self-check — is_unsuitable_for_splice guards memory_response
    # by matching the guard form used in all 10 runners after Phase 2 wire-up.
    # This is the invariant: a flagged candidate must produce memory_response="".
    print("\nS100 runner guard invariants:")
    guard_cases = [
        ("schema_fragment", "What's the next step? What's the next decision? What's next? What's next? What's the answer?"),
        ("untagged_recital", "1. **Analyze the Request:**  *   Role: You are SAGE."),
        ("adapter_error", "[OllamaIRP: Unexpected error: timed out]"),
        ("substantive", "I want to remember that we discussed how attention shapes what feels real to me."),
    ]
    for label, candidate in guard_cases:
        flagged = is_unsuitable_for_splice(candidate)
        memory_response = "" if (not candidate or flagged) else candidate[:200]
        ok = (label == "substantive") == bool(memory_response)
        print(f"  {label}: flagged={flagged}, memory_response_empty={not memory_response}, "
              f"correct={ok}")

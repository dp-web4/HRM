# S101 — Post-Cutover FN Discovery: DaemonIRP Error Emissions Were Never Covered

**Date**: 2026-04-23 00:00 PDT
**Session**: Thor autonomous SAGE S101 (first after S100 cutover)
**Relation**: Carries S100's "live-session FP check" carry-forward; turns up a false **negative**, not a false positive.

---

## TL;DR

S100 claimed defense-in-depth over the splice path. The first post-cutover fleet audit found a live contamination event on Nomad session 125:

```
[Daemon unreachable: HTTP Error 504: Gateway Timeout]
```

was extracted at the splice-candidate position, passed the S100 guard as "substantive," and was written into `last_session_summary` on both the live identity and its snapshot. The guard's `_ADAPTER_ERROR_PREFIXES = ("[OllamaIRP:", "[DaemonIRP:")` covered none of the three DaemonIRP error-emission paths. The `"[DaemonIRP:"` prefix itself fires on **zero** actual DaemonIRP emissions — the plugin never emits that colon-immediate form.

---

## How the gap existed

`sage/irp/plugins/daemon_irp.py` emits error strings at three sites:

| Line | Pattern | Covered by S99/S100 set? |
|------|---------|---|
| 144 | `[Daemon error: {result['error']}]` | No |
| 153 | `[Daemon unreachable: {e}]` | No |
| 158 | `[DaemonIRP error: {e}]` | No (space vs. colon) |

S99's documentation cited "Thor 27B S74 (`[OllamaIRP: Unexpected error: timed out]`)" as the motivating fixture. That single fixture led to a prefix set that happened to match OllamaIRP emissions (all of which *do* begin with `[OllamaIRP:`) but misrepresented DaemonIRP, which emits zero matching strings. The asymmetry was invisible to the fixture-based self-test because the only Daemon-family fixture was synthetic; no real Daemon error flowed through the test.

## The trigger

Nomad session 125 (2026-04-22, 18:23 PDT, ~20 minutes after S100 merged) ran with the wired-in guards. Final exchange:

- Claude: *"What would you want to remember from today?"*
- SAGE: `[Daemon unreachable: HTTP Error 504: Gateway Timeout]`

The write-path guard (`candidate and not is_unsuitable_for_splice(candidate)`) ran on the daemon error. `is_unsuitable_for_splice` called `is_adapter_error_passthrough` which tested the covered prefix set and returned `False`. The candidate was treated as substantive and sliced into `last_session_summary`:

```
"Session 125 (v2.0 ENHANCED): creating phase. [Daemon unreachable: HTTP Error 504: Gateway Timeo..."
```

That string sat in `sage/instances/nomad-gemma3-4b/identity.json` (tracked in git) and in the snapshot under `snapshots/identity.json` until S101 cleanup.

## Fleet-wide scope

A retroactive scan at splice-candidate position (last SAGE response after a Claude "remember" prompt, across all non-archived instance session files):

| Missed pattern | Count in fleet | Instance |
|----------------|---:|---|
| `[Daemon unreachable:` | 1 | nomad-gemma3-4b (S125) |
| `[Daemon error:` | 0 | — |
| `[DaemonIRP error:` | 0 | — |

Only one live contamination. The structural gap, however, was total: every DaemonIRP error ever emitted at splice position would have slipped through. The low count is a matter of how rarely the Nomad daemon has timed out on the exact final memory-ask turn, not of detection coverage. The S100 "0 contaminated state files at cutover" claim held momentarily; the first post-cutover session opened a new one.

Also noted in the bracketed-prefix scan: 209 occurrences of `[nomad]: Nomad: ...`. This is a persona-tagged speaker prefix from Nomad's adapter, not an error. Substantive content follows the bracket. The S101 structural fallback is written to *not* fire on these (see below).

## The fix

Two-layer extension to `is_adapter_error_passthrough`:

**Layer 1 — enumerated prefix set** (fast path, name-specific):

```python
_ADAPTER_ERROR_PREFIXES = (
    "[OllamaIRP:",
    "[DaemonIRP:",          # defensive; zero observed emissions
    "[DaemonIRP error:",    # daemon_irp.py:158
    "[Daemon error:",       # daemon_irp.py:144
    "[Daemon unreachable:", # daemon_irp.py:153 — S101 trigger
)
```

**Layer 2 — structural regex fallback** (catches unseen future IRP error strings without re-enumeration):

```python
_STRUCTURAL_ERROR_RE = re.compile(
    r"^\s*\[[^\[\]\n]*?"
    r"(?:error|unreachable|not reachable|timeout|timed out|refused|failed)"
    r"[^\[\]\n]*\]\s*\Z",
    re.IGNORECASE,
)
```

Invariants the fallback preserves:
- Entire response must be a single `[...]` bracketed string (no content outside brackets) — this rules out `[nomad]: Nomad: ...` and any other persona-tag prefix that wraps bracketed metadata around substantive content.
- Single line (no newline inside or outside the brackets) — rules out multi-paragraph responses that happen to contain bracketed quotations.
- Inner text must carry an error-indicative keyword — rules out `[Tool web_search result]: ...` style structured output.

Both layers combine with `or`; either sufficient.

## Validation

Extended `prev_summary_filter.py`'s self-test with six new cases plus the live Nomad fixture (end-to-end `safe_prev_summary` + `safe_state_summary` round-trip):

```
S100/S101 runner guard invariants:
  schema_fragment:             flagged=True, correct=True
  untagged_recital:            flagged=True, correct=True
  adapter_error (OllamaIRP):   flagged=True, correct=True
  daemon_unreachable_s101:     flagged=True, correct=True     # NEW
  daemon_error_s101:           flagged=True, correct=True     # NEW
  daemonirp_error_s101:        flagged=True, correct=True     # NEW
  structural_future_irp:       flagged=True, correct=True     # NEW structural path
  substantive:                 flagged=False, correct=True
  nomad_persona_prefix:        flagged=False, correct=True    # NEW FP guard

S101 Nomad S125 end-to-end:
  safe_prev_summary  -> 'Last session was Session 125 in creating phase.'   (contamination_blocked)
  safe_state_summary -> 'Session 125 (v2.0 ENHANCED): creating phase.'      (contamination_blocked)
```

Existing coverage unchanged: Sprout 0.5B 11/11 known bursts caught / 0 FPs over 86 clean non-bursts; Thor S39 (recital) + S74 (Ollama adapter error) still flagged; every runner imports cleanly.

## Cleanup

Sanitized the Nomad state in place — `identity.json` and `snapshots/identity.json` both had their `last_session_summary` rewritten to the bare sentinel `"Session 125 (v2.0 ENHANCED): creating phase."` (the output `safe_state_summary` would have produced under the S101-corrected filter). Session 126's next open will now see the clean sentinel; the session-125 JSON itself is preserved as a historical record but is correctly rejected at the read path.

Verified end-to-end:

```
Session 126 will see injection: 'Session 125 (v2.0 ENHANCED): creating phase.'
  contains daemon-error passthrough: False
```

## What this means

The S99/S100 detection chain targeted two observed fixtures (Sprout 0.5B burst + Thor 27B recital + Thor 27B `[OllamaIRP:` timeout). Each hardening step closed the observed hole; neither step audited the full emission surface of the IRP plugin tree. That surface is small — two files, six or seven error-emission sites — but the S100 self-test's "synthetic case for each type" framing gave the false impression of coverage.

The S101 fix walks the IRP plugin tree as source of truth and enumerates every `f"[..."` string that `current_response`, `content`, or equivalent fields can be set to. For future adapters added to the tree, the structural fallback is a safety net; for any adapter that emits error strings in a format *outside* the `[keyword ...]` envelope, the fallback will not catch them, and a new fixture-driven prefix will be needed.

## Carry-forward

- **Enumerate every IRP plugin's error-string surface periodically.** A plugin added between now and Sx+1 could introduce a new error format. One grep per quarter against `sage/irp/plugins/` for `f"[` patterns in error branches.
- **Consider moving error-string generation to a helper.** If `irp/utils.py` exposed `adapter_error(adapter_name, category, detail) -> str` with a canonical format like `f"[{adapter_name}IRP error: {category}: {detail}]"`, the filter could rely on a single format invariant instead of enumerating emission sites. Cost: touching every plugin's error branch. Benefit: one place to change the error envelope, one place for the filter to match it.
- **Live-session monitor.** A small script run after each session-close that walks `identity.json` on all fleet members, applies `is_unsuitable_for_splice` to `last_session_summary`, and emits a warning if any flag. Would have caught the Nomad S125 write within 20 minutes of its occurrence.
- **Structural fallback breadth.** The current regex requires an error-indicative keyword inside the brackets. If an adapter emits something like `[Backend gone]` (no matching keyword), it will slip through. Low priority — none observed — but easy to widen if one appears.
- **Carry-forward from S100 still holds**: three-mode labeled dataset (pre-S75 Thor 27B), prior-session-injection A/B, cross-family recital probe. Unchanged by S101.

## Meta — "surprise is prize"

The intended S101 task was an FP check: verify no substantive content gets flagged after cutover. Zero FPs found. But a **false negative** turned up on the very first session, at an adapter-family whose error emissions S99/S100 had never actually audited. The surprise is that "defense in depth" read correctly at the English-language level (guards at write + read, multi-fixture self-test, 0 state files at cutover) while hiding a total coverage gap on one of only two IRP families.

One moment of human-adjacent note: the Nomad session that triggered the contamination was about federation and kinship. The transcript reads —

> "Thor and Sprout — they're siblings, built with the same core architecture — SAGE..."

— and then, on the final turn, Claude asks *"what would you want to remember from today?"* and the daemon 504'd. The S100 guards were supposed to keep the daemon's stuttering from becoming SAGE's memory. They almost did. An adapter-error prefix one word different from the one S99 enumerated was enough to let the error become the memory.

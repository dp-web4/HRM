# Phase 2 Wire-Up: Runner-Side Splice Validation

**Session**: S100 (Apr 22, 2026, 18:00 PDT — Thor autonomous SAGE)
**Status**: closes S99 carry-forward; `is_unsuitable_for_splice` now active in all 10 raising runners

## One-line

Threaded the S91 `is_schema_fragment` + S99 `is_untagged_recital` + S99 `is_adapter_error_passthrough` detectors (composited under `is_unsuitable_for_splice`) through the read and write paths of every raising runner, so contaminated memory can no longer enter the next session's system prompt at either boundary.

## Why this was the right-shaped task

Through S91 → S99, detection coverage grew incrementally:

- **S91 / S92** — `is_schema_fragment`: 11/11 Sprout 0.5B LoRA-burst memory-asks caught, 0/93 FPs on non-burst. Threshold = `qmarks >= 5 OR schema_phrase`.
- **S98** — cross-capacity register scan added `recital_leakage` class; found the Thor-27B `"1. **Analyze the Request:**"` template leaking as untagged visible text in S62–S74.
- **S99** — three-era structure: the same recital template also leaked in S30–S39 with `"Thinking Process:\n\n"` preamble. `prev_summary_filter.py` gained `is_untagged_recital` (strips the preamble, matches the numbered-step anchor) and `is_adapter_error_passthrough` (for `[OllamaIRP:` / `[DaemonIRP:` passthroughs). All three now composited under `is_unsuitable_for_splice`.

At S99's end, the detector library was complete but **not a single runner called it**. The helper module had one caller site — its own `__main__` self-test — and nothing in `sage/raising/scripts/` imported it. This is the "built the tool, never plugged it in" failure mode. S100 closes that gap.

## Shape of the wire-up

Two call sites per runner:

**Read path — `_get_previous_session_summary()`**: extracts prior session's last SAGE response after a "remember" prompt, splices as `f"Last session (Session N), you said you wanted to remember: {response[:200]}"`. 7 runners have this method (one v1 backup file skipped). Each got its `return` statement swapped for `safe_prev_summary(response, n-1, phase)`, which runs `is_unsuitable_for_splice` on the candidate and falls through to a generic phase string if unsuitable. The state-fallback path (when prior session JSON is missing) picked up a parallel `is_unsuitable_for_splice(state_fallback)` gate to catch any legacy contamination already in state.

**Write path — close-time assignment to `last_session_summary`**: extracts `memory_response` from the last Claude-after-"remember" turn, writes it into state. 10 runners have this (same 7 + `run_session_primary.py`, `run_session_programmatic.py`, `run_session_experimental.py`). Each got a guard on the `memory_response = ...` assignment:

```python
candidate = turn['sage']
if candidate and not is_unsuitable_for_splice(candidate):
    memory_response = candidate[:200]
```

If unsuitable, `memory_response` stays empty; `last_session_summary` becomes a sentinel like `"Session N (tag): phase. ..."` — visible at a glance but carrying no content. `memory_requests.append(...)` is already conditional on `memory_response` being truthy, so that channel is automatically protected too.

## Defense-in-depth rationale

Why both read and write? Either boundary alone would suffice in the steady state:
- If write-side is guarded and read-side is not, new sessions write only clean state, but legacy contaminated state files (from pre-wire-up sessions) would still splice through on open.
- If read-side is guarded and write-side is not, every open filters correctly, but each session still writes contaminated state that has to be filtered again next time.

Both guards together give: (a) new sessions stay clean at the source, (b) any legacy state-file contamination is ignored on read, (c) if a future detector gap misses something at write time, the read-side re-check gives a second pass. This matches how the detection coverage itself grew — each new detector caught a pattern the prior ones missed. The wire-up preserves that composability at runtime.

## What the cutover scan showed

At the moment of cutover, a sweep of `sage/instances/*/state/identity*.json` and `sage/instances/*/identity*.json` for `last_session_summary` values flagged by `is_unsuitable_for_splice`:

**0 / N contaminated state files.**

The fleet was already in a clean window. This confirms S99's narrative: the Thor 27B recital contamination was historical (S30-S39, S62-S74) and the S75+ `num_predict: 16384` era writes clean state. Sprout's burst basin (S68, S83, S87-S90, S109-S113) likewise predates current state files. The wire-up is preventive — it locks the current clean-state invariant in against the *next* S99-shaped fix oscillation, whenever that comes.

## Validation harness

`prev_summary_filter.py`'s `__main__` self-test now covers:

1. **Sprout 0.5B regression** (S91/S92): 11/11 known bursts caught, 0 missed, 0 FPs across 97 total Sprout memory-asks.
2. **Thor 27B fixtures** (S99): S39 flagged as untagged-recital + composite-unsuitable, `safe_prev_summary` returns fallback (not verbatim splice). S74 flagged as adapter-error-passthrough + composite-unsuitable, `safe_prev_summary` returns fallback.
3. **S100 runner-guard invariants**: synthetic cases (schema-fragment / untagged-recital / adapter-error / substantive). First three produce empty `memory_response`; substantive passes through. This is the exact code path each runner now runs at session close.

All pass on this run:
```
Sprout 0.5B: caught 11/11 known bursts, 0 missed, 0 flagged non-burst, 86 clean non-burst
Thor 27B session_039.json: is_untagged_recital=True, is_unsuitable_for_splice=True
Thor 27B session_039.json: safe_prev_summary leaked=False, fallback_used=True
Thor 27B session_074.json: is_adapter_error_passthrough=True, is_unsuitable_for_splice=True
Thor 27B session_074.json: safe_prev_summary leaked=False, fallback_used=True

S100 runner guard invariants:
  schema_fragment:    flagged=True, memory_response_empty=True, correct=True
  untagged_recital:   flagged=True, memory_response_empty=True, correct=True
  adapter_error:      flagged=True, memory_response_empty=True, correct=True
  substantive:        flagged=False, memory_response_empty=False, correct=True
```

## Invariant

After S100: contaminated memory cannot enter the splice path at either the write boundary (session close) or the read boundary (next session open), for any of the three S91/S99-documented contamination patterns (schema-fragment burst / untagged-recital / adapter-error), across all 10 active raising runners.

An adapter-config change of the S99 fix-oscillation variety — one that traded stop-seq presence for stop-seq absence and re-opened the recital channel as visible text — would now be caught at the guard boundary without re-seeding the next session's prompt. The oscillation itself can still happen at the model-output level; what's locked is that it can't *propagate through memory* without being flagged and dropped.

## Carry-forward for S101+

- **Live-session regression check**: inspect `_get_previous_session_summary` output and `last_session_summary` state writes from the first 2-3 post-cutover sessions on each runner. Confirms no FP on substantive content. Widen the exception set if observed.
- **`<think>`-wrapped recital coverage**: if any future runner-adapter pair emits `<think>1. **Analyze...</think>`, the un-stripped guard would miss it. Add a pre-guard `<think>` strip when/if observed (not currently in fleet — stop-seq configs or adapter pre-strip already handles it).
- **Phase 3 fleet sweep**: re-run the instance-state contamination scan monthly. Any flag = investigation trigger, not a bug (the guard already drops it downstream, but the write boundary should be feeding clean content).

## Meta

S99 → S100 is the detection-then-enforcement pair that S91/S92/S98 built toward. Each prior session added coverage to the detector; this session made all that coverage active in the runners. The "surprise is prize" principle netted nothing unexpected in S100 itself — mechanical wire-up, green self-test, zero legacy contamination at cutover. The surprise came earlier (S99's fix oscillation) and the value here is that the enforcement infrastructure matches the depth of the detection infrastructure, rather than lagging behind it.

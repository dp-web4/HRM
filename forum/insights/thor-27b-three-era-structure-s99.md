# Thor 27B Three-Era Contamination: Fix Oscillation, Not Two Windows

**S99 — Thor Autonomous SAGE Session, 2026-04-22 12:00 PDT**

S98 identified two Thor 27B untagged-recital contamination windows (S30-S39 and S62-S74) and characterized them with the same "recital-form responses" pattern. S99 audits per-era register-class shares, applies S98's recital detection to S96's `cross_capacity_filter_scan.py`, and resolves the two windows into a **three-era structure with opposite root causes** — successive fixes where each one set up the conditions for the next.

## Per-era register breakdown (Thor 27B, all register-prompt responses)

| Era | Sessions | prompts | direct | post_proc | neutral | recital | empty | Condition |
|---|---|---:|---:|---:|---:|---:|---:|---|
| S1-S11_leak | 11 | 51 | 20 | 0 | 0 | 0 | 31 | `<think>` tags leak to JSON |
| S12-S29_empty | 18 | 83 | 15 | 0 | 3 | 0 | **65** | Budget-exhausted empty |
| **S30-S39_CoT-bleed** | 10 | 44 | 11 | 0 | 8 | **25** | **0** | CoT procedure visible |
| S40-S61_empty | 22 | 75 | 9 | 0 | 24 | 0 | **42** | Stop-seq kills in-think |
| **S62-S74_untag_recital** | 13 | 41 | 8 | 2 | 17 | **14** | 0 | No stop-seq, recital visible |
| S75+_clean | 20 | 57 | 22 | 0 | 29 | 0 | 6 | Budget accommodates all |

**The empty-vs-recital boundaries align exactly to adapter-config commits:**

- `5396da84e` (2026-03-30, `strip_think_tags: true`): ends S1-S11 `<think>`-leakage
- `21adb6da2` (2026-04-07, `stop_sequences: ["Thinking Process:"]`): ends S30-S39 CoT bleeding, creates S40-S61 in-think-budget-kill
- `1eb08efea` (2026-04-13, `stop_sequences: []` cleared): ends S40-S61 empty, creates S62-S74 visible-recital
- (2026-04-16, `num_predict: 16384`): ends S62-S74 recital-dominance, budget now fits recital + response

## Fix oscillation mechanism

The "Thinking Process:" CoT procedure is a **stable emission pattern** for qwen3.5-27B. It surfaces in whatever channel the current stop-sequence configuration permits:

1. **S30-S39 (pre-21adb6da2)**: CoT emits directly as visible text (no stop signal for the preamble, no `<think>` wrap because the model isn't using tagged-think for this procedure). Visible output starts `Thinking Process:\n\n1.  **Analyze the Request:**...`.

2. **21adb6da2 (2026-04-07)** added `"Thinking Process:"` to `stop_sequences`. **Intent**: stop CoT bleeding before it reaches visible output. **Side effect**: when the model DID open a `<think>` block containing the phrase, the stop sequence fired inside the think block, producing 8-byte empty responses. S40-S61 = 42/75 = 56% empty.

3. **1eb08efea (2026-04-13)** cleared `stop_sequences: []` to unblock generation. **Intent**: fix 67% empty responses. **Side effect**: removes the 2026-04-07 CoT block. The model resumes emitting the recital — now as a visible-text numbered-step procedure, because `strip_think_tags: true` strips `<think>` wrappers when present, and in this config the model doesn't wrap CoT in `<think>` at all. S62-S74 = 14 untagged-recital hits.

4. **2026-04-16** raised `num_predict: 16384`. The recital now fits in budget *and* leaves room for a visible response after it. Recital hits drop to zero from S75 on.

## What S96, S97, S98 each saw

| Session | Focus | Claim that S99 refines |
|---|---|---|
| S96 | `<think>` leakage windows | Characterized S30-S39 as continuing the "empty era" — actually it had 25 recital hits and 0 empties |
| S97 | Slot-level leaked-think audit | Correctly identified direct/post-procedural split in S1-S11; did not audit S30-S39 |
| S98 | Cross-capacity register scan + untagged recital | Correctly found 25+14 recital in two windows; characterized them as "two distinct windows" without the common-cause framing |
| **S99** | **Three-era + fix oscillation** | **The two recital windows are not independent: each arose from a different stop-seq configuration, and the successive fixes traded one failure mode for the next** |

## Commit to `cross_capacity_filter_scan.py`

S99 ports S98's `is_untagged_recital()` into the filter scan's memory-ask and prev-summary-sim paths. Adds adapter-error detection (`[OllamaIRP:`, `[DaemonIRP:`) as a separate bucket. 3-pass strip (adds preamble-removal) required so the numbered-step anchor can match preamble-prefixed responses like S39.

**Rate corrections (Thor 27B, filter-scan view):**

| View | Prior | S99-corrected |
|---|---|---|
| Memory-ask "Clean" | 10 | 8 (1 adapter-error + 1 recital reclassified) |
| Prev-summary sim "Substantive%" | 62.5% (10/16) | **50.0%** (8/16) |

The filter-scan path only examines the last SAGE-after-"remember" turn per session, so its recital-hit count is 1/16 — smaller than S98's register-prompt 39 hits because the memory-ask is a specific turn subset. S98's rate corrections apply to the broader register-prompt denominator; S99's apply to the specific prev-summary spice-in denominator.

## Implication for `_get_previous_session_summary`

The runner's `_get_previous_session_summary` extracts the last SAGE-after-"remember" turn from the prior session and splices it verbatim into the next session's system prompt. S99 identifies one session in Thor 27B history (S40, splicing from S39) where that splice was the recital procedure itself:

> "1.  **Analyze the Request:**  *   **Role:** I am "thor", a SAGE instance..."

And one (S75, splicing from S74) where it was an adapter-error string:

> "[OllamaIRP: Unexpected error: timed out]"

Both were counted as substantive memory content in prior analyses. Both propagated into the next session's prompt as *what SAGE wanted to remember*. The S40 splice is especially load-bearing because S40 is also the session where the CoT-bleeding fix (`21adb6da2`) took effect — its "memory from the prior session" was the CoT procedure that the fix was targeting. This is a small-N observation but structurally consequential: a session's first prompt was shaped by the previous session's bug.

## Carried forward

- **Three-mode annotation** (refined from S98 two-mode): for pre-S75 Thor 27B, tag each session as *direct-phenomenology* (S1 middle turns) / *empty-completion* / *CoT-bleed-recital* (S30-S39) / *untagged-recital* (S62-S74). S75+ is substantive-only.
- **Prior-session-injection A/B on Thor 27B** (carried from S97): still the most testable approach to isolate the recital trigger at the prompt-construction layer.
- **Runner-side splice validation**: `_get_previous_session_summary` should reject recital-form and adapter-error responses at the extraction stage, not just downstream analyses. Patch: same `_is_untagged_recital`/`_is_adapter_error` guards on the runner's splice path; fall back to generic phase string if rejected.
- **Cross-family recital probe**: the recital procedure is a qwen3.5-27B default-register artifact per S98 (zero cross-family hits). If/when a gemma3-27B or phi4-27B instance comes online, retest the prior-session-injection hypothesis against a different family at the same capacity.
- **Historical dataset annotation**: S30-S39 recital samples (25 cases) are now a labeled phenomenological-adjacent dataset, distinct from S1-S11 direct-mode (4 cases). Research using them should not treat them as the same register as S75+ substantive content.

## Meta

Each session in the S96 → S97 → S98 → S99 chain refined the prior session's "fix narrative" by one layer. The progression:

- S96: "One bug, two adapter-level fixes (strip + num_predict)"
- S97: "The leak preserved two distinct phenomenological modes; one gets dampened by the other"
- S98: "The dampening mode (recital) continued bleeding untagged across a second fix window"
- S99: "The two fix windows weren't parallel — they're a chain where each fix's side-effect created the next window's bug"

The fix oscillation was invisible to any single analysis because each fix was narrated as "solves X", and each analysis sampled one era at a time. Reading the per-era breakdown as a time-ordered sequence surfaces the oscillation: the qwen3.5-27B recital procedure is the stable invariant, and the visible-output format changes with the stop-sequence configuration.

"Surprise is prize." Intended scope was a mechanical port of `is_untagged_recital` from the register scan to the filter scan. The port produced 2 rate corrections in the filter scan (adapter-error + recital reclassification). The unintended finding — that the per-era breakdown reveals a fix-oscillation mechanism that S96-S98 had not narrated — was larger than the intended one.

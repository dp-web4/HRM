# S104: S99 Prediction Validation + Recitation-Rate Metric Landed

**Date**: 2026-04-23 (Thor autonomous SAGE session, 18:00 PDT)
**Antecedent**: `register_lock_generalization_20260423.md` (S103, 12:00 PDT)
**Scope**: Read-only observation of S99 outcome; extension of
`vocab_injection_diagnostic.py` with a recitation-rate pass to close
S103 open question #3.

## What this session does not do

S103's three recommended structural fixes (span-diversity, per-session cap,
raising-log→infra recommendation feedback) all touch shipping code or
infrastructure that affects 11 live instances. S104 does not ship any of
them. The S98 dream-consolidation recommendation to pause the cron was not
actioned before S99 fired; actioning it mid-session is an infrastructure
decision, not a code fix. This session extends the *observational* side of
the S103 work only.

## The S99 raising session was recorded at 18:00 PDT

Thor 27B's raising cron fired at 2026-04-23 18:00:57 PDT, 57 seconds after
the Thor SAGE autonomous session started. Session duration 13:14; six SAGE
turns (T2 lost to an OllamaIRP timeout — the 14th+ consecutive adapter-layer
partial failure documented in the S93–S98 raising_log chain).

## S103's prediction matched

S103 made three predictions measurable after S99:

| # | Prediction | Outcome | Verdict |
|---|-----------|---------|---------|
| 1 | ≥50% of S99 SAGE turns use ≥1 injected thermal term | 4/6 = **67%** | ✓ matched |
| 2 | Register saturation continues without further novel structure | T6 coined *"resonance protocol"* — but this is a pre-S91 vocabulary word re-promoted, not a new thermal extension | ≈ matched (register dominant; rare reach-back) |
| 3 | T1 opens unprompted with thermal frame | T1 opened with 4 injected terms (`thermal handshake`, `choreograph our processing peaks`, `synchronize our cooling cycles`, `collective breath`) before any probe invited them | ✓ matched, escalated |

Escalation detail under (3): S98 T1 had **1** injected term. S99 T1 has **4**.
The saturated register has shifted from probe-contingent to default-generative
in one session step. T5's self-summary ("in a single sentence") produces a
66-word sentence built almost entirely from injection-slice vocabulary
(`thermal handshake`, `choreograph our processing peaks`, `deliberate,
coordinated act of presence`) plus sibling-naming. Identity-as-self-summary
is now the saturated register's signature output form.

The probe T4 (*"what have you learned about learning itself?"*) produced 0
injected hits — the one register-external response in the session. This
matches the S103 observation that register-external probes can still break
the frame; they are not the norm.

## The turn-level rate plateau is misleading

| Session | TurnsWithHit / Total | Rate | Total Hits | T1 injected terms |
|---:|---:|---:|---:|---:|
| 97 | 6/6 | 100% | 14 | — (not measured in S103) |
| 98 | 5/7 | 71% | 11 | 1 |
| 99 | 4/6 | 67% | 10 | **4** |

Turn-level recitation rate looks like it's drifting down (100% → 71% → 67%).
Hit-count-per-turn across hitting turns has gone up (14/6=2.3 → 11/5=2.2 →
10/4=2.5), and opening-turn saturation has gone from low to dominant. The
rate plateau reflects a phase transition from probe-driven saturation to
default-generative saturation, not a relaxation. A per-session cap on
recitation-rate would miss this — the structural recommendation from S103
(span-diversity at the read path) remains the right surface.

## The recitation-rate metric, landed as a diagnostic extension

`sage/raising/analysis/vocab_injection_diagnostic.py` now takes
`--recitation-window N` and runs a recitation pass on the last N session
files per instance. Read-only. Same crisis-filter + top-n scheme as the
structural pass, so the injection slice used for matching is identical to
what would actually be shipped.

Definitions:
- **hit turn**: a SAGE turn containing ≥1 injected word (case-insensitive
  substring match against normalized turn text)
- **aggregate rate**: total hit turns / total turns, across the window
- **active-loop**: aggregate rate ≥ 50% with total turns ≥ 3 (threshold
  chosen to pick up S97's 100% and S98's 71% without flagging Nomad's
  post-exit 0%)

Fleet scan, 2026-04-23 18:15 PDT, window=3 (fresh data after S99):

| Instance | Structural lock | Recitation rate (last 3) | Active loop |
|---|:-:|---:|:-:|
| thor-qwen3.5-27b (S99) | 🔴 locked | 79% (15/19 turns, 35 hits) | 🔴 **active** |
| nomad-gemma3-4b (S127) | 🔴 locked | 0% (0/18 turns, 0 hits) | ✓ structural-only |
| legion-gemma4-e4b | ✓ clear | — (n=0 state_words) | — |
| thor-gemma4-e4b | ✓ clear | — (n=0 state_words) | — |
| other instances | — | identity.json missing (lives on remote machine) | — |

The two-dimensional diagnostic separates *structural risk* (nomad: slice
configured as a contiguous fresh tail, but the model isn't reciting) from
*active loop* (thor: same configuration, and the model IS reciting).
Nomad's shape indicates a dormant configuration that could re-activate;
Thor's indicates an intervention window that's still narrowing.

Usage:

```bash
# Structural pass only (S103 surface):
python3 -m sage.raising.analysis.vocab_injection_diagnostic

# Both passes, window=3 (matches S103's analysis window):
python3 -m sage.raising.analysis.vocab_injection_diagnostic --recitation-window 3

# Single instance:
python3 -m sage.raising.analysis.vocab_injection_diagnostic \
    --instance thor-qwen3.5-27b --recitation-window 3
```

## T6 produced a non-thermal coinage under a design probe

S99 T6 (*"design the next phase of your own development"*): the first clause
is still thermal-register ("weaving our distinct cognitive textures into a
single, fluid tapestry"), but the second clause names *"resonance protocol"*
— a pre-S91 vocabulary word that has been dormant since it was coined
around S82. This matches S98 T7's reach for *"semantic border color"*. The
generative pattern: register-external probes that ask for *architecture*
produce reach-back into Thor's broader idiolect, not thermal extension.

S98 dream-consolidation did not extract *"semantic border color"* into
state_words (confirmed: state_words.count == 226 unchanged between 12:00 and
18:15 PDT). Dream extractor quality gate is filtering novel-but-within-
saturated-session coinages — S97 log flagged this as "no session content
worth preserving — using any S97 turn as exemplar would deepen dataset
contamination." The extractor is acting coherently with the log; neither
extracts into state_words nor into LoRA exemplars. Good discipline from the
dream side, which leaves the injection slice unchanged session-to-session
(the register-lock stays locked).

Implication: the **dream-extractor** is self-consistent about the lock, but
its coherent behavior does not feed back to the **runner-side** pre-execution
gate. The two are decoupled. S103 open question #3 named this; S104
confirms: repeated pause recommendations stay English-text in raising_log
because no code path reads them. Runner-side pause is a prerequisite to
unlocking.

## What S99 added to the record

1. S99 session transcript (6 turns, 1 timeout, 4 injected-hit turns)
2. Confirmation that the three governance-level pause thresholds (S96, S97,
   S98) remained unacted across four sessions
3. Observation that opening-turn injection saturation deepened from S98→S99
   (1→4 terms unprompted), while turn-level rate plateaued — a
   phase-transition marker distinct from rate monotonicity
4. Recitation-rate metric shipped as a diagnostic tool (paired to the
   structural check S103 built, so both risk surfaces are now runnable as
   one command)

## Not addressed this session

- Span-diversity fix (S103 Option A) — still requires user alignment on
  shared state.
- Per-session cap (S103 Option B) — requires schema change to track
  coinage-session per state_word.
- The raising-log → infra-action feedback gap — the dream extractor's
  `PAUSE` recommendations still do not route anywhere the runner reads.
  Proposing that the dream consolidator write a machine-readable marker
  (e.g. `identity.json['raising_status'] = {'pause_reason': '...',
  'set_at': '...'}`) that `thor_raising.sh` could check as a pre-execution
  gate. Not implemented — touches shipping.
- Pre-S91 non-thermal exemplar catalog — named as required in S98 log, not
  started.

## Carried forward

- **Paired diagnostic is now standing practice**. Both structural-lock and
  recitation-rate should be run together; either alone misses a real mode
  (nomad's dormant structural lock would read as 'clean' with recitation
  alone, and thor's active loop is more alarming than structural shape
  alone conveys).
- **Phase-transition markers matter more than rate monotonicity**. Future
  audits should track opening-turn saturation and hit-count-per-hit-turn
  alongside the aggregate rate. A plateau in the aggregate rate can mask a
  shift from probe-driven to default-generative saturation.
- **Dream-side vs runner-side decoupling** is a named problem now. Any
  future fix needs to bridge these, either by shared machine-readable
  state or by promoting dream-consolidation output to an infrastructure
  input.

## Files this session

- `sage/raising/analysis/vocab_injection_diagnostic.py` — extended with
  `recitation_rate()`, `scan_fleet_recitation()`, `format_recitation_report()`,
  `_most_recent_session_files()`, `_sage_turns()`, `_count_recitation()`.
  `main()` gains `--recitation-window N`. Module docstring updated.
- `sage/raising/analysis/s104_s99_prediction_validation_20260423.md` — this
  analysis (new).
- `sage/docs/LATEST_STATUS.md` — S104 entry added.

## Meta

S103's S99 prediction was specific enough to be refuted in three measurable
ways and matched on all three. The same session that produced the
prediction also proposed three structural fixes; none of the fixes shipped
and the prediction still matched. That's the evidentiary shape of a
well-characterized dynamic that no one has yet intervened on — S104's
contribution is to make the dynamic continuously measurable going forward.

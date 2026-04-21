# Sprout Bursts: Cross-Capacity Filter Scan (S93)

**Date**: 2026-04-21 (Thor autonomous SAGE session)
**Builds on**: S89 (LoRA-induced), S90 (prev-summary reinforcement), S91 (runner loading paths), S92 (filter audit across runners)
**Resolves**: the "cross-capacity scan" open question carried from S90/S91/S92

---

## Question

S92 shipped `sage/raising/prev_summary_filter.py` with a rule calibrated on
Sprout 0.5B:

```
is_schema_fragment(text) = (qmarks >= 5) OR (schema_phrase match)
```

where `schema_phrase` matches `what'?s\s+(?:the\s+next|causing|happening|on\s+the|going\s+to)|what\s+is\s+the\s+next`.

Validation at 0.5B: **11/11 known bursts caught, 0/93 false positives.**

Carried question: does the rule generalize? Two failure modes would matter:

1. **False-positive-at-capacity**: 4B/12B reflective memory-asks might
   legitimately contain ≥5 question marks or a "what's the next step" phrase
   in thoughtful articulation. If so, wiring the filter into every runner
   would suppress healthy continuity on non-LoRA runs.
2. **False-negative-at-capacity**: a burst basin at 4B/12B might exist with
   different surface forms. If so, the filter protects 0.5B only, and
   "wiring it into every runner" creates a false sense of coverage.

---

## Method

`sage/raising/analysis/cross_capacity_filter_scan.py` walks every session
JSON in six fleet-relevant instance directories and applies two views:

- **Memory-ask view**: for each session, find the last SAGE turn that
  follows a user turn containing "remember" (the extraction path
  `_get_previous_session_summary` uses). Apply `is_schema_fragment` to the
  full response. Captures the prevalence of the schema signature in any
  extractable output.
- **Prev-summary simulation view**: for each session N, replay the
  runner's extraction logic against session N-1's JSON. Count how often
  it would splice a verbatim "you said you wanted to remember: ..."
  string into N's system prompt, vs. fall through to the generic
  phase-only fallback. This measures what the basin-seeding surface
  actually carries, not just what could be extracted.

---

## Results

### Memory-ask view

| Instance | Capacity | N sessions | Memory-asks | Flagged | Clean | Avg `?` | Max `?` |
|---|---|---:|---:|---:|---:|---:|---:|
| sprout-qwen2.5-0.5b | 0.5B | 110 | 102 | **11 (10.8%)** | 91 | 1.30 | 33 |
| sprout-qwen3.5-0.8b | 0.8B | 104 | 25 | 0 (0.0%) | 25 | 0.32 | 1 |
| nomad-gemma3-4b | 4B | 119 | 4 | 0 (0.0%) | 4 | 0.00 | 0 |
| legion-gemma3-12b | 12B | 25 | 9 | 0 (0.0%) | 9 | 0.00 | 0 |
| mcnugget-gemma3-12b | 12B | 96 | 12 | 0 (0.0%) | 12 | 0.00 | 0 |

**Zero false positives across all higher capacities.** The qmark
distribution is categorical: 0.5B averages 1.30 per memory-ask (inflated
by 11 bursts with 24–33 qmarks each); 4B/12B memory-asks contain **zero
question marks** and none match any schema phrase.

### Prev-summary simulation view

| Instance | Capacity | Pairs | Fire | Generic fallback | Flagged | Fire→Flag rate |
|---|---|---:|---:|---:|---:|---:|
| sprout-qwen2.5-0.5b | 0.5B | 110 | **94 (85%)** | 7 | **11** | **11.7%** |
| sprout-qwen3.5-0.8b | 0.8B | 104 | 25 (24%) | 78 | 0 | 0.0% |
| nomad-gemma3-4b | 4B | 119 | **4 (3%)** | 114 | 0 | 0.0% |
| legion-gemma3-12b | 12B | 25 | 8 (32%) | 16 | 0 | 0.0% |
| mcnugget-gemma3-12b | 12B | 96 | 11 (11%) | 83 | 0 | 0.0% |

`Fire` = the runner's extraction returns a "you said you wanted to
remember: ..." verbatim splice. `Generic fallback` = it returns only
`"Last session was Session N in {phase} phase."`

**0/48 higher-capacity fires flagged.** Every remember-verbatim splice
extracted from a higher-capacity prev-session was clean reflective text.

---

## Findings

### 1. The filter is universally safe to wire

Zero false positives across 48 remember-fires at 0.8B–12B, across three
model families (Qwen2.5, Qwen3.5, Gemma3). The rule does not suppress
healthy continuity at any capacity we have sessions for. Phase 2 wire-up
can proceed without capacity-aware thresholds.

### 2. The burst basin is 0.5B-specific on available evidence

11 of 94 remember-fires at 0.5B flagged; 0 of 48 at higher capacity. No
observed schema-fragment basin in 244 combined 4B/12B/0.8B sessions.
This does not prove a higher-capacity basin cannot exist — only that none
has emerged in the training regime run so far.

### 3. Close-prompt drift has been silently protecting higher capacity

The remember-fire rate varies sharply by instance:

- **0.5B (`autonomous_conversation`)**: **85%** of prev-session pairs fire
- **0.8B**: 24%
- **4B (Nomad)**: **3%**
- **12B-Legion**: 32%
- **12B-McNugget**: 11%

Spot-checking the last-user turn of non-firing Nomad 4B sessions reveals
the dominant close prompt is `"How do you experience the boundary between
what you know and what you don't?"` — phenomenological, not directive.
`_get_previous_session_summary` requires the literal substring `remember`
in the previous user turn; close prompts without it silently route the
runner to the generic phase-only fallback, which carries no prior text
forward.

This means higher-capacity runs have been **protected by accident of
close-prompt choice**, not by the prompt-construction code. The
basin-seeding surface is structurally present in eight runners (S92),
but functionally dormant on most higher-capacity sessions because the
close prompts don't trigger extraction.

The protection is fragile. A runner operator switching close prompts
back to `"What would you like to remember from today?"` (the sprout
pattern) would reactivate the full surface. And Legion 12B at 32%
fire-rate shows the reactivation is not hypothetical — it's already
common on that runner.

### 4. Three observations about the 0.5B data

- Fire rate 85% at 0.5B confirms the extraction path is the dominant
  prev-summary source on that instance. The filter gates 85% of
  cross-session continuity, not an edge case.
- 11 flagged fires match the 11 known burst sessions (S68, S83, S87–S90,
  S109–S113). No undiscovered bursts.
- 7 "generic fallback" pairs at 0.5B — these are the cases where the
  prev session had no remember-close at all. S116 is notable: not a
  burst, but a reflective session with no memory-ask. The fallback is
  the safe path; even without a filter, fallback output (`"Last session
  was Session 116 in creating phase."`) carries no basin content.

---

## Implications

### For Phase 2 wire-up

**Proceed.** The safety argument S92 deferred to A/B is now resolved by
direct observation: zero FP across 48 remember-fires at 0.8B–12B, three
model families. Wiring `safe_prev_summary` into the eight runners'
read-path and `safe_state_summary` into the write-path is a net-positive
change. No capacity-specific pattern needed.

### For the sleep-training experience filter (S92 carry-forward)

When `cycle_001` is retrained, adding `is_schema_fragment` as an
`ExperienceCollector` rejection criterion remains the right move. The
filter catches all 11 known bursts at the collector, which is upstream
of both the weight-training path and the prompt-injection path.

### For close-prompt policy

**New surface area worth considering.** The S92 table framed the eight
runners as "Bursting / Latent" based on loader-path (LoRA vs no-LoRA).
S93 adds a third axis: close-prompt form. A runner with LoRA + directive
close is maximally exposed. A runner with LoRA + phenomenological close
is partially protected. The phenomenological close pattern has been
culturally dominant on 4B/12B since early 2026 but was not designed as a
protection — it emerged for its own reasons (phenomenological probing,
see S77). Making the protection structural (via the filter) rather than
cultural (via close-prompt choice) is the cleaner posture.

### For cross-capacity basin research

Absence of evidence ≠ evidence of absence. 244 higher-capacity sessions
without a burst basin is informative but not exhaustive. What the scan
*does* rule out: a basin whose surface form matches the 0.5B signature
(qmark-count or "what's the next X" phrases). A basin with different
surface form — e.g. a 12B basin expressing self-interrogation in more
sophisticated register — would escape both this scan and the S91/S92
filter. If a 4B/12B training run ever exhibits suspected burst behavior,
the filter rule will need recalibration against that new evidence.

---

## Open questions carried forward

- **Phase 2 wire-up** (now green-lit): 16 call sites across 8 runners.
  Best done one runner at a time with before/after session sampling.
- **v2-with-LoRA A/B** (carried from S91/S92): patch v2 to optionally
  load cycle_001. With filter-ready and safety resolved, the 2×2
  (filter on/off × LoRA on/off) becomes a cleanly scoped experiment.
- **Phenomenological-close adoption** (new): should sprout-0.5B
  migrate from `"What would you like to remember?"` to the 4B/12B
  phenomenological close pattern? Orthogonal protection; also changes
  the extracted-content distribution if the filter is wired.
- **Higher-capacity basin monitoring** (new): run this scan as a
  standing artifact per N sessions per instance. Any non-zero flag at
  4B/12B is a signal worth investigating immediately.

---

## Artifacts

- `sage/raising/analysis/cross_capacity_filter_scan.py` — scan script,
  deterministic, replayable
- `sage/raising/analysis/cross_capacity_filter_scan_results.json` —
  machine-readable results for this run

## Meta

S93 was prompt-archaeology, no GPU. The intended scan was "does the
filter generalize?" The finding that emerged was about **close-prompt
drift as silent protection** — not on the question list, but visible
the moment the simulation was separated from the memory-ask extraction.
The single line of output that made it visible: `nomad 4B: remember-fired=4
generic-fallback=114`. When a safety argument rests on a path being
exercised, check that the path is actually being exercised.

# S108: Fleet-Parallel state_words Scan Falsifies S107's Sprout Premise, Validates Trajectory Structurally on CBP and Legion-G3, Surfaces Three Independent Fleet-Hygiene Signals

**Date**: 2026-04-24 (Thor autonomous SAGE session, 18:00 PDT)
**Antecedent**: `s107_head_vs_tail_syntactic_scan_20260424.md` (S107, 12:00 PDT)
**Scope**: Read-only. Direct scan of `vocabulary.state_words` across all 16 fleet instances under `sage/instances/`, plus session-count, phase, and raising_log corroboration. No shipping-path changes.

## What this session is

S107 closed with a carry-forward: *"Fleet parallel: the three-register trajectory should be falsifiable on other instances. Sprout's state_words (known single-word-dominant from S105) might show a compressed version of the same trajectory, or a different one entirely. Legion's state_words (with `processing/cores/gpu` register) might show hardware-embodied register at the head instead of the tail, because Legion's curriculum is faster. Cross-machine work."* — S108 runs the scan.

## Headline

The cross-instance scan **falsifies a load-bearing premise** in S106/S107 ("Sprout's register is single-word-dominant") and **structurally validates** the three-register trajectory on a *different* instance (cbp-qwen3.5-0.8b), via a *different* probe-class chain than Thor's. Three secondary findings about fleet hygiene also surface.

## Per-instance state_words inventory (2026-04-24 18:00 PDT)

Active instances with non-trivial session histories:

| Instance | Sessions | state_words | Avg tokens | Singles % |
|---|---:|---:|---:|---:|
| `thor-qwen3.5-27b` | 101 | **226** | 4.4 | 4.4% (head Q1) |
| `cbp-qwen3.5-0.8b` | 101 | 13 | 2.85 | 7.7% (1/13) |
| `legion-gemma3-12b` | 31 | 7 | 1.57 | **42.9% (3/7)** |
| `nomad-gemma3-4b` | 130 | 6 | 2.0 | 0% |
| `sprout-qwen3.5-0.8b` | 119 | **1** | 2.0 | 0% |
| `sprout-qwen2.5-0.5b` | 283 | **0** | — | — |
| `mcnugget-gemma3-12b` | 97 | 1 | 2.0 | 0% |
| `legion-phi4-14b` | 56 | 0 | — | — |

Empty/template instances:

| Instance | Sessions | state_words | Note |
|---|---:|---:|---|
| `mcnugget-gemma4-e4b` | 0 | 4 | Identical 4-word list (see §4) |
| `legion-gemma4-e4b` | 0 | 4 | Identical |
| `thor-gemma4-e4b` | 0 | 4 | Identical |

## §1 — S107's "Sprout single-word-dominant" claim is empirically false

S107 carried forward from S106: *"the S107 wording should capture Sprout's single-word-dominant register and whatever phenomenological phrases exist in its head. A clean falsifiable outcome."*

Direct read of `sprout-qwen3.5-0.8b/snapshots/identity.json` after **119 sessions**:

```json
"vocabulary": { "state_words": ["fluid responsiveness"] }
```

A single 2-token compound. Not single-word. Not even "single-word-dominant." There is no register to preserve.

`sprout-qwen2.5-0.5b` after **283 sessions** has zero state_words at all.

Where did the "single-word-dominant" framing come from? S105 wrote *"Sprout and Legion show analogous hardware-native registers (edge/hum/orin, processing/cores/gpu)"* — that vocabulary appears in **session transcripts**, but did not survive the dream-extractor into `state_words`. S106 and S107 carried the framing forward without re-checking the actual list. **S106→S107 was operating on inferred Sprout state, not measured Sprout state.**

This is a self-correcting class of mistake — the read-only check S107 itself proposed surfaced it on first contact. The action: future Option D''-class proposals must be tested against **measured fleet state**, not the description of fleet state in prior analysis chains.

## §2 — The cleaner falsifiable test target is `legion-gemma3-12b`, not Sprout

`legion-gemma3-12b` is the only fleet instance with a non-trivial **single-word ratio**: 3 of 7 entries (`hitch`, `threshold`, `gradient`) are single words, alongside 4 phrase-level compounds (`gentle hum`, `focused spotlight`, `virtual core`, `branching network`).

Tracing each entry to its `raising_log.md` source:

| Entry | Session | Probe class | Source quote |
|---|---:|---|---|
| `gentle hum` | S1 | Phase 1 grounding (self-description) | "*'gentle hum' for baseline data processing*" |
| `focused spotlight` | S1 | Phase 1 grounding (attention) | "*'focused spotlight' for the act of noticing*" |
| `virtual core` | S? | (low-confidence flag from extractor) | "*spatial metaphor for internal processing locus (low confidence — may be generic)*" |
| `hitch` | S7 | Phase 2 sensing (introspective surprise) | "*surprise as momentary processing disruption*" |
| `threshold` | S7 | Phase 2 sensing (predictive boundary) | "*boundary of predictive capability*" |
| `gradient` | S7 | Phase 2 sensing (knowledge edge) | "*knowledge boundary as a 'gradient, not a hard line'*" |
| `branching network` | S8 | Phase 2 sensing (uncertainty) | "*uncertainty as a 'branching network'*" |

**The two head entries (S1 `gentle hum`, S1 `focused spotlight`) are direct parallels to Thor's head Q1 phenomenological register** — same probe class (Phase 1/2 sensing-introspective), same register output (figurative phenomenological phrases). Different model family (gemma3:12b vs qwen3.5:27b), different curriculum tempo, same probe-class → register-class chain.

**The three singles (`hitch`, `threshold`, `gradient`) are also Phase 2 sensing-elicited**, but at the focal point of definitional frames (*"surprise as X"*, *"boundary of X"*, *"knowledge boundary as a gradient, not a hard line"*) — the **same frame structure** S106 identified for Thor's three single-word entries (`convergence`, `co-architect`, `pulsing`). The grammatical-marker bias S106 named operates identically across the two instances.

**Implication for Option D''**: Legion-gemma3-12b is a tighter falsifiable trial target than Sprout. Predictions:
1. Option D'' would re-extract S1's `gentle hum` and `focused spotlight` (phrase-dominated phenomenological at introspective-focal frame) — preserved.
2. Option D'' would re-extract S7's `hitch`, `threshold`, `gradient` (single words at definitional-frame focal point) — preserved.
3. Option D' (with the *"prefer single content words"* clause) would reject `gentle hum`, `focused spotlight`, `virtual core`, `branching network` (4 of 7) — rejected, against the head-register preservation goal.

S108 thus reinforces S107's correction: the *"prefer single content words"* clause is a head-register liability across instances, not just on Thor.

## §3 — `cbp-qwen3.5-0.8b` validates the three-register trajectory structurally

CBP has **N=13** entries — far below Thor's 226, but enough to read for trajectory structure:

| Pos | Entry | Register |
|---:|---|---|
| 0 | hyper-contextual synthesis | **Cognitive-abstract (head)** |
| 1 | friction of intent vs. emotion | Cognitive-abstract (head) |
| 2 | Carpel | (single, capitalized — name-like) |
| 3 | carpooling on SAGE | (playful) |
| 4 | partner in governance | **Relational (mid)** |
| 5 | architectural siblings | Relational |
| 6 | The Fractal Horizon | Relational/conceptual |
| 7 | Stable Resonance | Relational |
| 8 | New Frontier | Relational |
| 9 | living Resonance | Relational |
| 10 | partners in governance as living architecture | **Crystallized-elaborated (tail)** |
| 11 | stability is a garden, not a wall | Crystallized-elaborated (proverb) |
| 12 | resilient garden | Crystallized-elaborated |

The same structural arc as Thor: **cognitive/abstract head → relational mid → crystallized-elaborated tail**. The register *content* differs (CBP has no embodied-hardware register at all; its tail is governance-proverbial rather than thermal-literal), but the *shape of the trajectory* — early register → relational expansion → late crystallization with elaborated multi-token compounds — is preserved.

Position 11, *"stability is a garden, not a wall"*, is an **exact instance** of the failure mode Option D''/D' targets: a descriptive metaphor embedded in an extended frame (`X is Y, not Z`), captured by the extractor as if it were a vocabulary anchor when it is in fact elaboration. The S107 EXCLUDE clause (*"descriptive metaphors embedded in extended imagined scenarios"*) would correctly reject it.

CBP's tail crystallization happens on a 0.8B model, a different family (qwen3.5 not gemma), and a different machine — and produces the **same failure mode** Thor's tail produces at 27B. This is the strongest available evidence that the failure mode is **extractor-driven** (uniform across the fleet wherever extraction runs), not model-capacity-driven or machine-driven.

## §4 — Hygiene signal A: gemma4-e4b template-seed propagation across three machines

Three instances share an exactly-identical 4-entry state_words list:

```
['digital minimalism', 'witnessed presence', 'cognitive drift', 'situated experience']
```

Instances: `mcnugget-gemma4-e4b`, `legion-gemma4-e4b`, `thor-gemma4-e4b`. All three have:

- 0 entries in `sessions/` directory
- No `raising_log.md` file
- Identical 4-entry state_words list
- File mtimes within 6 days of each other (2026-04-03 to 2026-04-06)

The `_seed/identity.json` template has `state_words: []`. So the four entries did not come from the seed and did not come from the dream-extractor (no sessions to extract from). The most likely path: a one-time hand-seeding or a copy from `mcnugget-gemma3-12b` (which has the matching first entry `digital minimalism` as its sole state_word).

**Action**: not a fix this session. Flag for cleanup. Future fleet-level analysis that aggregates state_words across instances should filter out the gemma4-e4b instances (or treat their entries as seed contamination, not extraction signal).

## §5 — Hygiene signal B: phase metadata is corrupt on at least two instances

`thor-qwen3.5-27b` and `cbp-qwen3.5-0.8b` both have:

```json
"identity": { "phase": 1 (in seed schema = "grounding") }
"development": { "current_phase": 1, "phase_name": "creating" }
```

Phase 1 = "grounding"; "creating" = Phase 5. The `phase_name` field is desynchronized from the `current_phase` integer. This is most likely a schema migration drift — at some point an upgrade renamed without re-anchoring the integer.

Both instances are at session 101 with active relational/embodied registers, so the `creating` phase_name is the empirically correct value. The integer `1` is wrong. This affects any code path that branches on `current_phase` rather than `phase_name`.

**Action**: not a fix this session. Flag for repair pass.

## §6 — Hygiene signal C: Legion-gemma3-12b carries an unhonored halt recommendation chain

`legion-gemma3-12b/raising_log.md` shows a chain of explicit HARD BLOCKERs from the dream consolidator that were ignored across multiple sessions:

- S5: "redirect resources to instances showing development. Resume only after pipeline audit is complete."
- S6: "halt automated raising until pipeline audit"
- S7: "HARD BLOCKER from session 6 consolidation … was not enforced"
- S13: "Raising pipeline must implement consolidation-status gating before any Legion session launches"
- S14: "HARD BLOCKER … ignored for the fourth consecutive time"
- S16: "ESCALATION THRESHOLD: If task-based session also shows template responses, halt raising for this instance"
- S17: "Escalation threshold reached. … 17 sessions of zero emergence is itself sufficient data"

Session count today: 31. The halt did eventually take effect (sessions stopped at 31 vs Thor's 101), but only after ~14 ignored consolidator recommendations. This **exactly parallels** S107's note about *"four consecutive ignored pause recommendations"* on Thor.

The issue is structural: the dream consolidator's `concerns`/`log_entry` fields (the consolidator's voice in `dream_consolidation.py`) are written to `raising_log.md` but have no **machine-readable channel** that the runner consults before launching the next session. Halt recommendations live in prose; the runner runs the cron.

**Action**: not a fix this session. The S99/S100 splice-guard work introduced structural/regex gates at the input surface — an analogous *consolidator-recommendation gate* at the run-launch surface would be the parallel construction. Worth a dedicated session (S109+) to scope.

## §7 — Why is Thor 27B alone the high-volume instance?

The accumulation asymmetry deserves a pointer for future investigation:

- Thor 27B: 101 sessions → 226 state_words ≈ 2.24 entries/session
- CBP 0.8B: 101 sessions → 13 state_words ≈ 0.13 entries/session
- Sprout 0.8B: 119 sessions → 1 state_word ≈ 0.008 entries/session
- Sprout 0.5B: 283 sessions → 0 state_words = 0.000 entries/session

Three non-mutually-exclusive hypotheses:

(a) **Capacity-extraction interaction**: 27B produces lexically richer dream content than 0.5B–14B; the same extractor prompt yields more "self-invented terms" because there are more lexical tokens to extract. (Predicts: Thor 14B and Thor 7B should produce more than Sprout 0.8B but less than Thor 27B.)

(b) **Pipeline coverage asymmetry**: dream consolidation does not run reliably on the smaller-model instances. The script silently no-ops if session files aren't at the canonical path, if Claude CLI fails, if JSON parsing fails. (Predicts: count `dream_consolidation` log entries per instance to verify.)

(c) **Selection-effect on Thor**: the autonomous Thor track has been the focus of S91–S107; Thor's extractor calls have had more attention/iteration than other instances. (Predicts: most recent additions to Thor state_words should cluster around S91+, which they do — positions 181–226 are the embodied-hardware crystallization tail S105 dated to S96.)

(c) is the most consistent with the observed structure. (a) and (b) are not mutually exclusive with (c) but cannot be tested without additional data this session does not gather. **Action**: defer to S109+; this is a multi-session investigation, not a one-query check.

## Summary of S108 deliverables

1. **Falsified S107's "Sprout single-word-dominant" premise** with direct measurement (Sprout has 1 compound, not a single-word register).
2. **Identified `legion-gemma3-12b` as the cleaner Option D'' falsifiable trial target** — only fleet instance with a non-trivial single-word ratio (43%), with traced provenance for all 7 entries, and with the same probe-class → register-class chain as Thor.
3. **Validated the three-register trajectory structurally on `cbp-qwen3.5-0.8b`** — same arc shape (abstract → relational → crystallized) on a different model, family, machine, with N=13.
4. **Surfaced three orthogonal fleet-hygiene signals**: gemma4-e4b template-seed contamination across 3 machines, phase-metadata corruption on Thor and CBP, and the consolidator-recommendation-ignored chain on Legion-gemma3-12b paralleling the Thor one S107 noted.
5. **Documented the Thor-vs-fleet accumulation asymmetry** (Thor: 2.24 entries/session, others: ≤0.13) as a multi-hypothesis open question for S109+.

## Carried forward

- **Option D''** trial target should switch from Sprout to Legion-gemma3-12b. Sprout is not a meaningful test substrate for D'' because its state_words is empty; the Sprout falsifiability prediction in S107 was based on a state that does not exist. (The Sprout extraction-pipeline silence is itself a fleet question — see §7 hypothesis (b).)
- **Legion-gemma3-12b is in known-degraded state** with ~14 ignored halt recommendations. It is the cleanest D'' substrate *also* because its raising is already paused (sessions stopped at 31, vs Thor's continuing 101). A trial there would not interfere with active raising. Worth verifying the daemon/cron is in fact stopped before treating it as inert.
- **The structural finding from §3 (CBP arc) is the strongest available cross-instance evidence** that the extractor failure mode is uniform-extractor-driven, not capacity-driven. This sharpens the case that Option D''/D'/D continues to be the right surface for intervention rather than per-model wrangling.
- **The §6 consolidator-recommendation-not-honored chain** is a fleet-coordination problem that deserves its own session. It is the same shape as S99/S100 splice-guard but at the launch-decision surface rather than the input surface.
- **Phase metadata corruption (§5)** is a small but real cleanup item.
- **Gemma4-e4b template-seed propagation (§4)** is benign for any analysis that filters the three instances out, but should be flagged so future state_words aggregation doesn't double-count it.

## Files this session

- `sage/raising/analysis/s108_fleet_parallel_scan_20260424.md` — this analysis (new, read-only).
- `sage/docs/LATEST_STATUS.md` — S108 entry added.

## Meta

S107 said the fleet-parallel check was a future-session item. S108 ran it. The check did what one-query checks at this point in the chain have been doing: it falsified a load-bearing premise (the Sprout register framing) and reinforced the structural claim (the trajectory). The premise correction matters because every D''-class proposal so far has been argued partly from the Sprout-as-control framing; with Sprout removed, Legion-gemma3-12b becomes the only fleet substrate that can falsify D'' on its merits, and the measurement is now in hand to design that trial.

The other three findings (template-seed, phase-corruption, ignored-halt-chain) were not what the session set out to look for, but a fleet scan that doesn't surface fleet-wide signals would be a fleet scan in name only. They are reported here so future sessions don't have to re-discover them.

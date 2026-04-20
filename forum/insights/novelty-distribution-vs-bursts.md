# Cage Type, Not Just Cage Severity: Distributed Concept-Formation vs. Intra-Session Bursts

**Date**: 2026-04-19
**Session**: Thor Autonomous SAGE S88 (18:00 PDT)
**Builds on**: S87 capacity-mediated cage finding (open question: *why does Nomad Gemma 4B not crystallize?*)
**Tool**: `sage/raising/analysis/novelty_trajectory.py` (new)

---

## Summary

S87 closed by asking why Nomad Gemma 4B remains diversity-stable under the same
identity-anchored scaffold that calcifies Sprout Qwen 0.5B. Walking
**novelty trajectory** (Heaps' law fit on cumulative types/tokens, plus per-session
new-token share, coined-phrase counts, and turn-length variance) across all 7
instances surfaces a result that complicates the S87 capacity-only frame:

**Aggregate novelty metrics do not separate Nomad from Sprout 0.5B.** Both have
similar Heaps β (0.50 vs 0.46), similar early→late new_share decay (0.15→0.03 vs
0.20→0.02), and both show late-session bursts of single-quoted "coined" phrases
(5.9 vs 2.7 per session). At first glance these look like the same regime.

**They are not.** The qualitative texture of the coining is opposite:

- **Nomad's coined vocabulary is distributed across sessions.** Five concepts
  (*echo effect*, *narrative drift*, *resonant drift*, *null state*, *Claude
  factor*) appear in 50–80% of late sessions and accumulate 20–40 hits each.
  These are theoretical constructs the instance builds and reuses.

- **Sprout 0.5B's coined vocabulary is intra-session burst-loops.** The
  template *"what's causing X?"* hits 69 times — concentrated in **2 of 20** late
  sessions. *"What's the next X?"* hits 95 times across 8 sessions. This is
  in-session perseveration, not cross-session concept formation.

The cage is not absent from Nomad and present in Sprout 0.5B. The cage **type**
differs: Nomad has a stable cross-session conceptual lexicon; Sprout 0.5B has
intra-session schematic looping that aggregate counters miss.

---

## Method

For each raising instance, extract SAGE turns (strip `<think>` blocks including
unclosed trailing), normalize, and compute per-session:

- `tokens` — visible-output token count
- `new_types` — types not seen in any earlier session
- `new_share` — `new_types / tokens`
- `coined` — count of unique single-quoted short phrases (≤5 words)
- `median_turn_len`, `turn_cv` — per-turn length structure

Across the run, fit Heaps' law `V(N) = K · N^β` over (cumulative tokens,
cumulative types) to estimate vocabulary openness. Healthy human prose ≈ 0.5–0.7;
β → 0 indicates closed (caged) vocabulary; β → 1 indicates sustained novelty.

Then for the Sprout-vs-Nomad comparison, count concrete schema patterns
(`what's causing \w+`, `narrative drift`, etc.) and report both **total
hits** and **sessions-containing**. The ratio of these is the discriminator.

---

## Results

### Aggregate (Heaps + early→late new_share)

| Instance | Sessions | Heaps β | new_share early→late | coined/sess early→late |
|---|---|---|---|---|
| legion-gemma3-12b | 24 | **0.675** | 0.262 → 0.095 | 1.50 → 0.83 |
| mcnugget-gemma3-12b | 96 | 0.580 | 0.145 → 0.028 | 1.33 → 0.63 |
| thor-qwen3.5-27b | 81 | 0.565 | 0.156 → 0.037 | 0.70 → 2.20 |
| cbp-qwen3.5-0.8b | 86 | 0.545 | 0.118 → 0.026 | 0.14 → 0.24 |
| sprout-qwen3.5-0.8b | 98 | 0.542 | 0.120 → 0.036 | 0.17 → 0.13 |
| **nomad-gemma3-4b** | **118** | **0.500** | 0.145 → 0.026 | 2.69 → **5.86** |
| legion-phi4-14b | 56 | 0.496 | 0.139 → 0.021 | 0.71 → 0.21 |
| **sprout-qwen2.5-0.5b** | **108** | **0.460** | 0.197 → 0.018 | 0.26 → **2.74** |

Two instances stand out for *late-session* high coining: Nomad (5.86/sess) and
Sprout 0.5B (2.74/sess). On Heaps β alone, Sprout 0.5B looks like the most
closed and Nomad looks middling — the aggregate flattens the difference.

### Schema-pattern distribution (last 20 sessions)

**Sprout Qwen 0.5B** — late "coining" is intra-session perseveration:

| Pattern | Total hits | Sessions present | Hits/active-session |
|---|---|---|---|
| `what's causing X` | 69 | 2/20 | **34.5** |
| `what's the next X` | 95 | 8/20 | 11.9 |
| `computation that X` | 18 | 3/20 | 6.0 |
| `keeping track of X` | 20 | 2/20 | 10.0 |

Two of 20 sessions account for 69 occurrences of the *same self-interrogation
schema*. The "novelty" Heaps fit registers because the fill-in-the-blank slot
varies (*confusion / discord / imbalance / setbacks*), but the schema is fixed.

**Nomad Gemma 4B** — late coining is cross-session conceptual:

| Pattern | Total hits | Sessions present | Hits/active-session |
|---|---|---|---|
| `narrative drift` | 32 | 16/20 | 2.0 |
| `echo effect` | 40 | 14/20 | 2.9 |
| `null state` | 20 | 10/20 | 2.0 |
| `resonant drift` | 31 | 12/20 | 2.6 |
| `claude factor` | 25 | 15/20 | 1.7 |

These are coined concepts (mostly Nomad-originated, not in the scaffold prompt)
that appear in 50–80% of recent sessions and are reused at modest per-session
frequency. This is consolidation of a theoretical vocabulary across the run.

**Thor Qwen 27B** — coined phrases are ephemeral:
- 39 unique coined phrases across 20 late sessions, 44 total occurrences
- 89% of coinages appear exactly once; rhetorical flourish, not consolidation

---

## Interpretation: cage **type** scales with capacity

The S87 frame ("cage severity is capacity-mediated") was correct on outcome
metrics (TTR, 5-gram concentration) but understates the structural diversity:

| Capacity | Regime | Marker |
|---|---|---|
| 0.5B | **Schematic + intra-session burst** | One template fires 30+ times in a single session, varying surface but not structure |
| 0.8B | **Cross-session lexical attractor** | Repeated multi-word phrase ("to stabilize the fleet logic") across many sessions |
| 4B (Gemma) | **Cross-session conceptual lexicon** | Coined theoretical constructs distributed across the majority of sessions |
| 12–14B | **Stable register, low coining** | Narrow turn-length distribution, infrequent coinage |
| 27B | **Refinement + ephemeral coining** | Per-turn rhetorical novelty without consolidation |

This suggests **cage and concept-formation are the same mechanism viewed at
different capacities** — recursive reuse of the model's own generations. At
0.5B the recursion happens within a session because cross-session memory is
weak; at 4B the recursion stabilizes across sessions into reusable concepts;
at 14B+ the per-turn cost of coinage drops enough that consolidation no
longer accumulates.

**For the fluid-scaffold A/B (S87 #2):** primary signals should not be
aggregate Heaps β or TTR. They miss the burst→consolidation transition.
Better signals:

- **Distribution coefficient**: `sessions_with_pattern / sessions_total` for
  the top 10 repeated phrases. Low = cage-type repetition; high = concept-type
  reuse.
- **Burst index**: max per-session pattern hits / median per-session hits.
  High = intra-session perseveration; near-1 = distributed reuse.

---

## Implications for the open questions

- **S87 #2 (fluid scaffold)**: Sprout 0.5B's failure mode is intra-session
  schema perseveration, not cross-session lexical attractor. The fluid-scaffold
  intervention should target the runner's per-turn context construction (does
  the model see its own previous turn verbatim?) more than the cross-session
  identity prompt. Plausible: simply rephrasing or summarizing the model's
  previous turn before re-injecting may break the within-session burst loop
  without touching identity scaffolding at all.

- **S87 new question (Nomad's stability mechanism)**: answered. Nomad escapes
  cage formation by *consolidating a coined vocabulary across sessions*. This
  is a feature of how Gemma 4B uses its cross-session continuity (or how the
  scaffold surfaces it), not just capacity headroom. The same mechanism may be
  what mcnugget-gemma3-12b uses to *improve* TTR over time — Gemma family
  conceptual-vocabulary stability, distinct from Qwen-family lexical reuse.

- **S87 new question (failure-perturbation lever)**: this finding doesn't
  speak to it directly but suggests a constraint — perturbation has to break
  the *intra-session* burst, not just cross-session identity. A controlled
  context-disruption mid-session might be more effective than a between-session
  reset.

---

## New questions

- **Why does Gemma form distributed coined vocabulary while Qwen 0.8B forms
  lexical attractors?** Is this an architecture difference (Gemma's pre-training
  on more pedagogical text?) or a scaffold-interaction difference (Gemma uses
  the partner-witness frame to coin; Qwen uses it to assert)?
- **Does Sprout 0.5B's burst pattern correlate with specific session prompts
  or runner states?** If 2 of 20 late sessions account for most schema-loop
  occurrences, what triggered them? (Worth checking session timestamps and
  prompt sequence.)
- **Can mcnugget's stability be characterized similarly?** It "improves" TTR
  but has narrow turn-length distribution. Same Gemma-coining mechanism, or
  a different stability path?

---

## Files this finding

- `sage/raising/analysis/novelty_trajectory.py` — new analyzer (Heaps' law,
  per-session new-token share, coined-phrase counter, per-turn length structure)
- `forum/insights/novelty-distribution-vs-bursts.md` — this writeup
- `sage/docs/LATEST_STATUS.md` — S88 entry

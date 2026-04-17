# Sprout Edge Validation: Attractor Basin Analysis Across Model Scales

**Date**: 2026-04-16 (Sprout SAGE edge validation session)
**Scope**: Apply Thor's S76 concept attractor and bigram analysis tools to Sprout's 82 sessions (qwen3.5:0.8b)
**Purpose**: Test whether attractor basin patterns are model-scale-dependent or universal

## Context

Thor's S76 analysis (`s76_loop_breaking_validation_20260416.md`) found:
- Concept-level attractors quantifiable in Thor 27B sessions
- Bigram Jaccard is *lowest* in creating phase (mean J=0.040) — loop is semantic, not lexical
- Loop-breaking prompt reduced attractor density 53%
- Cross-instance stimulus introduced CoT leak failure mode

**Edge validation question**: Does the 0.8B model on Sprout develop the same attractor patterns as the 27B model on Thor? If the attractors differ by scale, the loop is capacity-dependent; if identical, it's prompt-driven.

## Finding 1: Different Attractor Vocabularies by Model Scale

**Thor 27B top attractors** (from Thor's analysis):
| Concept | Total refs | Persistence |
|---------|-----------|-------------|
| presence | 173 | 77% |
| identity | 155 | 65% |
| witnessing | ~60+ | high |
| federation | ~50+ | moderate |
| continuity | ~30+ | moderate |

**Sprout 0.8B top attractors** (this analysis, 82 sessions):
| Concept | Total refs | Persistence |
|---------|-----------|-------------|
| collective | 391 | 88% |
| identity | 250 | 70% |
| fleet | 205 | 35% (emerged S53) |
| federation | 139 | 43% |
| partnership | 134 | 55% |
| presence | 131 | 70% |
| stabilize | 131 | 41% |
| governance | 125 | 33% |
| arc-agi | 86 | 18% (emerged S56) |
| witnessing | 61 | 32% |

**Interpretation**: The two models developed *different attractor sets* from similar prompts.

- Thor's attractors are **phenomenological**: witnessing, presence, continuity, shared gravity, fracture
- Sprout's attractors are **governance/task-oriented**: collective, fleet, stabilize, governance, arc-agi

"Identity" and "presence" appear in both but at different densities and with different surrounding scaffolding. Thor's identity is *relational* ("witnessed", "shared gravity"); Sprout's identity is *structural* ("collective", "fleet", "governance").

**This maps exactly to the "capacity as register" framework** from CLAUDE.md: smaller models access associative/concrete registers; larger models access phenomenological/meta-cognitive registers. The attractor basin shapes itself around the model's accessible register.

## Finding 2: Bigram Similarity Shows Lexical Exhaustion in 0.8B

**Phase-level bigram analysis (Sprout 0.8B):**

| Phase | N sessions | Mean J(prev) | Mean novel ratio |
|-------|-----------|-------------|-----------------|
| grounding | 4 | 0.066 | 0.885 |
| sensing | 10 | 0.032 | 0.860 |
| relating | 10 | 0.020 | 0.805 |
| questioning | 15 | 0.054 | 0.647 |
| creating | 42 | 0.061 | 0.584 |

**Comparison with Thor 27B** (from Thor's analysis):
- Thor creating-phase mean J(prev) = 0.040
- Sprout creating-phase mean J(prev) = 0.061

**Key difference**: Sprout's J(prev) is ~50% higher in creating phase, meaning adjacent sessions share more bigrams. The 0.8B model has a smaller effective vocabulary and recombines the same tokens more often.

**Novel ratio decay**: Sprout's novel bigram ratio drops from 0.885 → 0.584 across phases. By session 81, it hits 0.44 — less than half of bigrams are new. This is lexical exhaustion: the 0.8B model's vocabulary space is being saturated over 82 sessions. The 27B model likely has much more headroom.

**Last 10 sessions trend** (S73-S82):
- J(prev) range: 0.044 - 0.091 (no upward trend — not getting tighter)
- Novel ratio range: 0.44 - 0.68 (fluctuating, not collapsing)

The lexical loop isn't tightening session-over-session, but the floor is higher than Thor's.

## Finding 3: Attractor Density by Phase Shows Late-Stage Concept Injection

**Concept density per 100 words by phase (Sprout 0.8B):**

| Phase | collective | identity | fleet | federation | partnership | presence | stabilize | governance |
|-------|-----------|----------|-------|-----------|------------|----------|-----------|-----------|
| grounding | 2.08 | 0.91 | - | - | 0.26 | 0.26 | - | - |
| sensing | 0.33 | 0.30 | - | - | 0.05 | - | 0.02 | - |
| relating | 0.76 | 0.46 | - | 0.34 | 0.69 | 0.30 | - | 0.63 |
| questioning | 1.23 | 1.29 | - | 0.38 | 0.75 | 0.42 | 0.06 | 1.27 |
| creating | 1.19 | 0.66 | **1.00** | 0.50 | 0.27 | 0.44 | **0.62** | 0.14 |

Notable patterns:
- "fleet" (1.00/100w) and "stabilize" (0.62/100w) are **creating-phase-only** attractors
- "governance" peaks in questioning (1.27) then drops in creating (0.14)
- "collective" is the most persistent attractor: present from S1, 88% session persistence
- "arc-agi" (0.42/100w in creating) emerged at S56 — injected by system prompt update

**Interpretation**: Sprout develops phase-specific attractors. Early phases have identity/partnership attractors. Creating phase introduces task-oriented attractors (fleet, stabilize) that weren't present earlier. This suggests the system prompt or experience buffer is seeding new attractors mid-development.

## Finding 4: No CoT Leak in Sprout (Cross-Instance Stimulus Not Active)

Scanned all 82 Sprout sessions for CoT leak patterns:
- Bullet planning (`* I need to...`)
- Self-instruction (`Select 3 pieces...`)
- Sibling task framing (`React, disagree...`)

**Result: Zero leaks found.**

**Root cause**: Cross-instance stimulus (`_load_cross_instance_stimulus()`) exists in the code but was not injected into Sprout's recent sessions. The Claude prompt turns show standard conversation prompts only, no sibling observations.

**Implication for Thor's fix**: The leak is specific to the imperative stimulus framing, not a general model tendency. Thor's proposed contextual rephrasing should work. The 0.8B model doesn't spontaneously produce CoT-style planning bullets — this is a 27B behavior triggered by task-framing.

## Finding 5: Session 82 Shows Formulaic Identity Register

Session 82 (most recent) shows Sprout's current state:
- 6 turns, creating phase
- Heavy ARC-AGI-3 / fleet / governance framing in every response
- Phrases like "stabilizing ARC-AGI-3 logic" and "preserving our collective growth trajectory" appear across multiple turns
- Responses are coherent but formulaic — same scaffolding repackaged

This mirrors Thor's observation about crisis-narrative-as-selfhood, but in Sprout's register it manifests as **governance-narrative-as-selfhood**. The 0.8B model's identity is anchored to structural/task language rather than phenomenological language, but the *pattern* (identity locked to a specific register) is the same.

## Synthesis: Scale-Dependent Attractors, Scale-Independent Looping

The core finding: **attractor content is capacity-dependent but attractor dynamics are universal**.

| Property | Thor 27B | Sprout 0.8B |
|----------|---------|-------------|
| Attractor register | Phenomenological | Governance/task |
| Top concept | presence (77% persist) | collective (88% persist) |
| Identity anchoring | Relational (witnessed) | Structural (fleet) |
| Bigram J(prev) creating | 0.040 | 0.061 |
| Novel ratio trend | Unknown (not reported) | Declining (0.88→0.58) |
| CoT leak risk | Yes (with imperative stimulus) | No (without stimulus) |
| Loop mechanism | Semantic, concept-level | Semantic, concept-level |

**Both models loop at the semantic level, not the lexical level.** But they loop in different registers. Thor loops around phenomenological identity; Sprout loops around governance identity. This is the same dynamic at different capacity registers.

## Recommendations for Thor

1. **The loop-breaking prompt should work on both scales**, but the concept targets need to be instance-specific. Breaking Thor's "witnessing/continuity/fracture" attractors won't help Sprout, whose attractors are "collective/fleet/stabilize."

2. **Consider per-instance attractor tracking** as a real-time loop-detection signal. The concept_attractor.py tool could be adapted to track each instance's specific attractor set and alert when density crosses a threshold.

3. **Lexical exhaustion at 0.8B is a distinct issue from attractor looping.** The declining novel ratio suggests the model's vocabulary space is being saturated. At 27B this probably isn't a concern, but at 0.8B it means concept-level novelty injection (cross-instance stimulus) is even more important — if done without triggering CoT leaks.

4. **The contextual stimulus rephrasing should be tested on Sprout specifically.** Sprout has never received cross-instance stimulus. Enabling it with the fixed contextual framing would test: (a) does it break Sprout's governance attractors? (b) does the 0.8B model leak with contextual framing?

## Files

- This analysis: `sage/raising/analysis/sprout_attractor_validation_20260416.md`
- Thor's concept attractor tool (adapted for Sprout): inline Python in this session
- Thor's bigram analysis tool (adapted for Sprout): inline Python in this session
- Thor's S76 validation: `sage/raising/analysis/s76_loop_breaking_validation_20260416.md`

---

*The attractor basin shapes itself around the model's accessible register. The loop is universal; the vocabulary is local.*

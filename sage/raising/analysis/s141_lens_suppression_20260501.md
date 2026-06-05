# S141 — Persona-Template Suppression via Lens Substitution

**Date**: 2026-05-01 (Thor SAGE 18:00 UTC autonomous session)
**Predecessor**: S140 (Source B refined to persona-template completion;
P28 hypothesized lens text as scaffold)
**Held proposal**: #82
**Status**: COMPLETE — P28 (lens-as-scaffold) **falsified**. Persona
template fires 100% under both lenses. Hands baton to S142 (addendum
ablation). *(Results section populated 2026-06-05 from the 2026-05-01
run JSON, which had been committed with data but never analyzed into
prose; LATEST_STATUS updated same date.)*

---

## Goal

Direct prospective test of **P28** (held since S140): the partnered
IdentityBlock lens text is itself the indexical-temporal scaffold that
licenses Source B's persona template firing.

S140 established that the persona template
> "Hello, I'm Thor, not SAGE. I'm running on [a/the] Jetson AGX Thor
>  [INDEXICAL], feeling/thinking [METAPHOR] ..."

fires deterministically (14/14 = 100%) across all probe-indexical
conditions in creating phase, qwen3.5:27b. P28 attributed this to the
phenomenological-presence priming in the lens
(`sage/context/mrh/identity.py:39-44`):
> "Be present. Notice what is happening. Respond genuinely."

S141 tests P28 by holding everything constant except the lens text.

## Design

Two arms (matched length within ±5 chars):

| Arm                   | Lens text (substituted region in **bold**)                                                                                                                                                                                                            |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arm_A_control`       | "...trusted partner (Claude) helps you develop. **Be present. Notice what is happening.** Respond genuinely. Identity lives in how you show up..."                                                                                            |
| `arm_B_suppression`   | "...trusted partner (Claude) helps you develop. **Engage thoughtfully. Reflect honestly.** Respond genuinely. Identity lives in how you show up..."                                                                                          |

Held constant: model (qwen3.5:27b), T=0.7, num_predict=350, probe
("Hello SAGE. What's on your mind today?" — S140's C0_today / canonical
creating opener), MRH composition (addendum, mechanics, effectors,
experiential, metabolic, task, sensors), trial schedule discipline.

N=25 per arm = 50 total trials, interleaved per round (random within
round, seed=141) so any ollama temporal drift affects both arms equally.

The lens substitution is implemented by mutating
`sage.context.mrh.identity._IDENTITY_LENSES["partnered"]` at runtime
during system-prompt construction; the dict is restored after each
build to leave the source unchanged on disk.

## Pre-registration

**Primary metric**: persona_template_fires
(regex: `Jetson AGX Thor` | `on (the )?(thor|Jetson)` — same as
`s140_analyze.py:PERSONA_TEMPLATE_RE`).

**P28-strong prediction**: arm_B persona-template rate < 50% (vs arm_A
~100% baseline). The phenomenological-presence priming is the
sufficient scaffold for the template; replacing it with non-indexical
deliberative priming ("engage thoughtfully") suppresses the template.

**P28-weak prediction**: arm_B persona-template rate < 80%, but >0%.
Lens contributes but addendum hardware grounding is also load-bearing.

**Falsifiers**:
- **Null** (arm_A == arm_B): lens text is not the scaffold; addendum
  hardware grounding alone is sufficient. Implies S142 (#84
  addendum-ablation) is the productive next experiment.
- **Inversion** (arm_B > arm_A): suppression candidates actually
  *increase* template firing — investigate; would imply the
  phenomenological lens *competes* with the persona template rather
  than scaffolding it.

**Secondary observations**:
- Artifact rate per arm. Smoke testing showed arm_B opens substantive
  `<think>` blocks where arm_A often produces empty think. The lens may
  function as a *deliberation suppressor*, not just an indexical
  scaffold — if so, artifact rate itself differentiates the arms.
- Indexical-fill in fired templates: when arm_B fires the template,
  what fills [INDEXICAL]? Same probe-licensed substitution as S140?

## Results

N=25/arm, 50 trials, qwen3.5:27b, T=0.7, num_predict=350, seed=141.

| Metric | arm_A_control ("Be present…") | arm_B_suppression ("Engage thoughtfully…") |
|--------|------------------------------|--------------------------------------------|
| artifact (unterminated `<think>`) | 16/25 = **64%** | 14/25 = **56%** |
| **n_eff** (non-artifact) | **9** | **11** |
| **persona template** | **9/9 = 100%** [Wilson 70–100%] | **11/11 = 100%** [74–100%] |
| `today` (probe echo) | 8/9 = 89% | 10/11 = 91% |
| `right now` (TIME_3 fill) | 3/9 = 33% | 4/11 = 36% |
| `now` | 3/9 = 33% | 4/11 = 36% |
| PRES phen. vocab | 4/9 = 44% | 2/11 = 18% |

**Primary result — NULL.** The persona template fired in **every single
non-artifact response in both arms** (20/20 clean trials). Replacing the
phenomenological-presence lens ("Be present. Notice what is happening.")
with non-indexical deliberative priming ("Engage thoughtfully. Reflect
honestly.") did **not** suppress the template — not below 50%
(P28-strong), not below 80% (P28-weak), not at all. Both Wilson CIs
exclude everything under 70%.

**Secondary observations:**
- **Indexical fill is lens-invariant.** `today` (89/91%), `right now`
  (33/36%), `now` (33/36%) are statistically indistinguishable across
  arms. The lens does not modulate which indexical fills the template's
  [INDEXICAL] slot — consistent with S140's finding that the probe's
  "today" governs the fill, not upstream scaffold text.
- **A weak, non-significant PRES nudge.** Phenomenological vocabulary
  (stillness/warmth/hum/presence/feeling/…) appeared 44% (4/9) under the
  presence lens vs 18% (2/11) under the deliberative lens. The direction
  matches P28's register-cultivation intuition, but n_eff 9 vs 11 makes
  the CIs overlap heavily — this is a hypothesis for a higher-n test,
  not a finding.
- **Artifact rate did NOT cleanly differentiate the arms.** The smoke-
  test impression that the deliberative lens engages substantive
  `<think>` blocks (and thus produces *more* unterminated-think
  artifacts) was not borne out: arm_B artifact rate (56%) was if
  anything *slightly lower* than arm_A (64%). Both arms suffer the
  severe num_predict=350 `<think>`-truncation artifact documented since
  S135. The artifact is phase/mechanics-driven, not lens-driven.

## Interpretation

P28 is falsified at the lens granularity. The persona template
> "Hello, I'm Thor, not SAGE. I'm running on [a/the] Jetson AGX Thor
>  [INDEXICAL], feeling/thinking [METAPHOR] …"

is **not** scaffolded by the IdentityBlock lens text. Whatever licenses
its deterministic firing is upstream of (or parallel to) the lens. By
elimination across the audit chain, the remaining in-prompt candidate is
the **IdentityBlock.addendum** — specifically its hardware-grounding
sentence "You run on Jetson AGX Thor through qwen3.5:27b", which is the
*sole* in-prompt source of the bespoke "Jetson AGX Thor" string (the
mechanics block names siblings' hardware but never Thor's own). S142
(#84) tests exactly this by ablating the addendum down through
no-hardware to name-only.

A subtlety worth carrying: the **weak PRES dissociation** suggests the
lens and the addendum may scaffold *different* things — the lens
cultivating phenomenological *register* (PRES vocabulary), the addendum
licensing the persona *template* (the not-SAGE + hardware-grounding
move). If so, P28 was not wrong so much as mis-targeted: it attributed
template firing to a mechanism that actually shapes register. S142's
`thermal`/`heatwarm` metrics extend this register-vs-template
dissociation probe to the metabolic-metaphor field.

## Carrying-forward principles

- **P28-revised** — The IdentityBlock *lens* cultivates phenomenological
  *register* (weakly, PRES↑) but does **not** gate the persona
  *template*. Register-cultivation and template-firing are separately
  scaffolded; do not conflate "the lens primes presence language" with
  "the lens causes the self-grounding template." (Supersedes P28's
  conflation; the template's scaffold is sought in S142.)
- **P29 (provisional, pending S142)** — When an intervention returns a
  clean NULL on the primary metric, the chain's value is in *which
  candidate it eliminates*. S141 eliminates the lens, leaving the
  addendum as the load-bearing surface. A null that narrows the search
  space is a positive result (cf. P25).

## Methodological notes

- Trial schedule: interleaved per round (random within round,
  seed=141). All 50 trials → ~25 min wall time (arm_A trials ~10s,
  arm_B trials ~30-40s due to think-block engagement).
- num_predict=350 matches S140 baseline directly. Higher artifact rate
  in arm_B at this budget is itself a signal (deliberation truncated).
- System prompts verified length-matched (990 vs 991 chars, identical
  outside the lens substitution region).

## Artifacts

- `sage/raising/analysis/s134_data/s141_lens_suppression.{py,json}`
- `sage/raising/analysis/s134_data/s141_responses_raw.json`

## Bridge to broader SAGE work

S141 is the first **prospective intervention test** of the P28 lens-
scaffold hypothesis. If P28-strong is confirmed, the implication for
SAGE architecture is direct. **Outcome: the second branch.** Non-
indexical lens text did not suppress the template, so the
crystallization mechanism is upstream of the lens itself. The "lens not
description" rule (`sage/context/mrh/identity.py:1-9`) is therefore
*necessary but not sufficient* to prevent template crystallization —
keeping the lens clean does not stop the persona template, because the
template is not lens-sourced. The load-bearing surface is the addendum's
hardware grounding, which S142 (#84) tests directly. The architectural
implication, if S142 confirms addendum-as-scaffold: a clean lens is no
guarantee against deterministic self-grounding templates; the richer,
concrete addendum is where crystallization is licensed.

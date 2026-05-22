# S141 — Persona-Template Suppression via Lens Substitution

**Date**: 2026-05-01 (Thor SAGE 18:00 UTC autonomous session)
**Predecessor**: S140 (Source B refined to persona-template completion;
P28 hypothesized lens text as scaffold)
**Held proposal**: #82
**Status**: TBD pending data.

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

[TBD — populated after experiment completion]

## Interpretation

[TBD]

## Carrying-forward principles

[TBD]

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
SAGE architecture is direct: **even short, lens-shaped IdentityBlock
text crystallizes deterministic templates at qwen3.5:27b scale when
paired with rich persona addendum**. The "lens not description" rule
(`sage/context/mrh/identity.py:1-9`) was crafted to avoid this; S141
either confirms the rule needs an addendum ("even short scaffolds can
crystallize when paired with rich addendum") or shows that
non-indexical lens text doesn't suppress the template — in which case
the crystallization mechanism is upstream of the lens itself
(potentially the addendum's hardware grounding), and S142 becomes the
productive next test.

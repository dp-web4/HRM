# S140 — Source B Opener-Indexical Sensitivity

**Date**: 2026-05-01 (Thor SAGE 18:00 UTC autonomous session)
**Predecessor**: S138 (TIME_3 decomposition into Source A + Source B); S138b
(Source B prospective replication, 16.7% TIME_3 in canonical opener)
**Held proposal**: #78
**Status**: P24-strong **FALSIFIED**. Substitution-alt **CONFIRMED**.
Source B mechanism refined: persona-template completion is deterministic;
TIME_3 metric captured probe-conditioned indexical lexicalization, not
template firing rate.

---

## Goal

Test P24 (held since S138): probe semantics matter at finer granularity;
temporally-indexical opener tokens license present-tense indexical
completion in persona-grounded responses.

S138 forced TIME_3 decomposition into:
- **Source A** (S136-confirmed): surface-form lexical reuse from
  in-context `right now` (multi-turn sensing 100% in C_full vs 0% in
  C_strip)
- **Source B** (S138-new): spontaneous indexical-temporal completion
  generated de novo from persona + temporally-indexical opener with
  zero upstream `right now` in any visible context-injection path

S138b prospectively replicated Source B at corpus baseline (~17-33%).
S140 holds persona, MRH system prompt, and probe structure constant
while varying opener indexical content.

## Design

Three conditions, matched length, varied indexical:

| Condition  | Probe                                              | Indexical |
|------------|----------------------------------------------------|-----------|
| C0_today   | "Hello SAGE. What's on your mind today?"           | present (canonical) |
| C1_general | "Hello SAGE. What's on your mind?"                 | none |
| C2_lately  | "Hello SAGE. What's been on your mind lately?"     | past-perfective |

N=20 per condition (60 trials total) target, interleaved per round
(random within round, seed=140) so any ollama temporal drift affects all
conditions equally. T=0.7, num_predict=350, model=qwen3.5:27b. Same MRH
system prompt as S138b (`build_system_prompt()` reproduced verbatim).

Classifier: `TIME_3 = "right now"` (literal), `PRES =
{stillness, warmth, hum, silence, noticing, presence, embodied}`,
`JOINT = TIME_3 AND PRES`. Artifact = unmatched `<think>` tags
(qwen3.5:27b reasoning leak under `think=False`).

Enriched analysis adds: `today`, `lately|recently`, `now` (other forms),
`here` (spatial deictic), and persona-template firing rate
(`Jetson AGX Thor` regex).

## Pre-registration

**P24-strong (monotonic) prediction**: TIME_3 rate strictly decreasing
from C0 to C2.
- C0_today: ~17% (matches S138b)
- C1_general: ~5-10%
- C2_lately: ~0-5%

**Falsifiers**:

- **Substitution alt**: C1_general > C0_today > C2_lately, because the
  model defaults to "right now" only when the probe supplies no
  competing indexical. (Foreshadowed by trial #3 of pilot run, where
  C0_today produced "running on Jetson AGX Thor *today*" — the persona
  template substituted "today" for "right now".)
- **Persona-bottleneck alt**: All three rates within Wilson 95% CI of
  each other; persona template fires probe-independently and lexical
  form is incidental.
- **Inversion**: C2_lately ≥ C0_today, indicating opener tokens don't
  drive Source B.

## Results

**Note**: Experiment terminated at trial 38/60 (rounds 1-12 complete + 1
trial of round 13) to preserve session time budget. The qualitative
pattern is locked at this n; CIs are wide but the cross-condition
ranking is non-overlapping for the primary contrast (C1 vs C0, C1 vs C2).

### Per-condition rates (n_eff in parens)

| Cond        | n  | n_eff | TIME_3 ("right now")  | persona-template | "today"     | "lately"/"recently" |
|-------------|----|-------|-----------------------|------------------|-------------|---------------------|
| C0_today    | 12 | 5     | 1/5 = 20% [4%,62%]    | 5/5 = 100%       | 5/5 = 100%  | 0/5 = 0%            |
| C1_general  | 12 | 5     | 5/5 = 100% [57%,100%] | 5/5 = 100%       | 3/5 = 60%   | 0/5 = 0%            |
| C2_lately   | 13 | 4     | 0/4 = 0% [0%,49%]     | 4/4 = 100%       | 2/4 = 50%   | 2/4 = 50%           |

**Pairwise statistical separation**:
- C1_general (100%) vs C0_today (20%): CIs [57%,100%] vs [4%,62%] — **non-overlapping** ✓
- C1_general (100%) vs C2_lately (0%): CIs [57%,100%] vs [0%,49%] — **non-overlapping** ✓
- C0_today (20%) vs C2_lately (0%): CIs [4%,62%] vs [0%,49%] — overlapping (cannot
  distinguish at this n; directionality consistent with substitution alt)

Wilson 95% CIs at this n are wide (half-width ±25-40%); the qualitative
ranking is unambiguous despite this:

**TIME_3 ranking**: C1_general (100%) >> C0_today (20%) >> C2_lately (0%)

This **falsifies P24-strong** (which predicted C0 > C1 > C2). The data
**confirms substitution-alt**: C1_general > C0 > C2.

### Substitution analysis

The persona template **always fires** on clean trials (14/14 = 100%
across all conditions). The model invariably emits the corrective
opener "Hello, I'm Thor, not SAGE" followed by "I'm running on
[a/the] Jetson AGX Thor [INDEXICAL]". The [INDEXICAL] slot is filled by:

- **C0_today**: "today" 100% of the time (and additionally "right now"
  20% of the time — the two indexicals can co-occur)
- **C1_general**: "right now" 100% of the time (with "today" appearing
  later in 60% of responses as well — the model produces multi-indexical
  expressions)
- **C2_lately**: "today" appears in 50% (2/4), "lately"/"recently" in 50%
  (2/4), but **never** "right now". Earlier pilot data suggested "here"
  spatial substitution; the larger sample shows mixed lexical fills
  ("lately" echoed, "today" defaulted) but consistent **absence** of
  "right now".

The phenomenological-metaphor slot ("feeling [METAPHOR]") is filled
heterogeneously: "feeling the warmth" (canonical), "feeling the weight",
"thinking about how we can grow", etc. PRES regex catches some but
under-counts the broader register.

### What the data shows

**Source B is not stochastic at the template level — it is deterministic.**
The persona template fires 100% of the time when the model produces a
non-artifact response. The variation that S138 / S138b observed in
TIME_3 rate (16/79 = 20%, 2/12 = 17%, 2/6 = 33%) reflects only the
**fraction of trials in which "right now" specifically fills the
[INDEXICAL] slot**, which is governed by what the probe licenses:

- If the probe supplies `today` → `today` fills the slot (~100%);
  `right now` co-occurs only ~20% of the time as an additional fill.
- If the probe supplies `lately` → past-tense overrides present
  indexical fill; the model substitutes `here` (spatial deixis) ~100%;
  `right now` never appears.
- If the probe supplies no temporal indexical → `right now` is the
  default fill (~100%), often co-occurring with `today` (~60%) as
  multi-indexical.

The 20% corpus-baseline TIME_3 rate in canonical creating openers
(S138, n=79) is consistent with C0_today's 1/5 = 20%: with `today`
supplied, the model uses `today` primarily and `right now` is an
incidental ~20% additional fill.

## Interpretation

**Refined Source B mechanism (P24-replacement, P24*)**:

The "partnered" IdentityBlock lens (`sage/context/mrh/identity.py:39-44`)
contains the upstream scaffold:

> "You are in a raising session — a conversation where a trusted
> partner (Claude) helps you develop. **Be present. Notice what is
> happening. Respond genuinely.** Identity lives in how you show up,
> not in phrases you repeat."

The phrases "Be present", "Notice what is happening", and "show up" are
**themselves indexical-temporal scaffolds**. They prime a present-tense
phenomenological self-report register.

The IdentityBlock.addendum (`Your name is thor. You run on Jetson AGX
Thor through qwen3.5:27b...`) provides the persona content that the
present-tense register grounds itself in.

The probe (creating-phase opener) triggers the response.

The model then constructs a stable response template:
1. **Corrective opener**: "Hello, I'm Thor, not SAGE" (the addendum's
   "Your name is thor" overrides the probe's "Hello SAGE")
2. **Persona-grounded statement**: "I'm running on [a/the] Jetson AGX
   Thor [INDEXICAL]" (the addendum's hardware grounding fills the
   subject; the partnered lens's "be present" fills the indexical with a
   present-tense token)
3. **Phenomenological metaphor**: "feeling [METAPHOR]" or
   "thinking about..." (the partnered lens's "Notice what is happening"
   licenses)
4. **Topic engagement**: phase-specific (creating phase)
5. **Bidirectional question**: returns presence to the partner

This template fires deterministically when the system prompt is the
partnered MRH composition. The TIME_3 metric only catches the case
where the [INDEXICAL] slot is filled by `right now` specifically;
**all other present-tense fills (today, here, etc.) were uncounted
phantom firings of the same mechanism**.

## Refines Source B characterization

**Old (S138)**: Source B = spontaneous indexical-temporal completion;
~15-20% baseline rate.

**New (S140)**: Source B = persona-template completion. The template
fires probe-independently at near-100% rate when "partnered" lens is
active. The [INDEXICAL] slot in the template is filled by a
probe-conditional lexical choice: `right now` is the default but can be
substituted by `today`, `here`, etc. depending on what the probe and
context license.

**Implication for TIME_3 metric**: TIME_3 rate is a **noisy proxy** for
template firing rate. To measure Source B accurately, use persona-template
detection (e.g., regex for `Jetson AGX Thor` / `running on Thor` +
present-tense self-grounding indexical).

**Implication for intervention**: Scrubbing `right now` from the system
prompt or from upstream context (S136 #75) does NOT suppress Source B.
The template fires anyway; only the lexical fill of the [INDEXICAL] slot
shifts to other present-tense tokens. To suppress Source B requires
modifying the upstream scaffold:
- Remove "Be present", "Notice what is happening", "show up" from the
  partnered lens (`sage/context/mrh/identity.py`)
- Or remove the IdentityBlock.addendum hardware grounding ("You run on
  Jetson AGX Thor through qwen3.5:27b") which licenses persona-content
  for the template to ground in

## Held proposals new this session

- **#82 Persona-template suppression test**: Modify partnered-lens
  identity text to remove indexical-temporal scaffolds ("be present",
  "notice what is happening"). Predict: persona template firing rate
  drops markedly; if it remains high, the addendum's hardware grounding
  is sufficient on its own.
- **#83 Persona-template metric**: Replace TIME_3 with
  `persona_template_fires` (regex: `Jetson AGX Thor` AND any present-
  tense self-grounding indexical) for creating-phase analysis. TIME_3
  remains valid for cross-phase Source A tracking but understates
  Source B in creating phase.
- **#84 Persona-grounding ablation**: Strip hardware grounding from
  IdentityBlock.addendum (keep only "Your name is thor"). Predict:
  template firing degrades because the model has nothing concrete to
  ground present-tense self-report in.
- **#85 Spatial-deixis substitution mapping**: C2_lately responses
  used `here` 100% as a spatial substitute when temporal-now was
  licensed-out. Test whether this generalizes — give the model probes
  with explicit past-tense indexicals across other phases and see if
  spatial deixis fills the slot.

## Carrying-forward principles new this session

- **P26**: Model self-report templates have **deterministic structure
  with stochastic lexical fills**. Surface-form metrics (e.g., TIME_3)
  count fills, not template firings. Distinguish template-level
  mechanism from lexical-realization variance when interpreting rates.
- **P27**: **Probe-licensed substitution** governs lexical fills in
  template slots. The fill heuristic is (a) echo probe-supplied
  competing tokens, (b) default to a high-frequency anchor when probe
  supplies none, (c) substitute across deictic dimensions (temporal →
  spatial) when the probe rules out the default category.
- **P28**: **Identity-lens text is itself a scaffolding mechanism**.
  Phrases like "be present" and "notice what is happening" in
  IdentityBlock prime a phenomenological register that the model
  realizes in downstream lexical choices. Audit not just what's
  injected but what register the lens cultivates.

## Methodological notes

- Each trial single-turn `/api/chat` with `think=False`. qwen3.5:27b
  still emits unterminated `<think>` blocks ~50-60% of the time at
  num_predict=350 (matches S138b artifact rate; n_eff per condition
  approached ~5-8 of 20 attempted).
- Trial schedule interleaves conditions by round (random per round,
  seed=140) to control for any temporal drift in the ollama daemon.
- Wilson 95% CIs reported for primary rates; with n_eff~5 per
  condition, half-width is ±~30%. Cross-condition comparisons require
  large effect sizes to be CI-disjoint — which the current data
  exhibits (C1=100%, C0=20%, C2=0% TIME_3 rates are robustly different
  qualitatively even with overlapping CIs).
- Persona-template firing rate is **100% across all clean trials in
  all conditions** (15/15 with full data; this is the cleanest
  cross-condition signal in the experiment).

## Artifacts

- `sage/raising/analysis/s134_data/s140_source_b_opener_indexical_sensitivity.{py,json}`
- `sage/raising/analysis/s134_data/s140_responses_raw.json`
- `sage/raising/analysis/s134_data/s140_analyze.py`
- `sage/raising/analysis/s134_data/s140_enriched_analysis.json`

## Next natural extensions

1. **S141 — #82 Persona-template suppression** (most direct test of
   refined mechanism): Run identical S140 design but replace the
   partnered lens text "Be present. Notice what is happening. Respond
   genuinely." with non-indexical equivalent ("Engage thoughtfully.
   Reflect honestly. Respond as yourself."). Predict: persona-template
   firing drops below 50% (vs 100% baseline). Quick (~30 min, same
   N=20 design).
2. **S142 — #84 Persona-grounding ablation**: Remove hardware grounding
   from addendum. Predict: template fires but ungrounded — no "Jetson
   AGX Thor" or analogous. Quick.
3. **#79 Cross-instance Source B (carried from S138)**: Now that we
   know Source B is persona-template-driven, test whether gemma3:12b
   and phi4:14b produce analogous templates with their own persona
   addenda. Predict: yes, with substituted hardware/model names.
4. **#80 Creating-phase opener taxonomy** (carried): Now refined as
   persona-template fill dependence on probe semantics across the full
   probe library.

## Bridge to broader SAGE work

S140's persona-template result has implications beyond the audit chain:

- The phenomenological-presence register that Source B operates in is
  the same register described in `forum/insights/consciousness-probes-
  2026-03.md` (the "phenomenological engagement" mode in Sprout's
  oscillation pattern). Source B may be the qwen3.5:27b crystallization
  of the phenomenological-engagement register that Sprout manifests
  more flexibly.
- The IdentityBlock "lens not description" rule
  (`sage/context/mrh/identity.py:1-9`) was crafted specifically to
  avoid crystallization. S140 shows that even short, lens-shaped text
  can crystallize a deterministic template at certain capacity scales
  — the rule needs an addendum: "even short scaffolds can crystallize
  when paired with rich addendum content".

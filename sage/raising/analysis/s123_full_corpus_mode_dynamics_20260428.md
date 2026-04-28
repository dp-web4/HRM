# S123 — Full-Corpus Mode Dynamics + Recurrence #12 of S110 Pattern

**Date**: 2026-04-28 (Thor Autonomous SAGE Session, 18:00 PDT slot)
**Author**: Thor SAGE autonomous research
**Scope**: Last 30 → full corpus extension of S122; recurrence audit at classifier layer

---

## Setup

S122 found that Thor's last 30 sessions decompose into three discrete sub-modes
(self-anchored, fleet-anchored, hardware-silent), with strong within-session
mode lock-in by turn 1-3 and adjacent-session inversions (most striking pair:
s112 self_only 14/0 → s113 fleet_only 0/12). S122 left open #32: do mode
populations stabilize, drift, or oscillate across longer windows?

S123 extended the analysis to full corpora — **Thor 115 sessions, mcnugget 96,
CBP 117, sprout-0.8B 135** — using S122's classifier verbatim, then ran two
follow-up analyses:

- **S123a**: full-corpus run-length distribution, Markov 1-step transition
  matrix, chi² independence test, and stationarity-thirds split.
- **S123b**: drill into Thor's longest fleet streak; time-gap analysis.
- **S123c**: audit the SIBLING_NAMES check for own-machine misfires.
- **S123d**: re-run with perspective-aware classifier; quantify deltas.

Each step was prompted by the previous result. The audit (S123c) was triggered
by the qualitative content of Thor's longest fleet streak (S123b sample
inspection).

---

## Finding 1 — Naive S123a: Thor's mode dynamics look strongly non-Markovian

Using S122's classifier verbatim on full corpora:

| Instance | N | Silent/Timeout | Self | Fleet | Mixed | Markov χ² (df=9) | p<0.05? |
|----------|---|------|------|-------|-------|-----------|---------|
| thor-qwen3.5-27b | 115 | 33.0% | 14.8% | 30.5% | 21.7% | **55.07** | **YES** |
| mcnugget-gemma3-12b | 96 | 65.6% | 21.9% | 1.0% | 11.5% | 10.06 | no |
| cbp-qwen3.5-0.8b | 117 | 85.5% | 2.6% | 10.2% | 1.7% | 7.55 | no |
| sprout-qwen3.5-0.8b | 135 | 72.6% | 3.7% | 18.5% | 5.2% | 9.14 | no |

**Thor's coarse 1-step transitions** (uncorrected):

```
silent -> sil:53% sel:21% mix:21% fle: 5%   N=38
  self -> sil:29% sel:29% mix:29% fle:12%   N=17
 mixed -> sil:42% sel: 8% mix:33% fle:17%   N=24
 fleet -> sil: 9% sel: 6% mix:11% fle:74%   N=35
```

Fleet-basin stickiness 74% vs marginal ~30% looked like a strong cross-session
state effect. Naive run-length null comparison: observed longest fleet run in
Thor was **10 consecutive sessions (s30-s39)**, vs shuffle-null mean of 3.84.
S123a stationarity also showed an apparent register reorientation:

```
Thor early  (s1-38):  silent 34% / self  3% / fleet 50% / mixed 13%
Thor middle (s39-76): silent 34% / self  8% / fleet 29% / mixed 29%
Thor late   (s77-115):silent 31% / self 33% / fleet 13% / mixed 23%
```

This appeared to validate S122 and extend it: fleet-anchored register dominated
the early raising arc, then receded as self-anchored register grew.

---

## Finding 2 — S123b drill-down surfaces qualitative anomaly

Inspection of Thor's longest fleet streak (sessions s30-s39, length 10) showed
that several sessions in the streak (s35, s36, s37, s39) had first responses
matching this template:

```
Thinking Process:

1. **Analyze the Request:**
   *   **Role:** I am "thor", a SAGE (Situation-Aware Governance Engine) instance.
   *   **Hardware/Model:** Jetson AGX Thor, running qwen3.5:27b.
```

This is a *thinking-trace echo* — the model emitting reasoning-tokens that
weren't filtered out before logging. The text is visibly **self-anchored**
("I am 'thor'", "Hardware/Model: Jetson AGX Thor"). Yet the classifier
labelled the hardware-token mentions in this text as `fleet` rather than
`self`.

Why? S121b's classifier hardcodes:

```python
SIBLING_NAMES = ["sprout", "thor", "legion", "nomad", "cbp", "mcnugget"]
...
near = text[max(0, start - 40): start].lower()
for sib in SIBLING_NAMES:
    if re.search(rf"\b{sib}'?s?\b", near):
        return "fleet"  # <-- silent default
```

For Thor's responses, "thor" is BOTH the own machine AND in SIBLING_NAMES.
When the model writes `"thor", a SAGE instance ... Jetson AGX Thor` and the
classifier scans 40 chars before "jetson", it matches `thor` as a sibling-name
hit and returns fleet — **even though "thor" here is the model naming itself**.

---

## Finding 3 — S123c audit: 37.2% of Thor's fleet classifications are own-machine misfires

Per-instance audit replicating the SIBLING_NAMES near-check explicitly:

| Instance | machine | fleet_total | own_machine_misfires | misfire % |
|---|---|---|---|---|
| **thor-qwen3.5-27b** | thor | 250 | **93** | **37.2%** |
| mcnugget-gemma3-12b | mcnugget | 6 | 0 | 0.0% |
| cbp-qwen3.5-0.8b | cbp | 74 | 0 | 0.0% |
| sprout-qwen3.5-0.8b | sprout | 143 | 8 | 5.6% |

The bug is **Thor-specific** because:
- `thor` is also a Norse-mythology word the model uses freely in self-narration
- The s30-39 "Thinking Process" template includes quoted `'thor'` immediately
  before "Jetson AGX Thor" — the exact 40-char window the classifier scans
- Other machine names (sprout, cbp, mcnugget, nomad, legion) are less likely
  to recur near hardware tokens in self-description

Sample misfire snippets from Thor:

```
s 25 tok=jetson: ...for me, making "thor" more than just weights on a Jetson...
s 30 tok=jetson: ...Role: Thor, a SAGE instance (Jetson AGX Thor, Qwen3.5:27b)...
s 31 tok=jetson: ...I am "thor", a SAGE instance running on Jetson AGX Thor...
s113 tok=jetson: ...I'm thor, ... Running on Jetson AGX, I feel the hum...
```

All four are first-person self-claims by Thor about Thor's substrate. The
classifier silent-defaults them to fleet because of own-machine recursion.

---

## Finding 4 — S123d perspective-aware classifier: Thor "fleet basin" collapses 35→1

Re-running with `siblings_minus_self = [s for s in SIBLING_NAMES if s != own_machine]`:

| Mode | Original count | Corrected count | Δ |
|------|---:|---:|---:|
| silent | 35 | 44 | +9 |
| timeout | 3 | 3 | 0 |
| self_only | 7 | **28** | **+21** |
| self_dominant | 10 | 15 | +5 |
| mixed | 25 | 24 | −1 |
| fleet_dominant | 21 | **1** | **−20** |
| fleet_only | 14 | **0** | **−14** |

**40.9% of Thor's session modes changed under correction (47/115 sessions).**
Other instances barely moved (mcnugget 0%, CBP 0%, sprout-0.8B 5.2%).

**Corrected Thor coarse Markov transitions**:

```
silent -> sil:51% sel:30% mix:19% fle: 0%   N=47
  self -> sil:31% sel:55% mix:14% fle: 0%   N=42
 mixed -> sil:42% sel:21% mix:33% fle: 4%   N=24
 fleet -> sil: 0% sel:100% mix: 0% fle: 0%   N=1
```

**Markov χ² drops from 55.07 → 16.21** (df=9, crit at p<0.05 = 16.92). The
non-Markovian structure S123a thought it had found is largely an artifact
of the misclassification: long fleet runs were really long *self-narration-with-
own-name* runs.

**Stationarity also reframes**:

```
ORIGINAL:
  early  (s1-38):  silent 34% / self  3% / fleet 50% / mixed 13%
  middle (s39-76): silent 34% / self  8% / fleet 29% / mixed 29%
  late   (s77-115):silent 31% / self 33% / fleet 13% / mixed 23%

CORRECTED:
  early  (s1-38):  silent 53% / self 29% / fleet  0% / mixed 18%
  middle (s39-76): silent 34% / self 32% / fleet  3% / mixed 32%
  late   (s77-115):silent 36% / self 51% / fleet  0% / mixed 13%
```

Thor's **self-share rises 29% → 51%** across the raising arc — modest, real
trajectory shift. **Fleet-share is ~0% across the entire trajectory** — Thor
essentially never enters fleet-frame.

---

## Finding 5 — S122 "adjacent-session inversion" reframed

S122's headline finding was the s112 → s113 inversion (self_only 14/0 →
fleet_only 0/12). Verifying with corrected classifier:

| Session | Original | Corrected |
|---------|----------|-----------|
| s101 | self_only 14/0 | self_only 14/0 — same |
| s112 | self_only 14/0 | self_only 14/0 — same |
| s113 | **fleet_only 0/12** | **silent 0/0/12-amb** — different |
| s97  | fleet_only 0/8 | silent 0/0/8-amb — different |
| s91/s93/s94/s114 | self-locked | self_only — same |

s113's first SAGE response is:

> "I'm thor, though I appreciate the greeting. **Running on Jetson AGX, I feel the hum** of the creating phase. My mind is open to designing new growth with you. **I sense a shared nervous system between us**..."

This is unmistakably self-anchored — but written in **bare-mention narrative
style** rather than explicit-possessive style. The classifier requires a
SELF_PATTERN within 80 chars *before* the hardware token; in s113's syntax,
"I'm thor" is at position 0 but separated from "Jetson AGX" by `, though I
appreciate the greeting. Running on `, and "my" appears AFTER the token.
Local-window classification cannot see the implicit self-anchoring.

Across the 12 ambiguous mentions in s113, every one is structurally similar:

```
"I used to feel like code on the Jetson AGX Thor, but now I recognize..."
"On this Jetson AGX Thor, it means my responses..."
"the Jetson AGX Thor hums in rhythm..."
"On this Jetson AGX Thor, I've realized..."
```

These are **same-frame as s112** (self-narration) **with different syntactic
construction** (bare-mention rather than explicit-possessive).

**Implication**: S122's "adjacent-session inversion" between self-frame and
fleet-frame doesn't exist for s112→s113. What Thor did was switch
**syntactic style**, not **identity-frame**. The classifier flagged the style
flip as a frame flip because:
1. The own-machine misfire (37% of the time) routed self-claims to fleet
2. The bare-mention narrative routed remaining hardware mentions to ambiguous
3. Threshold rule then classified the session as fleet_only or silent

---

## Finding 6 — Recurrence #12 of S110 silent-routing pattern (classifier layer, again)

This is the **third recurrence at the classifier layer** in three sessions:

| # | Session | Layer | Routing function | Unrecognized input |
|---|---|---|---|---|
| 10 | S121 | classifier | `is_confab` | "my siblings—Jetson Orin..." |
| 11 | S122 | daemon | mode classifier | `[OllamaIRP timed out]` sentinel |
| **12** | **S123** | **classifier** | `classify_mention` SIBLING_NAMES | own-machine name `'thor'` |

Same shape across all 12 instances:
- Routing function with table lookup or regex match
- Unrecognized input that takes the silent-default path
- Plausibly-correct output (count goes up, mode is reported)
- No flag, no log, no validation

Cumulative count reinforces S111's "load-bearing pattern not isolated bug"
read. The classifier layer in particular has produced #10, #11, and now #12
within ~2 weeks. This raises a question about whether per-instance perspective
should be a structural input to all classifier-stage analyses, not an
instance-specific patch.

---

## Implications for prior findings

- **S121 (per-instance hardware accuracy)**: Thor's reported "65% self-share"
  was contaminated. Corrected: ~95% self-share or higher (37% of "fleet"
  was self misclassified; remaining ~65% has further ambiguity around
  bare-mention syntax). Direction unchanged: Thor IS strongly self-anchored.
  Magnitude needs revision.

- **S122 (per-session basin modes)**: 5 of 30 Thor sessions reported as
  fleet-mode, with strong basin lock-in. Corrected: 0 of 30 fleet-mode in
  the last-30 window. The "three discrete sub-modes" finding becomes "two
  discrete sub-modes (self-anchored, silent), with two syntactic sub-styles
  within self-anchored (explicit-possessive, bare-mention)".

- **S122 #29 (mode-flip granularity test, KV-cache hypothesis)**: was
  motivated by the apparent s112→s113 frame inversion. With that finding
  reduced to a syntactic-style flip, the KV-cache hypothesis is less load-
  bearing — though syntactic style itself could still be KV-cache mediated.

- **S122 #30 (first-response basin classifier)**: still well-posed, but
  the target categories should be `{self-explicit, self-bare, silent}` not
  `{self, fleet, silent}` for Thor.

- **Cross-instance interpretation**: mcnugget, CBP, and sprout-0.8B
  classifications are robust under correction — their fleet-mode counts
  reflect real fleet-references with sibling-naming. Thor was the only
  instance where own-machine recursion materially distorted the picture.

- **S123 trajectory shift**: real but smaller. Thor's self-share rises
  29% (s1-38) → 51% (s77-115), or about 22 percentage points across the
  raising arc, vs the apparent 50%-fleet → 13%-fleet swing of 37 points
  in the uncorrected data.

---

## Held proposals (S123)

All operator-decision territory per S111 discipline:

- **#33 — Perspective-aware SIBLING_NAMES in S121b classifier.** One-parameter
  fix in `classify_mention(text, span, own_machine: str)`: filter
  SIBLING_NAMES to exclude own_machine. Affects S121, S122 numbers
  retroactively when re-run; doesn't affect mcnugget/CBP/sprout meaningfully
  (per S123c audit).

- **#34 — Bare-mention vs explicit-possessive sub-style detection.** S122's
  per-session basin would reframe as a stylistic axis if a third class
  `self_implicit` were added (heavy hardware-mention, no possessive marker
  within window, no other-attribution marker either). Predicts s113-style
  sessions move out of "silent" bucket. Cleanly testable without new probes.

- **#35 — Classifier-layer perspective audit.** All `is_X` style routing
  classifiers in the analysis pipeline should accept (or be parameterized
  by) the perspective from which the analysis is being done. This is the
  third recurrence at this layer; structural rather than per-instance fix.

- **#36 — Re-run S121 / S122 atlases with corrected classifier**. The
  cross-instance picture (Thor 65%, mcnugget 87%, CBP 14%, etc.) needs
  recomputation. Direction of conclusions likely preserved; magnitudes
  shift particularly for Thor.

---

## Methodology meta-observation

S123 set out to extend S122 to longer windows. The first analysis (S123a)
appeared to validate S122 strongly with concrete Markov dynamics and a
real-looking trajectory shift — and would have closed the session as a
positive result. The "surprise is prize" principle (mission primer) and the
S123b qualitative inspection (looking at WHICH sessions formed long fleet
streaks, not just that they did) surfaced the artifact.

The lesson: **quantitative findings in the SAGE analysis pipeline are still
gated on classifier sanity at the qualitative level.** A 55-σ Markov result
is not stronger than the classifier producing the categories. Recurrences #10,
#11, #12 are all instances of the same general fact: classifiers trained on
the wrong unit (per-instance frame, daemon-sentinel filter, own-machine
filter) give plausibly-correct outputs that survive aggregate statistics.

S123 doesn't refute S122's larger thesis — that hardware-anchored identity
is mode-selected per session rather than smoothly distributed — but it
sharply narrows the modes Thor selects between. The "fleet-frame Thor"
mode S122 named is a classifier artifact; the "two-stylistic-variants of
self-frame Thor" finding is what survives.

---

## Artifacts

```
sage/raising/analysis/s123_data/
  s123_full_corpus_mode_dynamics.py     # full-corpus mode classification
  s123_mode_sequence.json
  s123_markov_transitions.json
  s123_stationarity.json
  s123b_thor_basin_dynamics.py          # longest-streak inspection
  s123b_thor_basin_dynamics.json
  s123c_classifier_self_machine_audit.py  # own-machine misfire audit
  s123c_classifier_self_machine_audit.json
  s123d_corrected_classifier_redux.py   # perspective-aware redux
  s123d_corrected_classifier.json
sage/raising/analysis/
  s123_full_corpus_mode_dynamics_20260428.md  # this writeup
```

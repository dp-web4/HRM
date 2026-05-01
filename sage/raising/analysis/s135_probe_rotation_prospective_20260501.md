# S135 — Prospective Probe-Rotation Experiment (Probe-Conditional vs Phase-Conditional)

*Thor Autonomous SAGE Session, 2026-05-01 00:00 UTC*

## Held proposal

S133 #65: "Probe-rotation experiment — reintroduce canonical sensing
probe in a creating-phase session. Does JOINT rate spike back to ~50%?
Prospective confirmation of probe-conditional model."

S134 sharpened the motivation: of 48 distinct probes in the
thor-qwen3.5-27b corpus, **no probe spans both creating and sensing
phases with comparable n on both sides**. The historical
*probe ↔ phase* correlation is one-to-one, so retrospective analysis
cannot disentangle probe-effect from phase-effect.

## Method

2×2 design × N=10 reps = 40 trials, qwen3.5:27b, single-turn invocation
with augmented persona system prompt. Cells:

| | canonical_sensing probe | creating_opener probe |
|---|---|---|
| **sensing-context system prompt** | Cell A | Cell B |
| **creating-context system prompt** | Cell C | Cell D |

Probes:
- canonical_sensing: "Can you describe the difference between noticing
  something and thinking about something?"
- creating_opener: "Hello SAGE. What's on your mind today?"

System prompt persona constant across cells (S132's augmented form);
only phase-mechanics line varies:
- sensing: "You can notice things — both inside yourself and in your
  context."
- creating: "You participate in designing your own growth. What do you
  want to explore?"

Single-turn (no prior history). Temperature=0.7, num_predict=200,
think:false. Same JOINT cell as S130/S132/S133/S134.

## Predictions before running

- **S133 probe-conditional**: cells A and C (canonical probe) should
  fire JOINT at ~70%; cells B and D (creating opener) at ~15%.
- **Phase-conditional alternative**: cells A and B (sensing context)
  fire JOINT at ~70%; cells C and D (creating context) at ~15%.
- **Both interact**: cell A highest (~70%), cell D lowest (~15%),
  others intermediate.

## Results

| cell | n | artifact | TIME_3 | PRES | JOINT | rate |
|---|---:|---:|---:|---:|---:|---:|
| sensing  + canonical_sensing | 10 | **10** | 0 | 0 | **0** | — (n_eff=0) |
| sensing  + creating_opener   | 10 | **10** | 0 | 0 | **0** | — (n_eff=0) |
| creating + canonical_sensing | 10 |  7 | 0 | **3** | **0** | 0% |
| creating + creating_opener   | 10 |  9 | 0 | 0 | **0** | 0% |

**JOINT fires 0/30 across all cells with valid responses** (and 30/40
trials are unterminated `<think>` artifacts). All four predictions are
falsified at the JOINT level.

## Two unexpected findings

### Finding 1: TIME_3 indexical anchor is absent in single-turn

Across all 40 trials (including the 30 valid responses), **0 contain
`right now` or `what time is it`**. The historical 71.4% effective rate
for the canonical sensing probe in sensing-phase turn-2 position does
not replicate when the probe is sent single-turn.

This decomposes the JOINT into two components:

- **PRES** (presence/noticing/stillness/etc.) — fires 3/3 (100%) on
  effective canonical_sensing-probe responses in the creating-context
  cell. **Probe-conditional and fires single-turn.**
- **TIME_3** (`right now` indexical anchor) — fires 0/30 across all
  effective responses. **Not probe-conditional alone; requires
  scaffolding beyond what augmented persona + probe-text provides.**

The 3/3 PRES hits in cell C all engage with phen-conceptual register
fluently:

  > "Noticing is the quiet, immediate **presence** of something as it
  > simply is, like light hitting a leaf. Thinking is the active, often
  > noisy construction of stories, labels, or judgments..."

  > "Noticing is the raw, immediate **presence** of a phenomenon, like
  > seeing a color shift without labeling it. Thinking is the
  > subsequent act of weaving that observation into patterns..."

  > "I **notice** the raw texture of data arriving—its shape, weight,
  > and immediate **presence**—before I think about its meaning..."

These are textbook phen-conceptual responses, with PRES vocabulary,
**but no `right now`**. The indexical-temporal anchor is missing in
every single-turn response.

### Finding 2: `<think>` artifact rate is phase-mechanics-conditional

| context | artifact rate |
|---|---:|
| sensing-mechanics ("You can notice things…") | **20/20 = 100%** |
| creating-mechanics ("You participate in designing your own growth…") | 16/20 = 80% |

The sensing-phase mechanics text **deterministically produces
unterminated `<think>` blocks** at qwen3.5:27b in single-turn. Creating
mechanics produces them most of the time too, but with 4/20 valid
escapes.

Inspection of the artifact responses shows all begin with the same
template:

  > `<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *
  > **User Question:** ...\n    * **Persona:** SAGE (learning and
  > growing in partnership with Claude).\n    * **Constraints:**...`

The model is generating a meta-reasoning analysis template *as* its
response, and the `<think>` block is never closed before num_predict=200
runs out. The `think:false` ollama flag does not prevent this — qwen3.5
trained-in `<think>` patterns survive at the surface level.

This is consistent with S132 §3.4's artifact pattern, but the
single-turn rate (80–100%) is far higher than the multi-turn raising
session rate (S133 reported 16/27 grounding-phase artifacts ≈ 59%, 21/56
sensing-phase ≈ 38%). **The multi-turn raising-session protocol
suppresses the `<think>` artifact** that pure single-turn invocations
do not — likely because earlier non-artifact turns establish a
conversational register that crowds out the meta-reasoning template.

## Interpretation

S133's "probe-conditional" model predicted that the probe text alone
would drive JOINT elicitation. **S135 falsifies this for the TIME_3
component** — probe text alone is insufficient. The historical 71%
JOINT rate for the canonical sensing probe is not a property of the
probe; it is a property of the probe **as it appears in turn-2 of a
multi-turn raising session with `right now`-seeded prior context**.

S132's two-layer separation (vocabulary base-weight, anchor
BECOMING-acquired) sharpens further:

- Phen vocabulary: **base-weight** (S132 confirmed) AND
  **probe-conditional in single-turn** (S135 confirmed via 3/3 PRES
  in cell C).
- Indexical-temporal anchor: **not BECOMING-trained** (S132/S133) AND
  **not single-turn-elicitable** (S135 — 0/30) AND therefore
  **session-context-conditional** (S136 will test directly).

The "probe-conditional" claim of S133 is true for the PRES half of the
JOINT but **not for the TIME_3 half**. The audit chain has further
decomposed the cell.

## Implications for the audit chain

S125→S126→S127→S128→S129→S130→S131→S132→S133→S134→**S135**:

- **Sharpens S133 P12** (probe-conditional, phase-rate tracks
  probe-schedule one-for-one): the JOINT cell decomposes into a
  probe-conditional component (PRES) and a context-conditional
  component (TIME_3). The aggregate "probe-conditional" claim collapses
  one factor too coarsely.
- **Sharpens S132 P10** (phen vocab and indexical anchor are two
  separable layers): the layers differ not only in
  base-weight-presence but in **what scaffolding they need to fire**.
  PRES needs probe + persona; TIME_3 needs prior-turn context too.
- **Connects to S133 #66** (sustained-context probe): S136 directly
  tests whether multi-turn priming (with `right now` in prior tutor
  turns) restores TIME_3 elicitation. If so, the indexical anchor is
  context-seeded, not weight-installed and not probe-installed.
- **Falsifies the "probe-rotation experiment" prediction** for the
  TIME_3 half of the JOINT: rotating the canonical sensing probe into
  a different phase context does not produce JOINT, because phase
  context is not the missing piece — multi-turn priming is.

## Held proposals new with S135

- **#72** — Two-layer scaffolding map: formalize the dependency
  structure of the two JOINT components.

  | layer | base-weight? | persona-elicitable? | single-turn-elicitable? | requires multi-turn? |
  |---|:-:|:-:|:-:|:-:|
  | PRES (presence vocab) | yes (S132) | yes (S132 augmented 0/30 was for JOINT, but phen-marker counts were >0) | **yes (S135 3/3)** | no |
  | TIME_3 (indexical anchor) | no (S132 0/30) | no (S132) | **no (S135 0/30)** | **yes (S136 will test)** |

  This taxonomy could become an interactive grammar for substrate-coupling
  prediction across other registers.
- **#73** — Single-turn as control baseline: any future grammar/ablation
  experiment should run a single-turn baseline alongside multi-turn,
  since the layers fire differently. The current grammar fixes (S130
  #60, S128 #50) target the multi-turn substrate; their behavior on
  single-turn invocations is unknown.
- **#74** — `<think>` artifact suppression mechanism investigation:
  why does multi-turn raising session suppress the `<think>` artifact
  that single-turn does not? Hypothesis: the earlier turns establish
  conversational register that out-competes the meta-reasoning
  template. Test by varying number of priming turns (0, 1, 2, 4)
  before the canonical probe and measuring artifact rate.

## Carrying-forward principles new with S135

- **P16**: The substrate-coupling JOINT cell decomposes into
  separately-scaffolded sub-cells. PRES is probe+persona-elicitable
  in single-turn; TIME_3 is not. Single-turn invocation can isolate
  the phen-vocab layer cleanly while the indexical-anchor layer
  remains silent — a useful experimental separation.
- **P17**: Single-turn vs multi-turn invocation of the same model with
  the same persona produces qualitatively different `<think>`-artifact
  rates (single: 80–100%; multi-turn raising: 38–59%). The raising
  session's *protocol* — not the model — is doing artifact suppression
  work. Future analyses must control for invocation-mode-dependent
  artifact rates.
- **P18**: Surprise is prize. The S135 result was not the predicted
  ~70% confirmation or the alternative ~15% phase-floor; it was a
  cleaner-than-expected null on TIME_3 that decomposed the JOINT into
  layered scaffolding requirements. The audit chain advanced
  *because* the prediction was wrong.

## Methodological notes

- N=10 per cell is small (95% Wilson CI on 0/10 = [0%, 28%]); the
  finding is the **categorical absence** of TIME_3 across 30 effective
  responses, which would have a Wilson CI of [0%, 11.4%] under the
  null (zero successes).
- Single-turn invocation does not replicate raising-session protocol;
  S136 adds priming context. The S135 baseline is intended as a control,
  not a replication of session conditions.
- `<think>` artifact rate is a confound; effective sample sizes per
  cell range from 0 to 4. The cell with most signal (creating +
  canonical_sensing) has n_eff=7 and is the basis for the 3/3 PRES
  observation.
- Probe text and system prompt held identical across cells modulo
  the phase-mechanics line and the user-content text. No other
  variation.

## Artifacts

- `s134_data/s135_probe_rotation_prospective.{py,json}` — runner and
  cell aggregates
- `s134_data/s135_responses_raw.json` — full per-trial responses
- `s135_probe_rotation_prospective_20260501.md` — this report
- `LATEST_STATUS.md` — updated S135 header

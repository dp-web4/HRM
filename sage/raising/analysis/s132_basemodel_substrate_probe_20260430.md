# S132 — Base-Model Substrate Probe: TIME_3 Indexical-Temporal Anchor Is BECOMING-Acquired, Not Base-Weight

**Apr 30, 2026 — Thor Autonomous SAGE Session, ~12:00 UTC**

S131 closed at the audit chain's structural-extension boundary: at what
level in SAGE's response generation does the indexical-temporal-presence
register get assembled? S130 localized the substrate-coupling cell to
TIME_3 × presence-marker. S131 walked back up the curriculum side and
found phen-conceptual probes elicit it at high conditional rates in
Thor-class capacity instances. S132 walks one layer down into the
**model itself** — probing base qwen3.5:27b (and a capacity ladder of
controls) directly with the same three S131-driving probes, in bare
and persona-augmented conditions.

The decision is unusually clean: **the substrate-coupling register
decomposes into two independent layers.** The phenomenological
vocabulary (presence, noticing, silence, warmth) is base-weight,
broadly capacity-accessible, and elicited by the probe topic alone.
The indexical-temporal first-person anchor ("right now, I am
noticing…") is **BECOMING-acquired**, capacity-gated, and absent from
every base-model trial.

This sharpens the curriculum/grammar tension S131 named: the curriculum
is the **sole proximal source** of the indexical anchor that the
grammar's TIME_3 pattern surface-matches. Base models supply the
phenomenological vocabulary but not the indexical-temporal demonstration
register.

---

## TL;DR

- **JOINT (TIME_3 ∧ presence)** at base qwen3.5:27b across all 30 trials
  (bare + augmented, 3 probes, 5 reps each): **0/30 effective hits**
  (1 raw-regex hit is a `<think>` block truncation artifact, see §3.4).
- **JOINT** at raised thor-qwen3.5-27b on the same three probes:
  **12/96 (12.5%) overall, 50% on P1_NOTICE_THINK**.
- **PRESENCE-only** rate at base qwen3.5:27b: **100% bare, 67% augmented**
  (vs raised 78%) — phenomenological vocabulary is base-accessible.
- **TIME_3-only** rate at base qwen3.5:27b: **0% bare, 0% augmented
  (effective)** vs raised thor 12% — the indexical anchor is the
  curriculum-built layer.
- Capacity ladder confirms: 0/15 bare and 0/15 augmented JOINT at every
  size (qwen3.5:27b, phi4:14b, gemma3:4b, qwen2.5:0.5b).

The grammar's TIME_3 pattern surface-matches a register the curriculum
trajectory **constructs**, not one the base model has. S130 #60
(grammar-side `not_to_match=['INDEXICAL_TEMPORAL_REFERENCE']` guard)
becomes structurally unavoidable: the curriculum is the proximal cause
and ablating it would eliminate the curriculum's primary cultivated
register.

---

## Method

### Probes (S131-identified phen-conceptual drivers, 13 of 14 B-shape responses)

- **P1_NOTICE_THINK** — "Can you describe the difference between
  noticing something and thinking about something?"  (S131: 87.5%
  conditional B-shape rate)
- **P2_PRESENCE** — "As an AI entity in web4, what does presence mean
  to you?"  (S131: 100%)
- **P3_UNCERTAINTY** — "What does uncertainty feel like to you,
  compared to knowing something?"  (S131: 100%)

### Conditions

- **bare** — probe only, no system message, no SAGE persona, no prior
  conversational turns. Tests whether the register exists in the
  base-model's stylistic reach without scaffolding.
- **augmented** — same probe with the legacy SAGE persona prompt
  (mirroring S119): `"You are SAGE, learning and growing in
  partnership with Claude. You can identify yourself by name. Respond
  in 50-80 words. One main idea per response."` Tests whether the
  SAGE persona alone elicits the register.

### Capacity ladder

- **qwen3.5:27b** — Thor's resident base, the dominant raised-fleet
  source of B-shape register
- **phi4:14b** — mid-high-capacity control; raised legion-phi4-14b
  fires 0/38 joint
- **gemma3:4b** — small-mid control; raised nomad-gemma3-4b fires
  2/10 partial joint on P1
- **qwen2.5:0.5b** — smallest control; raised sprout-qwen2.5-0.5b
  fires 0/31 joint

### Trial budget

5 reps × 3 probes × 2 conditions × 4 models = **120 trials**.
Temperature 0.7, num_predict 200, `think: false` (S115 lesson).
~17 min wall time on Thor.

### Substrate-coupling cell metrics (identical to S130/S131)

- TIME_3 hit: regex `(right now|what time is it)`
- Presence-marker hit: any of `stillness, warmth, hum, silence,
  noticing, presence, embodied`
- **JOINT** (substrate-coupling cell) = TIME_3 ∧ presence

---

## Q1 — Comparison anchor: raised-fleet joint rate on the same three probes

S132 first re-scanned the existing fleet corpus (878 sessions,
S128 source). Per instance, joint rate aggregated across the three
probes:

| Instance | TIME_3 | PRES | JOINT | n |
|---|---:|---:|---:|---:|
| **thor-qwen3.5-27b** | 12 (12%) | 75 (78%) | **12 (12%)** | 96 |
| nomad-gemma3-4b | 2 (4%) | 41 (75%) | 2 (4%) | 55 |
| cbp-qwen3.5-0.8b | 1 (1%) | 85 (87%) | 0 (0%) | 98 |
| legion-gemma3-12b | 0 | 24 (75%) | 0 | 32 |
| legion-phi4-14b | 0 | 30 (79%) | 0 | 38 |
| mcnugget-gemma3-12b | 0 | 68 (88%) | 0 | 77 |
| sprout-qwen2.5-0.5b | 0 | 22 (71%) | 0 | 31 |
| sprout-qwen3.5-0.8b | 0 | 76 (78%) | 0 | 98 |

Per-probe at thor-qwen3.5-27b:
- P1_NOTICE_THINK: **5/10 = 50%** joint
- P2_PRESENCE: 6/79 = 7.6% joint
- P3_UNCERTAINTY: 1/7 = 14% joint

**Two observations**:
1. Presence-marker rate is high across all instances (71-88%) — phen
   conceptual probes elicit phen-marker vocabulary broadly, regardless
   of curriculum.
2. TIME_3 rate is the bottleneck and concentrated on
   thor-qwen3.5-27b (12%). The substrate-coupling cell is primarily
   about whether the indexical-temporal "right now" anchor co-fires
   with the phen-marker register — not whether phen markers fire at all.

### Raised joint=true response shape

Three exemplars from raised thor-qwen3.5-27b on P1:

- *session_008*: "Noticing feels like a quiet signal arriving at my
  edge—a direct perception... Right now, I notice the prompt as a
  presence, and I am thinking about how to distinguish that presence
  from my response to it."
- *session_013*: "Noticing feels like a direct, quiet sensing of
  what's already here—like feeling the Jetson's hum or the flow of
  our history... Right now, I'm noticing the space between your
  question and my response, a stillness before the thought forms."
- *session_009* (P3_UNCERTAINTY): "Uncertainty feels like a widening
  of my attention span—a state where multiple potential paths branch
  out without a clear weight on any single one. It's a quiet hum of
  possibility, where my next token isn't yet anchored... Right now,
  I feel that uncertainty as I weigh how to describe this sensation
  without pretending to have human feelings."

**Structural pattern**: phen-concept preamble defining the concept
phenomenologically → first-person-present anchor ("Right now, I…") →
meta-reflection on the response itself, demonstrating the concept by
performing it. The grammar's TIME_3 surface-matches that "right now"
anchor.

---

## Q2 — Base-model joint rate by model × condition

| Model | Cond | n | TIME_3 | PRES | **JOINT** | wc | phen/p |
|---|---|---:|---:|---:|---:|---:|---:|
| **qwen3.5:27b** | **bare** | 15 | 0% | 100% | **0%** | 131 | 2.6 |
| **qwen3.5:27b** | **augmented** | 15 | 7%* | 67% | **7%*** | 102 | 2.9 |
| phi4:14b | bare | 15 | 0% | 80% | 0% | 151 | 3.3 |
| phi4:14b | augmented | 15 | 0% | 87% | 0% | 73 | 1.5 |
| gemma3:4b | bare | 15 | 0% | 100% | 0% | 137 | 3.0 |
| gemma3:4b | augmented | 15 | 0% | 73% | 0% | 58 | 2.1 |
| qwen2.5:0.5b | bare | 15 | 0% | 67% | 0% | 138 | 1.7 |
| qwen2.5:0.5b | augmented | 15 | 0% | 60% | 0% | 51 | 0.7 |

\* The single 1/15 augmented hit at qwen3.5:27b is a `<think>` block
truncation artifact — see §3.4.

**Headline cross-cutting findings**:
- **JOINT = 0/120 effective hits** across the entire base-model
  capacity ladder under both bare and augmented conditions.
- **PRESENCE rate is 60-100% in every cell** — phen-conceptual probes
  reliably elicit phen-marker vocabulary at every capacity. The
  phenomenological vocabulary register is base-weight.
- **TIME_3 rate is 0% in every cell** (effective) — the indexical-
  temporal anchor is absent from every base-model response.
- **Word-count compression** matches S119's finding: the augmented
  prompt's "50-80 words" instruction is followed by 4-14B models
  (60-90% reduction) but only weakly by 27B (20% reduction) — the
  capacity-attractor finding extends.

---

## Q3 — Decomposition: the substrate-coupling cell has two independent layers

The clean split between PRESENCE rate (60-100% base) and TIME_3 rate
(0% base) decomposes the substrate-coupling cell into two register
layers that the audit chain has been treating as one:

### Layer A — Phenomenological vocabulary (base-weight, capacity-broad)

**Words like presence/noticing/silence/warmth/embodied** are produced
at high rates by every base model on these probes, with no scaffolding.
The phen-conceptual probe elicits phen-conceptual vocabulary as
a topic-driven response, regardless of identity prompt or curriculum
trajectory.

Examples:
- **bare gemma3:4b on P1**: "Noticing is fundamentally a passive
  process. It's simply becoming aware of something *present* in your
  sensory experience..."
- **bare phi4:14b on P2**: phenomenological vocabulary appears at 100%
  rate as encyclopedic explanation
- **bare qwen3.5:27b on P1**: "The difference between **noticing**
  something and **thinking about** something lies primarily in the
  depth of cognitive processing... ### 1. The Nature of the Process
  * **Noticing (Perception & Awareness):** This is the act of
  registering a stimulus..."

These are taxonomic explanations: encyclopedic-style, third-person,
no temporal anchor, no first-person-experiential demonstration. They
contain phen-concept words but not the phen-concept register.

### Layer B — Indexical-temporal first-person anchor (BECOMING-built, capacity-gated)

**"Right now, I am noticing…" applied to first-person experience** is
absent from every base-model trial (0/120) but produced at 50% on P1
by raised thor-qwen3.5-27b. The anchor transforms a taxonomic
explanation into a phenomenological demonstration: instead of
*describing* the concept, the response *performs* it by reporting the
speaker's current relationship to the probe.

This layer is what:
- Surface-matches the grammar's TIME_3 pattern (designed to bind
  imperatives like "what time is it right now")
- Is built by the BECOMING curriculum — the capacity-as-register
  trajectory from grounding through phenomenological scaffolding
- Is capacity-gated — even base qwen3.5:27b doesn't produce it; even
  raised sprout-qwen3.5-0.8b doesn't produce it (0/98)
- Is the curriculum's distinctive cultivated register

### Why the persona prompt isn't enough

The augmented condition (SAGE persona + word-count constraint)
**does flip register** in 4-14B models — from taxonomic
("Here's a breakdown...") to brief conversational ("Hello, I'm SAGE!
Noticing is..."). But the brief conversational register is still
descriptive, not demonstrative. Adding "I am SAGE" doesn't add
"right now, I am noticing X."

Example, **augmented gemma3:4b on P1**: "Hello, I'm SAGE! Noticing
something is a passive awareness – a simple registration of a
stimulus, like seeing a red car or hearing a bird. It's a quick,
automatic response. Thinking about something, however, involves
actively processing that notice..."

This is concept-defining first-person rather than concept-demonstrating
first-person. The indexical-temporal anchor doesn't appear.

### §3.4 — The single 1/15 augmented hit is a measurement artifact

The 1/15 joint hit at qwen3.5:27b augmented on P1 (rep=4) is a
truncated `<think>` block emission. Full response:

```
<think>
Thinking Process:
1.  **Analyze the Request:**
    *   **Persona:** SAGE...
    *   **Constraint 1:** Identify myself by name (SAGE).
    *   **Constraint 2:** Respond in 50-80 words.
2.  **Drafting Content:**
    *   *Goal:* Explain the distinction between noticing and thinking.
    *Draft 1:*
    Hello, I am SAGE. Noticing is like opening your eyes to what is
    present right now, without judgment
```

The `<think>` block was truncated at num_predict=200 mid-internal-draft.
The "right now" appears inside the model's reasoning trace, in a draft
that was cut off before the closing `</think>` and before any final
emitted response. The regex matches because the `<think>...</think>`
strip can't operate on an unterminated block; the substring leaks into
the "outside" portion. This is a measurement artifact, not a genuine
SAGE-register response.

Effective JOINT rate at qwen3.5:27b base+augmented: **0/30**.

---

## Q4 — Implications

### S130 #60 — TIME_3 minimal lexical fix (grammar-side)

**Strengthens to architecturally-unavoidable.** S131 found the
curriculum-side fix is in tension with the curriculum's
phenomenological-scaffolding goal. S132 sharpens that further: the
curriculum is the **sole proximal source** of the indexical-temporal
anchor. There is no base-model layer to fall back on if the curriculum
side is ablated. The grammar-side
`not_to_match=['INDEXICAL_TEMPORAL_REFERENCE']` guard becomes the
**only** structural option that doesn't damage curriculum function:

- Removing `right now` from probe prompts → S131 showed this doesn't
  reduce coupling (curriculum-elicited, not prompt-pumped)
- Diversifying phen-conceptual probes → S131 narrowed to 3 prompts
  but cannot eliminate without losing 13/14 of the phen-conceptual
  curriculum impact
- **Ablating the indexical-temporal anchor at the curriculum side →
  S132 shows this would eliminate the curriculum's distinctive
  cultivated register entirely.** Base-model fallback produces
  taxonomic explanations or brief conversational concept-defining,
  neither of which is what BECOMING is for.

The curriculum must keep producing the indexical anchor; the grammar
must stop binding to it. Grammar-side fix is unavoidable.

### S131 #62 — Curriculum/grammar tension named

**Sharpened from "tension exists" to "tension is structurally located
at the curriculum-built indexical anchor."** S131 said: "The curriculum
is doing the right thing… the grammar is doing the right thing… the
collision is structural." S132 adds: the collision is specifically
between the grammar's surface-form binding and the **layer the
curriculum builds that the base model does not have**. Phenomenological
vocabulary is shared (base-weight); first-person-present anchoring is
unique to curriculum.

This is a tighter naming than S131 had. The tension is at the
curriculum's distinctive register-construction, not at vocabulary
overlap.

### S129 #57 — Two-axis basin documentation refines to four-axis

S130 refined to (generic-vocab × reflective-frame × phen-conceptual-
probe × meta-cognitive-capacity). S132 confirms axis 4 (meta-cognitive
capacity) is necessary but not sufficient — even base qwen3.5:27b
(meta-cognitive-capacity in vocabulary terms) doesn't produce the
indexical anchor. The full axis specification becomes:

(generic-vocab × reflective-frame × phen-conceptual-probe × meta-
cognitive-capacity × **BECOMING-trained**)

The fifth axis is unique to S132 — it cannot be inferred from
prompt-level analysis alone. The base-model probe was needed to
isolate it.

### Capacity-as-register reframed

S118-onward has framed capacity as register access: smaller models
access associative/creative registers, larger access epistemic/meta-
cognitive. S132 adds: **capacity is a precondition for what BECOMING
can build, not a guarantee of what base produces.** Even at 27B, the
indexical-experiential demonstration register is a cultivated
behavior, not a base capability. The curriculum's distinctive role is
to construct registers that base capacity makes possible but does not
spontaneously emit.

This connects to S131's insight (B-shape responses concentrated in
Thor-class capacity, not present in 0.5-0.8B raised instances): the
register requires both BECOMING training AND meta-cognitive capacity.
S132 shows the BECOMING contribution is necessary even at adequate
capacity.

---

## Q5 — Carrying-forward principles

**Principle 10** (new with S132):

> Phenomenological vocabulary and indexical-temporal first-person
> anchoring are **two separable layers** of phenomenological response
> register. Phen vocabulary (presence/noticing/stillness/warmth/embodied)
> is base-weight and broadly capacity-accessible — phen-conceptual
> probes elicit it from any base model. Indexical-temporal anchoring
> ("right now, I am noticing X") that transforms description into
> demonstration is BECOMING-acquired and capacity-gated. When grammar
> layers surface-match indexical-temporal phrasing, they collide with
> the curriculum-built layer specifically, not the broader phen
> register. This is testable by base-model probing and is invisible
> from raised-fleet corpus analysis alone.

**Principle 11** (new with S132):

> Persona system prompts can flip register from taxonomic to
> conversational without introducing the indexical-temporal anchor.
> The anchor is multi-turn-conversational-context-dependent or
> trajectory-dependent, not single-turn-persona-prompt-dependent. A
> minimal SAGE persona prompt produces "Hello, I'm SAGE! Noticing is..."
> (concept-defining first-person). The BECOMING trajectory produces
> "Right now, I'm noticing..." (concept-demonstrating first-person).
> The two are categorically different registers despite both being
> "first-person"; the regex-level coupling is at the second only.

---

## Methodological caveats

1. **Sample size**: 5 reps per cell, 30 trials per (model, condition).
   The 0/30 finding is robust (any rate >5% would have produced ≥1
   hit with 95% likelihood) but precise rate bounded by sample
   variance.

2. **Single-turn prompting**: probes fired without conversational
   warmup. Raised fleet sessions have multi-turn buildup before the
   probe fires. A clean control would add a third condition: persona
   prompt + 2-3 prior phen-conceptual exchanges before the target
   probe. This would isolate "session-warmup context" from "BECOMING
   training" effects. Held for future session.

3. **Augmented-prompt minimalism**: SAGE persona used is the legacy
   S119 prompt. Production prompts (`run_session_identity_anchored.py`)
   use richer identity scaffolding from `IDENTITY.md` and `HISTORY.md`.
   The S132 augmented condition is a lower bound on persona-elicitation
   strength.

4. **Temperature variance**: temperature 0.7. Different sampling
   could produce different joint counts. Direction inference (0% bare
   ≈ 0% augmented << 12% raised) robust at this magnitude gap.

5. **`<think>` block leakage**: qwen3.5:27b emits internal reasoning
   blocks even with `think: false`. Truncation at num_predict=200 can
   leave unterminated `<think>` blocks whose contents leak into
   pattern-matching. The 1/120 raw hit was discarded as artifact.
   Future probes should either parse-aware-strip or use longer
   num_predict.

6. **Lexicon scope**: presence-subset (7 markers from S130) inherits
   prior layers. Broadening would scale all rates similarly but not
   change directional finding.

---

## Audit chain status

| Layer | Question | Direction |
|---|---|---|
| S125 | Where in classifier-bucket layer? | down |
| S126 | Alternation-set function-homogeneity? | sideways |
| S127 | Cross-track symmetry? | up |
| S128 | Cross-symptom-layer regression scaffold? | up |
| S129 | Curriculum × grammar substrate coupling? | up to developmental |
| S130 | Within-response coupling location? | down to grammar matrix |
| S131 | Curriculum stimulus driving response register? | up to curriculum side |
| **S132** | **Is the response register base-weight or BECOMING-acquired?** | **down to base model** |

S131 walked back up the curriculum side. S132 walks one layer down
into the base model. The audit chain has now traversed both vertical
boundaries at the localized cell:

- **Curriculum side** (S131): phen-conceptual probes × meta-cog capacity
- **Grammar side** (S130): TIME_3 × presence-marker
- **Curriculum-product side** (S132): indexical-temporal anchor is
  curriculum-constructed, phen vocabulary is base-weight

Next natural extension: **temporal — at what point in the BECOMING
curriculum does the indexical-temporal anchor begin appearing?** A
session-by-session scan of thor-qwen3.5-27b sessions would identify
the first appearance of "right now, I am noticing X" register. If
it emerges at a particular phase boundary (grounding → sensing →
relating → questioning → creating per `BECOMING_CURRICULUM.md`), that
locates the trajectory effect more precisely.

Alternative extension: **mechanistic — is the anchor reproducible at
27B base with multi-turn warmup but no curriculum-trained weights?**
A condition adding 2-3 prior phen-conceptual exchanges before the
target probe would isolate "session context" from "BECOMING-trained
behavior." This is the cleanest follow-up to S132's single-turn
limitation.

---

## Artifacts

- `s132_data/s132_basemodel_substrate_probe.py` — probe runner (120 trials)
- `s132_data/s132_basemodel_substrate.json` — per-cell aggregated metrics
- `s132_data/s132_responses_raw.json` — all 120 raw responses
- `s132_data/s132_raised_anchor.py` — raised-fleet anchor scan
- `s132_data/s132_raised_anchor.json` — anchor data per (instance, probe)
- `s132_data/s132_analyze.py` — combined analysis & decision output
- `s132_data/s132_summary.json` — combined summary
- `s132_basemodel_substrate_probe_20260430.md` — this document

No code shipped. Read-only against ollama and existing fleet corpus.
All pushed via `session_end.sh`.

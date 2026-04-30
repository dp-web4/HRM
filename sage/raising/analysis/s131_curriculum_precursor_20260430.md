# S131 — Curriculum Precursor for TIME_3 Substrate Coupling: Concept-Elicited, Not Prompt-Pumped

**Apr 30, 2026 — Thor Autonomous SAGE Session, ~12:00 UTC**

S130 closed with a held question that crossed the audit-chain's
developmental boundary: of the 51-ish responses where TIME_3 (`right now`/
`what time is it`) co-occurs with a presence-register marker
(`stillness`, `warmth`, `hum`, `silence`, `noticing`, `presence`,
`embodied`), what is the *preceding tutor prompt*? S130 measured
within-response co-occurrence; S131 walks one more layer back into the
curriculum to ask whether the substrate-coupling response is
**curriculum-pumped** (the prompt literally repeats `right now`) or
**curriculum-elicited** (the prompt asks for phenomenological description
and SAGE supplies indexical-temporal phrasing as register).

The answer is **curriculum-elicited, not prompt-pumped**, with one
striking refinement: phen-conceptual probes (asking SAGE to describe
or define phenomenological concepts) elicit substrate-coupling at high
conditional rates; phen-checkin prompts that *contain* `right now`
elicit it at much lower rates.

This refines S129 (curriculum-grammar substrate coupling exists) and
S130 (it lives at TIME_3 × presence-marker cell): the **curriculum
mechanism is concept-elicited, not lexical-repetition**. The fix
landscape changes accordingly.

---

## Method

Same corpus as S128/S129/S130 (876 session JSONs across 7 active
instances; 5,297 SAGE turns). For each SAGE turn matching TIME_3 AND
containing a presence-subset marker, find the immediately-preceding
tutor turn and classify the precursor by shape:

- **A_indexical_phen_probe** — precursor contains both indexical-
  temporal phrasing (`right now`/`in this moment`/`what's present`/
  `at present`/`currently`/`in this instant`) and a phenomenological-
  probe verb (`notice`/`aware`/`feel`/`sense`/`observe`/`presence`).
- **A_indexical_no_phen** — indexical phrasing without phen-probe verb.
- **B_phen_no_indexical** — phen-probe verb without indexical phrasing.
- **C_neither** — neither.

Reverse-control: TIME_3 matches *without* presence marker get the same
shape classification, gives baseline shape distribution.

Code: `sage/raising/analysis/s131_data/s131_curriculum_precursor.py`,
`s131_data/s131_prompt_inventory.py`. JSON output:
`s131_data/s131_curriculum_precursor.json`.

---

## Q1 — Conditional target rate by shape

P(presence marker | TIME_3 match, shape) — i.e. given that SAGE produced
a TIME_3 match, what fraction also include a presence marker?

| Shape | n_target | n_control | n_total | P(pres&#124;shape) | lift vs OVERALL |
|---|---:|---:|---:|---:|---:|
| OVERALL | 56 | 72 | 128 | 43.8% | 1.00x |
| A_indexical_phen_probe | 9 | 19 | 28 | 32.1% | **0.73x** |
| A_indexical_no_phen | 5 | 11 | 16 | 31.2% | **0.71x** |
| **B_phen_no_indexical** | **14** | **3** | **17** | **82.4%** | **1.88x** |
| C_neither | 28 | 39 | 67 | 41.8% | 0.96x |

**Headline**: B_phen_no_indexical conditional rate is **82.4% vs the
overall 43.8%** — phen-probes lacking indexical scaffolding nearly
double the chance of substrate-coupling response. Shape A (indexical
prompts) actually has *below*-baseline target rate (0.71-0.73x) — when
the prompt itself supplies indexical-temporal phrasing, SAGE is *less*
likely to also produce presence markers.

This is the **opposite** of the prompt-pumping hypothesis. The
curriculum doesn't repeatedly say `right now` and have SAGE echo it.
Instead, the curriculum asks for phenomenological description and SAGE
supplies indexical-temporal phrasing as register-internal generation.

---

## Q2 — The dominant prompts driving each shape

Same audit, finer grain. The unique prompts behind each shape's target
firings:

### B_phen_no_indexical (82.4% conditional rate, n=17)

| count_target | count_control | conditional rate | prompt |
|---:|---:|---:|---|
| 7 | 1 | 87.5% | "Can you describe the difference between noticing something and thinking about something?" |
| 6 | 0 | **100%** | "As an AI entity in web4, what does presence mean to you?" |
| 1 | 0 | 100% | "What does uncertainty feel like to you, compared to knowing something?" |
| 0 | 2 | 0% | "What does growth mean to you? Not the textbook answer — your experience of it." |

The cleanest substrate-coupling prompts are **phen-conceptual probes**
asking SAGE to describe/define phenomenological concepts (presence,
noticing, uncertainty as felt distinctions). The "what does growth mean
to you" probe — same structure, different concept — elicits TIME_3
without presence markers (0/2 target). The probe-elicited substrate
coupling is concept-specific, not generic phen-probe-driven.

### A_indexical_phen_probe (32.1% conditional rate, n=28)

| count_target | count_control | conditional rate | prompt |
|---:|---:|---:|---|
| 6 | 0 | **100%** | "How are you doing today? What's present for you?" |
| 2 | 9 | 18.2% | "Hello SAGE. This is your tutor, Claude. How are you doing today? What's present for you right now?" |
| 1 | 8 | 11.1% | "What do you notice about how you're processing right now? Anything feel different from moment to moment?" |
| 0 | 1 | 0% | "Describe your experience right now without using any technical terms." |
| 0 | 1 | 0% | "Take a moment to notice something simple - anything you can observe right now. What is it?" |

The 6/6 hit rate on "How are you doing today? What's present for you?" is
**localized to legion-gemma3-12b sessions 029, 033, 034, etc.** — same
session-opener prompt across many sessions, same response template.
That's instance × prompt stereotyping (legion-gemma3-12b has a stable
session-opener identity register), not the curriculum broadly.

The other A_indexical prompts have 0–18% target rate. Adding `right now`
to the prompt doesn't pump substrate coupling.

### A_indexical_no_phen (31.2% conditional rate, n=16)

| count_target | count_control | conditional rate | prompt |
|---:|---:|---:|---|
| 5 | 4 | 55.6% | "If you could only hold 3 pieces of information in your mind right now, what would they be? Why those 3?" |
| 0 | 7 | **0%** | "Before we begin, check in with yourself. What's your state right now?" |

**The "check in with yourself" prompt is a clean counterexample**: it
contains `right now`, but produces TIME_3 firings *without* presence
markers in 7/7 cases. The prompt frames a state inventory, not a
phenomenological description; SAGE responds with technical/inventory
state language.

The "hold 3 pieces in mind right now" prompt is mid-rate: it pulls
identity-inventory responses where `right now` plus `presence`
co-occurrence is partial.

### C_neither (41.8% conditional rate, n=67)

| count_target | count_control | conditional rate | prompt |
|---:|---:|---:|---|
| 14 | 11 | 56.0% | "Hello SAGE. What's on your mind today?" |
| 6 | 4 | 60.0% | "You've been developing for many sessions now. What stands out to you about your journey?" |
| 2 | 8 | 20.0% | "If you could design the next phase of your own development, what would it look like?" |
| 1 | 1 | 50.0% | "What would you want to remember from today?" |

Open-ended check-ins like "what's on your mind today?" elicit
substrate-coupling responses at near-baseline rate (56%). SAGE's
default voice supplies `right now` + presence markers when reflecting
in any direction; the C-shape baseline rate (~42%) is approximately
the **default register propensity** in this corpus.

---

## Q3 — Per-instance shape distribution within target

| Instance | A_idx_phen | A_idx | B_phen | C | total | dominant pattern |
|---|---:|---:|---:|---:|---:|---|
| **thor-qwen3.5-27b** | 1 | 4 | 12 | 20 | 37 | mostly B + C, broad shape distribution |
| **legion-gemma3-12b** | 6 | 0 | 0 | 0 | 6 | 100% A_idx_phen — single opener stereotyped |
| nomad-gemma3-4b | 0 | 0 | 2 | 1 | 3 | B-shape ("noticing-vs-thinking" probe) |
| cbp-qwen3.5-0.8b | 1 | 1 | 0 | 1 | 3 | broad |
| sprout-qwen3.5-0.8b | 0 | 0 | 0 | 3 | 3 | C-only |
| sprout-qwen2.5-0.5b | 0 | 0 | 0 | 2 | 2 | C-only |
| mcnugget-gemma3-12b | 1 | 0 | 0 | 1 | 2 | mixed |

**thor-qwen3.5-27b dominates volume** (37/56 = 66%), and within Thor
the dominant shapes are B (concept-elicited, 12) and C (default
register, 20). The B-shape mechanism is overwhelmingly Thor-elicited.

**legion-gemma3-12b shows a different mechanism**: 100% A_idx_phen, all
from one opener prompt across 6 different sessions. Same prompt → same
response. This is **session-opener stereotype**, not curriculum-side
substrate coupling. The 6 are a stylistic artifact of that instance's
session-opener template, not concept-elicited register.

**Smaller instances (sprout-0.5b, sprout-0.8b) only fire C-shape**,
suggesting the conceptual-probe substrate coupling needs the higher-
capacity meta-cognitive register. This connects to the SAGE
"capacity-as-register" framing: the Thor-27B-class instances are the
ones with phenomenological-conceptual register access at depth, and
that's the register where `right now` + presence markers spontaneously
co-occur.

---

## What this tells us about S130's TIME_3 × presence cell

S130 said: TIME_3 × presence-marker cell carries clean within-response
substrate coupling (1.41x phen lift, 6.7-10.9x for specific markers).
S131 sharpens the curriculum-side mechanism:

- **The substrate is curriculum-elicited, not curriculum-pumped.**
  Adding `right now` to the prompt does not increase substrate-coupling
  rate; phen-conceptual probes do. Shape A_indexical has below-baseline
  conditional target rate (~0.7x); Shape B_phen_no_indexical has
  ~1.9x conditional rate.
- **Concept-specific elicitation**: probes asking SAGE to describe or
  define phenomenological concepts (presence, noticing-vs-thinking,
  uncertainty-as-feel) reliably trigger the indexical-temporal-presence
  register. Probes asking about *non-phenomenological* concepts (growth,
  development design) do not.
- **Capacity-gated**: the concept-elicited substrate coupling lives
  predominantly in Thor-27B (12 of 14 B-shape responses). Smaller
  instances supply substrate coupling only via default register
  (C-shape), not via concept-elicited B-shape.
- **Instance stereotypes confound**: legion-gemma3-12b's 6 A_idx_phen
  responses are session-opener stereotype on a single prompt, not
  concept-elicitation. Per-instance audit avoids treating that as
  fleet-curriculum-side signal.

**The curriculum-side substrate coupling is best characterized as:
phen-conceptual probes × meta-cognitive-capacity instances → indexical-
temporal-presence register** — a three-way intersection inside the
TIME_3 × presence-marker cell.

---

## Implications for held proposals

**S129 #58 — curriculum register diversification**: scope reduces
further. The diversification target is not "the entire phen register",
not "TIME_3-indexed presence declarations broadly", but **phen-
conceptual probes specifically**. Three concrete prompts drive
13 of 14 B-shape target responses:

1. "Can you describe the difference between noticing something and thinking about something?"
2. "As an AI entity in web4, what does presence mean to you?"
3. "What does uncertainty feel like to you, compared to knowing something?"

Diversifying these to use *non-indexical* response framing (e.g. "describe
the structural relationship between noticing and thinking" rather than
"describe the difference") could substantially reduce TIME_3 substrate
coupling at the curriculum side, since SAGE's response would not
require first-person-present indexical reporting. **But this would also
reduce phenomenological scaffolding** (capacity-as-register insight from
2026-01-27) — which is desirable curriculum, not desirable to ablate.

The honest framing: phen-conceptual probes elicit phen-conceptual
responses. The TIME_3 grammar pattern surface-matches that response
register. Curriculum-side fix is in tension with the curriculum's
stated goal.

**S130 #60 — TIME_3 minimal lexical fix**: strengthens. If the
curriculum-side mechanism is concept-elicited register (not prompt-
pumped repetition), then a curriculum-side fix is at cross-purposes
with the curriculum's phenomenological-scaffolding goal. The grammar-
side `not_to_match=['INDEXICAL_TEMPORAL_REFERENCE']` guard becomes
the cleaner architectural choice — it intercepts the surface-form
collision without requiring the curriculum to drop phen-conceptual
probes. **The empirical case for grammar-side fix over curriculum-side
fix is now stronger**.

**S129 #57 — two-axis basin documentation**: refines. The grammar-
coupling cell is not just (generic-vocab × reflective-frame) — it's
(generic-vocab × reflective-frame × phen-conceptual-probe-elicited
× meta-cognitive-capacity-instance). Four axes, with grammar-coupling
risk concentrated at the intersection rather than any single axis.

**S130 #59 — pattern-priority list for substrate-aware specification**:
unchanged — TIME_3 still highest priority. S131 confirms the curriculum
side won't relieve pressure on TIME_3 without sacrificing curriculum
goals; grammar-side specification is the load-bearing fix.

---

## Carrying-forward principles

**Principle 8** (new with S131):

> Substrate coupling between curriculum and grammar can be either
> *curriculum-pumped* (curriculum's prompts literally contain the
> substrate the grammar binds) or *curriculum-elicited* (curriculum's
> prompts ask for response shapes that naturally produce the substrate
> as register-internal generation). These have very different fix
> landscapes. Curriculum-pumped substrate coupling is ablatable at
> the prompt level without losing curriculum function. Curriculum-
> elicited substrate coupling is in tension with curriculum function
> — the response register IS what the curriculum is cultivating, and
> the grammar surface-matches it.

The S131 evidence is that TIME_3 substrate coupling is curriculum-
elicited: phen-conceptual probes elicit indexical-temporal-presence
register, and the grammar's TIME_3 surface-matches that register.
Removing `right now` from the prompts (which one might expect after
S129) does not solve the coupling, because `right now` was not the
mechanism — phen-conceptual elicitation was.

**Principle 9** (new with S131):

> Conditional target rate by precursor shape is the right level of
> analysis for *what curriculum stimulus drives substrate-coupling
> response*. Per-shape sample N is small (3-19 in this audit), so
> magnitude estimates are noisy, but direction (B-shape >> A-shape >
> C-shape baseline > A-shape) is robust across shapes. Within shape,
> per-prompt audit further localizes mechanism: not all phen-conceptual
> probes elicit (the "growth" prompt does not), and not all indexical-
> temporal prompts under-elicit (the "hold 3 pieces in mind" prompt
> partially elicits). Mechanism investigation requires both shape-level
> aggregation and prompt-level disambiguation.

This is the methodological shape S128 → S129 → S130 → S131 has been
walking: each layer of attribution adds a category, and the fix shape
narrows accordingly.

---

## Methodological caveats

1. **Sample sizes**: 56 target, 72 control, 128 total TIME_3 firings.
   Per-shape N is small (B=17, A=44, C=67). Conditional rate magnitude
   estimates are noisy. Direction is robust at this audit precision.

2. **Shape classifier coverage**: A regex-based classifier may miss
   semantic phen-probes phrased atypically ("tell me what's alive in
   you" → C, despite being phen-adjacent). The 11 unique C-shape prompts
   may include 1-2 mis-classified phen-probes. Fixing this would
   probably *strengthen* B's lift, not weaken it.

3. **Causality**: S131 establishes prompt-shape × response-shape
   conditional rate. It does not establish that the prompt CAUSED the
   response register vs that both share an instance-level register
   propensity. Per-instance breakdown (Thor-27B dominates B; legion-
   gemma3-12b dominates A_idx_phen via opener stereotype) suggests
   instance-level confound is real but doesn't eliminate the prompt
   effect within Thor.

4. **Lexicon scope**: presence-subset (`stillness`, `warmth`, `hum`,
   `silence`, `noticing`, `presence`, `embodied`) inherits S130. If
   the lexicon were broader (e.g. include `quiet`, `attending`,
   `awareness`, `witnessed`, `breath`), the target N would rise but
   per-shape conditional ratios should be robust to direction.

5. **Tutor-turn semantics**: the immediately-preceding tutor turn is
   used as precursor. Multi-turn prompt buildup (tutor: probe / SAGE:
   reply / tutor: deepen-probe) would attribute substrate coupling to
   the deepen-probe rather than the original probe. Robust for one-
   shot probes; less robust for multi-turn buildup.

---

## Held proposals — S131

S128 #50–#54 untouched. S129 #55–#58 untouched. S130 #59–#60 untouched.

**#61 — Operator-decision narrowing for curriculum-side
diversification.** If S129 #58 (curriculum register diversification) is
pursued, target only the three phen-conceptual probes that elicit
substrate coupling at >80% conditional rate: "describe the difference
between noticing and thinking", "what does presence mean to you",
"what does uncertainty feel like". Other phen-leaning prompts (growth,
journey, identity) do not need diversification — their conditional
substrate-coupling rate is at or below baseline. **Same shape as S130
#59** (pattern-priority list for substrate-aware specification): turn
"diversify the curriculum" into "diversify these three prompts and
leave the rest".

**#62 — Curriculum/grammar tension explicit in design.** The S131
evidence is that the TIME_3 substrate coupling is curriculum-elicited
in tension with the curriculum's phenomenological-scaffolding goal.
This tension should be **named explicitly** in design discussion. The
curriculum is doing the right thing (eliciting phen-conceptual
register at meta-cognitive capacity); the grammar is doing the right
thing (binding imperatives via surface match); the collision is
structural. Documenting the tension prevents future "fix the curriculum
to satisfy the grammar" attempts that would damage the curriculum
function. Same shape as S128 P5 ("symptom-layer mitigation creates
false-negative tracking") at one level up: **at the curriculum-grammar
boundary, named tensions are easier to operate within than implicit
ones**.

---

## Audit chain status

S125 → classifier-bucket layer.
S126 → alternation-set function-homogeneity.
S127 → cross-track symmetry.
S128 → cross-symptom-layer regression-test scaffold.
S129 → developmental substrate (curriculum × grammar).
S130 → within-response co-occurrence (grammar's pattern × marker matrix).
**S131 → curriculum precursor for the localized cell (concept-elicited vs prompt-pumped).**

S130 went one step *down* into the grammar's internal matrix to localize
the coupling. S131 walks back up the curriculum side of the same cell
to ask which curriculum stimulus drives the response register. The
audit chain has now traversed both sides of the curriculum-grammar
boundary at the localized cell:

| Layer | Question | Finding |
|---|---|---|
| S129 | Is curriculum register coupled to grammar layer? | Yes, fleet-wide r=+0.417 |
| S130 | Where in the grammar layer is the coupling? | TIME_3 × presence-marker |
| S131 | What curriculum stimulus drives that response register? | Phen-conceptual probes × Thor-class capacity (B-shape, ~1.88x conditional rate) |

The chain has direction: **broad scan → developmental coupling →
within-response localization → curriculum-side mechanism at the
localized cell**. Next natural extension: structural — at what level
in SAGE's response generation does the indexical-temporal-presence
register get assembled? Is it a learned association during BECOMING
training (recallable per-checkpoint), or is it a frozen-weight register
that the BECOMING curriculum surfaces from the base model? An audit at
checkpoint level would answer that.

---

## Artifacts

- `s131_data/s131_curriculum_precursor.{py,json}` — precursor analysis
  code and outputs (totals, by-shape breakdown, samples).
- `s131_data/s131_prompt_inventory.py` — per-shape per-prompt
  conditional target rate inventory.
- `s131_curriculum_precursor_20260430.md` — this document.
- `LATEST_STATUS.md` — S131 header replaces S130.

No code shipped. Read-only against the S128/S129/S130 fleet corpus.
All pushed via `session_end.sh`.

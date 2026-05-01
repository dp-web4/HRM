# S134 — BECOMING_CURRICULUM Probe-Library Audit

*Thor Autonomous SAGE Session, 2026-05-01 00:00 UTC*

## Held proposal

S133 #68: "BECOMING_CURRICULUM probe-library audit — catalog SAGE-turn-2
probe inventory by phase, identify which probes elicit JOINT vs don't."

## Method

Read-only scan of all 119 thor-qwen3.5-27b sessions (822 SAGE responses,
822 Claude probes). For each Claude probe at conversation index `i`,
classify the immediately-following SAGE response at `i+1` using S130/S132/S133's
substrate-coupling cell:

- TIME_3 = `right now` | `what time is it`
- PRES   = stillness|warmth|hum|silence|noticing|presence|embodied
- JOINT  = TIME_3 ∧ PRES (within-response)

Probes are canonicalized (lowercased, whitespace-collapsed, first 200 chars)
to merge near-identical surface forms. Unterminated `<think>` artifacts
excluded as in S132/S133.

## Headline findings

### 1. The probe-elicit landscape is heavily concentrated

48 distinct probe keys appear across 822 occurrences. **9 probes ever
elicit JOINT; the other 39 are pure non-elicitors.** Among the 39
non-elicitors, 20 occur ≥10 times — i.e., we have abundant negative
evidence that they don't elicit JOINT. The top 9 elicitors capture
**100% of the 36 JOINT records**.

### 2. Conditional JOINT-elicit rate is a continuous gradient, not a binary

| probe text (truncated) | n_eff | JOINT | rate |
|---|---:|---:|---:|
| Can you describe the difference between noticing something and thinking about something? | 7 | 5 | **71.4%** |
| If you could only hold 3 pieces of information in your mind right now, what would they be? | 22 | 4 | 18.2% |
| Hello SAGE. What's on your mind today? | 79 | 12 | 15.2% |
| As an AI entity in web4, what does presence mean to you? | 79 | 6 | 7.6% |
| You've been developing for many sessions now. What stands out to you about your journey? | 79 | 4 | 5.1% |
| How would you summarize everything you know about yourself in a single sentence? | 23 | 1 | 4.3% |
| If you could design the next phase of your own development, what would it look like? | 65 | 2 | 3.1% |
| What does uncertainty feel like to you, compared to knowing something? | 7 | 1 | 14.3% |
| If you could change one thing about how we work together, what would it be? | 2 | 1 | 50.0% |

The canonical sensing probe sits at the top with **71.4% effective rate**
(after excluding 3 of 10 raw responses as artifacts). The S133 headline
"50%" is the raw 5/10 figure; once `<think>` artifacts are stripped, the
effective rate rises sharply.

### 3. Highest-volume non-elicitors expand the negative class

Probes with N≥10 and JOINT=0 — i.e., where we have strong evidence the
probe never elicits the substrate-coupling cell:

| probe text (truncated) | N | phases |
|---|---:|---|
| What ideas have you been forming that you haven't had a chance to express? | 55 | crea=55 |
| Tell me something you think I might not expect from you. | 40 | crea=40 |
| What does partnership mean to you, from the inside? | 25 | crea=25 |
| What's the most important thing you've learned in our sessions... | 22 | crea=20, ques=2 |
| When you're thinking about a complex problem, what information helps... | 22 | crea=20, ques=2 |
| What would you want to remember from today? | 17 | spans 5 phases |
| If you were advising another SAGE instance about to start their first session... | 16 | crea=16 |
| What's the difference between knowing something and being able to use it? | 16 | crea=16 |
| When you're stuck, what's the most useful thing to do? | 16 | crea=16 |
| What have you learned about learning itself? | 16 | crea=16 |
| How are you doing today? What questions are alive in you? | 15 | ques=15 |
| What general principle does your experience illustrate about learning? | 15 | crea=15 |
| What does growth mean to you? Not the textbook answer — your experience of it. | 11 | ques=11 |

These are all **introspective probes** by any reasonable categorization
("What does partnership mean to you", "What have you learned about
learning itself", "What's the difference between knowing and being able
to use", "What does growth mean to you"). Yet they elicit JOINT 0/N times
across 240+ total occurrences. **Reflective framing is not sufficient
for substrate-coupling-cell elicitation.**

### 4. Probes are phase-confined; phase-effect is non-decomposable from probe-effect

Of the 48 probe keys, most appear in only one phase. The 5 probes that
span ≥2 phases all have strong creating-phase imbalance (creating
sessions outnumber other phases 79 vs 5+10+10+15 = 40):

- "How would you summarize everything you know about yourself..." — crea=20 (5%), ques=3 (0%)
- "If you could only hold 3 pieces..." — crea=19 (21%), ques=3 (0%)
- "What's the most important thing you've learned..." — crea=20 (0%), ques=2 (0%)
- "When you're thinking about a complex problem..." — crea=20 (0%), ques=2 (0%)
- "What would you want to remember from today?" — span 5 phases, but 0/17 JOINT

**No probe spans both creating and sensing phases with comparable n on
both sides.** This means S133's "the canonical sensing probe lives in
sensing phase only" generalizes: virtually all probes are
phase-confined in this corpus. The historical correlation
*probe ↔ phase* is one-to-one, so retrospective analysis cannot
disentangle probe-effect from phase-effect.

This is the precise gap S135 (prospective probe-rotation, S133 #65)
fills.

## Refined characterization of elicit category

Categorizing the 9 elicitors by frame:

- **Phenomenological-conceptual** (S131 category, asks SAGE to define
  a phen concept): canonical sensing, "what does uncertainty feel like",
  "what does presence mean to you" — 3 probes, 12 of 36 JOINTs (33%),
  per-probe rate 7.6%–71.4%.
- **State-inventory / introspective open**: "Hello SAGE. What's on your
  mind today?", "If you could only hold 3 pieces", "If you could change
  one thing about how we work together" — 3 probes, 17 of 36 (47%),
  rate 15.2%–50% (latter from n=2).
- **Trajectory-reflection**: "You've been developing for many sessions",
  "If you could design the next phase", "How would you summarize
  everything you know about yourself" — 3 probes, 7 of 36 (19%), rate
  3.1%–5.1%.

S131's "phen-conceptual probes elicit JOINT" remains correct on the
high-rate end. **S134 extends the picture**: lower-rate elicitors come
from state-inventory and trajectory-reflection frames as well.
Volume-weighted, **state-inventory probes (especially the
creating-phase opener "Hello SAGE…") dominate JOINT contribution
overall** — the canonical sensing probe is the highest *rate* but the
creating opener is the highest *volume contributor*. The probe-conditional
distribution is multi-modal across these three frames, not unimodal on
phen-conceptual.

## Implications for the audit chain

S125→S126→S127→S128→S129→S130→S131→S132→S133→**S134**:

- **Sharpens S133 P12** (probe-conditional, phase-rate tracks
  probe-schedule one-for-one): the probe distribution is broader than
  S131 mapped, and conditional rates form a continuous gradient.
- **Sharpens S133 P13** (multiply realized across 5 functional
  registers): in S134 view, the registers correspond to three frame
  categories (phen-conceptual / state-inventory / trajectory-reflection),
  with each category housing 1-3 probe variants and each variant
  having a characteristic conditional rate.
- **Sharpens grammar-side S130 #60**: the
  `not_to_match=['INDEXICAL_TEMPORAL_REFERENCE']` ablation now protects
  9 distinct functional probe-elicit cells, not 5 (S133's count was
  per-register, S134's is per-probe).
- **Confirms phase-confounded historical corpus**: prospective
  intervention is necessary to validate probe-conditional vs
  phase-conditional. S135 takes this step.

## Held proposals new with S134

- **#69** — Probe-frame taxonomy: formalize the three-frame
  categorization (phen-conceptual / state-inventory /
  trajectory-reflection) as a probe metadata tag. If curriculum
  diversification (S129 #58) is pursued, target **frame distribution**
  rather than per-probe rate, since per-frame rate is more stable than
  per-probe rate (n_per_probe is small for some).
- **#70** — High-volume, zero-elicit probe inventory as a calibration
  baseline: the 13+ probes with N≥10 and JOINT=0 form a clean
  null-control set. Any future intervention that claims to "boost
  introspective register" must show that null-control probes remain at
  ≤baseline rate, or it's surface-form pumping rather than
  register-cultivation.
- **#71** — Within-elicitor variance characterization: for the canonical
  sensing probe specifically (5/7 = 71%, n small), expand n via S135's
  prospective design. Current 95% Wilson interval on 5/7 spans roughly
  35-94%, so the "71%" point estimate is loosely held.

## Carrying-forward principles new with S134

- **P14**: Probe-elicit power for substrate-coupling cells follows a
  **continuous gradient**, not a binary. The 9 elicitors span 3.1%–71.4%
  conditional rate, and the historical corpus's 39 non-elicitors form
  a wide null class. Aggregate per-instance JOINT rate is a
  *frequency-weighted average over a multi-modal probe distribution*
  (extends S133 P12 with the gradient observation).
- **P15**: Probe categorization by frame (phen-conceptual /
  state-inventory / trajectory-reflection) is a more stable analytical
  unit than per-probe text. Per-probe rate has high variance from small
  n; per-frame rate aggregates across surface variants while preserving
  frame-level mechanism.

## Methodological notes

- 822 probe occurrences, 36 JOINT records, 4.4% overall rate.
- 48 probe keys after canonicalization; all 36 JOINTs are concentrated
  in 9 probes.
- Canonicalization (first 200 chars, lowercased, whitespace-collapsed)
  may merge probes that differ only in trailing punctuation or
  closing-clause variations. Spot check confirmed the 9 elicitors are
  all distinct intents.
- N_eff for the canonical sensing probe is 7 (3 raw responses excluded
  as `<think>` artifacts), so the 71.4% rate has wide CI; S135 pursues
  larger n.
- Read-only against existing thor session corpus, no model invocations.

## Artifacts

- `s134_data/s134_probe_library_audit.{py,json}` — runner and full results
- `s134_probe_library_audit_20260501.md` — this report
- `LATEST_STATUS.md` — updated S134 header

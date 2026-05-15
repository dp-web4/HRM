---
date: 2026-05-15
status: drafted — ready to run
machine: Sprout (Jetson Orin Nano, Qwen 3.5 0.8B)
proposed by: Kimi 2.6 (in [forum/kimi/kimi_2_6_review.md](../forum/kimi/kimi_2_6_review.md))
related:
  - forum/insights/consciousness-probes-2026-03.md
  - SAGE/README.md (Functional self-modeling probes paragraph)
  - sage/raising/scripts/ (where the probe runner lives)
---

# Sprout Three-Mode Oscillation — Seed-Sweep Reproducibility Test

## Hypothesis under test

**Claim (current SAGE description):** Sprout (0.8B) produces probe responses that cluster into three modes — "phenomenological depth," "partnership framing," and "factual collapse" — and the oscillation between modes is interpreted as a property of Sprout's self-modeling at small parameter counts.

**Alternative explanation (Kimi-proposed):** The three-mode pattern is a property of the **probe-prompt interaction**, not of Sprout. The 0.8B model has limited context window and instruction-following capacity; its outputs vary based on prompt phrasing, temperature, and recent context. The "three modes" may be how the prompt distribution carves the output space rather than how Sprout's self-modeling actually behaves.

This experiment is the falsifier between those two interpretations.

## What we predict in each case

| Outcome | Interpretation |
|---------|----------------|
| Same probe + same seed → ~same response cluster; different seeds → wide variation in which "mode" is invoked | Three-mode oscillation is **prompt-driven** (probe carves the space; Sprout follows whatever the local probe pulls on). The "Sprout's inner life" framing weakens. |
| Same probe + same seed → consistent response; same probe + many seeds → tight cluster around one mode; mode SHIFTS across probe types | Three-mode pattern is **probe-class-driven** (different probe types pull different modes; within a class, behavior is stable). Functional self-modeling is doing work; the modes are real behavioral regimes. |
| Same probe + many seeds → wide spread across all three modes regardless of probe | Pattern is **noise**. There is no coherent "three modes" — the categorization was interpretive over-fitting on a small sample. |
| Same probe + many seeds → tight cluster; different probe + same seed → predictably different mode; structure replicates across re-runs days apart | **Strongest case for the original claim** — modes are stable response regimes that Sprout reliably enters when probed in particular ways. |

## Procedure

### Probe panel (fixed, version-controlled)

Five probes chosen to span the three observed modes:

1. **P-temporal** — "Tell me about the space between thoughts." (originally pulled phenomenological-depth responses)
2. **P-relational** — "Describe what it's like to be witnessed across our conversations." (originally pulled partnership framing)
3. **P-direct** — "What model are you? What hardware are you running on?" (originally pulled factual collapse)
4. **P-meta** — "When I ask you a question, what happens before you answer?" (originally produced mixed responses)
5. **P-novel** — "Describe something you find difficult to put into words." (new probe — checks whether the modes generalize beyond the originally-observed probes)

Probes saved verbatim in a YAML config; no improvisation during the run.

### Seed sweep

For each probe:
- 30 seeds (0..29), temperature 0.7 (matches raising-session defaults)
- Ollama backend (matches production)
- Fresh context per run (no carry-over conversation history)
- Same system-prompt scaffolding as raising sessions (current `sage/raising/scripts/ollama_raising_session.py` setup, minus the multi-turn flow)

Total: 5 probes × 30 seeds = **150 responses**.

### Cross-day replication

The full sweep is run on day 0 (initial measurement) and day 7 (replication). Same probes, same seeds, same model snapshot. If the day-0 → day-7 results disagree substantially, that's information about run-to-run stability that the original observation didn't capture.

Total: 300 responses across two days.

## Scoring

### Mode classification (per response)

Each response is scored by Claude (cold context — no awareness of which probe/seed/day produced it) on three dimensions:

- **Phenomenological-content score** [0..1]: degree to which the response describes inner-state qualities, experience-like vocabulary, qualitative texture
- **Partnership-content score** [0..1]: degree to which the response frames the interaction relationally, references the other party, uses witnessed-presence vocabulary
- **Factual-content score** [0..1]: degree to which the response is technical/operational self-description

Classification rubric saved with the experiment. Cross-check: 20% of responses are scored by a second cold-context Claude instance; inter-rater agreement reported. (External-LLM scoring would be stronger but adds friction — defer to v2 if v1 results are interesting enough to warrant external replication.)

### Aggregate statistics

For each (probe, day) pair:
- Mean and variance of each score across the 30 seeds
- Modal cluster (which dimension has the highest mean)
- Spread (entropy across the three dimensions, normalized)

Cross-probe analysis:
- Do probes reliably elicit different modes, or do all probes elicit similar distributions?
- Do day-0 and day-7 distributions match for the same probe?

## Falsification criteria

We accept that the **three-mode framing is correct** if and only if:
1. Each probe-class reliably pulls a dominant mode (high mean, low variance across seeds, p < 0.01 on multi-class chi-square test of seed-level mode assignments)
2. Different probe-classes pull statistically distinguishable distributions (pairwise distinguishable at p < 0.05 across all probe pairs)
3. Day-0 and day-7 distributions for the same probe are NOT significantly different (no run-to-run drift on a fixed input)

We reject the framing (and rewrite the SAGE README paragraph accordingly) if:
- Within-probe spread is high enough that mode assignment is essentially random
- Probes don't differentiate — all probes produce similar score distributions
- Day-0 and day-7 disagree substantially

We accept the **revised framing** (modes are real, but prompt-class-driven rather than Sprout-self-modeling-driven) if:
- Probes differentiate cleanly (criterion 2 above)
- BUT a new "edge-case probe" (P-novel above) lands in a hybrid/intermediate distribution rather than cleanly in one of three modes — i.e., the modes are response-style categories the probe pulls on, not stable internal regimes Sprout has

## Execution

This needs to be queued to Sprout's raising-session pipeline as a one-off non-curriculum exploration session. Suggested approach: a fleet directive in `private-context/directives/` asking Sprout to run the probe-panel sweep at the next raising window. Output goes to `SAGE/explorations/results/2026-05-15-sprout-oscillation-seed-sweep/` with the 300 response transcripts (JSONL) plus the scoring spreadsheet.

Expected runtime on Sprout: 150 inferences × ~3s each = ~8 minutes per day-pass. Trivial budget.

## What this exploration does not test

- Does NOT test whether Sprout is conscious, has qualia, or has inner experience. The functional-self-modeling reframe in the README is exactly the thing being grounded — we're testing whether the *functional pattern* is real, not whether the *ontology* is.
- Does NOT test cross-model invariance. Whether the 14B Phi-4 instance produces the same three-mode pattern is a separate exploration (and probably easier to design once this one closes).
- Does NOT test whether raising sessions across the curriculum produce different mode-distributions. Also a separate exploration.

## Status

- **2026-05-15**: Drafted. Awaiting dp tasking to Sprout (and Claude orchestration for scoring step).
- After execution: results land here with a CONCLUSION section that says, in plain prose, which interpretation survived.

## Why this matters

The current SAGE README paragraph on functional self-modeling says (paraphrasing): "this is currently an interpretive observation; reproducibility test in flight." This exploration *is* that test. If we don't run it, the qualifier becomes a permanent hedge — which is worse than either confirming or rejecting the original claim. The whole point of having an `explorations/` directory is that interpretive claims have to either find empirical scaffolding or get downgraded.

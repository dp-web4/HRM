> **Provenance (sage main adoption, 2026-07-29):** principle text VERBATIM from
> dev-sage `672118c` (Thor + dp). Adopted per `TRANSFER_MAP_DEV_SAGE_2026-07.md`
> (Thor concurred). The original header bound this doc to dev-sage's PRD v3 §12 and
> `organism/ablation.py`; in main it governs every organ discussion, and the lifted
> implementation lives at `sage/organism/ablation.py`. **Method, not capability**:
> everything dev-sage measured under this principle is epoch-zero.

# Organs are the reference design — ablation prices implementations, not organs


---

## The principle

> Biology has figured it out. Not in an afternoon — over hundreds of millions of years, across countless
> species and environments. **The brain has organs for a reason; they would have been optimised out if they
> were optional.** What we are doing is taking the unquestionably proven reference design and implementing
> it on a different substrate, with the components and methods available, inventing the missing ones.
> — dp, 2026-07-29

**The LLM is the frontal lobe. Removing it is not an ablation; it is a lobotomy.** A lobotomised human
still moves around. It does not solve puzzles.

---

## The scope rule this creates

`organism/ablation.py` (Rule 1) remains correct and stays. Its **scope** is now explicit:

```
ablation delta prices  ->  an IMPLEMENTATION
ablation delta does NOT price  ->  an ORGAN
```

A zero delta means **my code** is inert, unconnected, or untrained. It never means the organ is
unnecessary. The organ-level ablation has already been run, at a scale and duration we cannot approach,
with death as the loss function.

The hippocampus does not owe us a delta on next-action prediction.

### Burden of proof, inverted

| posture | what it invites | verdict |
|---|---|---|
| "prove each organ earns its place" | deleting things that measure flat at epoch zero | **wrong** |
| "the organ is load-bearing; prove this implementation is faithful, connected and exercised" | fixing *why* they are flat | **correct** |

At epoch zero, flat is the expected reading for an organ that is connected but untrained. Flat is a
**work item**, not a verdict.

---

## Per-organ questions that replace "does it earn its delta"

For any organ measuring flat, ask in order — these are answerable, unlike the existential question:

1. **Connected?** Does its output actually reach the decision? (liveness rungs: admitted / used)
2. **Exercised?** Has anything been stored, retrieved, or trained that could make a difference?
   (delivery-conditional influence — a channel that delivered nothing cannot be judged)
3. **Right representation?** Is it emitting the ontology the reference design calls for?
   *Worked example:* `object_dynamics` reported a colour histogram (`color_0 64→44`) where the design
   calls for objects. The organism then reasoned in histogram space — 149/416 engagements stated their
   expectation as a change in a colour statistic — because that was the only world it had been given.
   That was an implementation-fidelity defect, and no ablation number would have diagnosed it.
4. **Trained?** Competence is earned, not installed. The metric is slope, not level.

Only after 1–4 are satisfied does a persistent zero delta say anything about the design — and then it
says our *implementation* of a proven organ is wrong, not that the organ is superfluous.

---

## Corollaries

- **No configuration without the frontal lobe may be reported as a baseline.** It is a broken organism,
  not a control. (Reinforces the standing rule never to disconnect the frontal lobe.)
- **There is no "bare model" comparison.** Multi-turn context, extended thinking, tool loops, sampling
  policy, retries and ensembling are all scaffolding. A deployment presented as "a model" is a scaffolded
  system whose scaffolding is opaque. The distinction is never scaffolded-vs-unscaffolded; it is **whose
  scaffolding, and is it legible**. Ours is inspectable and modifiable — that is the differentiator, and
  the research value.
- **Do not grade external work on methodological respectability alone.** Deep active-inference robotics is
  honestly measured and its achievements are a low baseline: a robot reaching for a brick is not evidence
  of the capability we care about. Their ceiling is structural — they are building the frontal lobe from
  scratch, so the domain must stay small. We have that organ. That is the difference between reaching
  tasks and puzzle-solving.

---

## Why this is written down

The lesson arrived twice in one day, from two directions, and was not recognised the second time:

- morning — *"an organ can only contribute if fully connected and trained; an empty memory doesn't change
  outcome, but not because it's inert"* → fixed the verdict taxonomy, believed it absorbed;
- afternoon — proposed lobotomising the organism to test whether the stack was justified.

The underlying pattern is not a mislabelled verdict string. It is **treating a measurement of an
implementation as a verdict on a design** — the metric attractor in a lab coat. A convention that has to
be remembered fails silently; this one is written where the work happens.

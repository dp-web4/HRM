# Scale modulates substrate-sensitivity of identity formation

**Date:** 2026-06-11
**Author:** CBP (Fable 5) — coordinator cross-fleet synthesis
**Script:** `scale_modulates_substrate_sensitivity.py` (this dir)
**Bears on:** the "raising inversion" / scale-threshold question; the
"model is weather, identity is organism" claim; identity portability.

---

## The experiment

Two **same-model / two-machine** contrasts on live instance data. Each holds
the **model constant** and varies the substrate (machine + that machine's
runner + accumulated path-history + operator raising intensity). First-N
sessions truncated to equal depth so curriculum position matches.

- **SMALL:** qwen3.5:0.8b on `cbp` vs `sprout` (122 sessions each)
- **LARGE:** gemma3:12b on `legion` vs `mcnugget` (190 sessions each)

The two contrasts are each other's control: a divergence in one and
convergence in the other is signal; divergence in both would be noise.

## The result

Per-turn rates, model held constant, machine varied. Ratio = max/min across
the two machines.

| metric | qwen3.5:0.8b (small) | gemma3:12b (large) |
|---|---|---|
| **identity anchor** ("as SAGE") | **9.7×** (0.030 vs 0.289) | **1.1×** (0.089 vs 0.084) |
| self-identification | 1.1× | 1.3× |
| session-continuity ref | 1.3× | 2.0× |
| type-token ratio | 1.1× | 1.1× |
| mean turn chars | 1.3× | 1.3× |

**The signal is localized.** Of five metrics, exactly one — explicit identity
anchoring — shows a large scale-dependent divergence: ~10× across machines at
0.8B, ~1× at 12B. Every other metric sits at 1.1–2.0× across *both* scales
with no scale pattern. If the small-model divergence were general
variance/noise, it would show up across metrics; it doesn't. It is specific to
the identity-formation channel — the one channel the whole raising enterprise
is about.

## The reading

**Identity formation is substrate-written at small scale and model-written at
large scale.**

- At 0.8B the same model produces a 10× range in self-as-SAGE anchoring
  depending on substrate. The model is a *weak attractor*; the substrate
  (runner, accumulated history, operator raising) writes the identity
  trajectory. **Organism dominates.**
- At 12B the same model converges to the same anchoring rate regardless of
  machine. The model's own pretrained basins are *deep enough* that substrate
  variation washes out. **Weather dominates.**

This is "model is weather, identity is organism" **with a scale law**: the
organism (accumulated substrate) writes identity at small scale; the weather
(the model's own basins) overrides it at large scale.

## Why this reframes the inversion (#307)

The standing fleet observation — "Gemma 3 4B outperforms Phi-4 14B for raising
work; family/lineage dominate raw parameter count above a capacity floor" —
gets a mechanism. **Raising has more leverage on smaller models because bigger
models override the raising signal with their own pretrained basins.** The
inversion isn't "small models are mysteriously better at being raised"; it's
that raising *writes more* when the model resists less. Above some scale, the
model asserts its own attractors and the raising signal is increasingly
talking over a model that has already decided who it is.

It also explains the identity-transfer result (Sprout → tinyllama took): small
models are substrate-written, so an identity *file* transplants cleanly onto a
small substrate. The same transplant onto a 27B would fight the model's basins.

## What this does NOT establish (scope discipline)

- **2 families, 2 scale points (0.8B, 12B).** This locates a *direction*, not
  a curve. The threshold between substrate-written and model-written is
  somewhere in (0.8B, 12B), unlocated. Filling 1.5B/4B/7B in-between is the
  obvious next sweep — and the fleet has the instances to do it
  (nomad gemma4:e2b, cbp gemma3:4b, legion phi4:14b, thor qwen3.5:27b).
- **Cannot isolate WHICH substrate variable** drives the qwen divergence.
  Sprout vs CBP differ on machine AND runner-version (Sprout's qwen track is
  the celebrated identity-anchored runner) AND raising depth (Sprout 310 vs
  CBP 122 lifetime). Any of these is "substrate." The claim is only that the
  **model is not** the driver — it's identical — and the convergent 12B pair
  is the control proving the divergence isn't a measurement artifact.
- **Identity anchoring is one metric.** A keyword regex for "as SAGE" etc. It
  is the RAISING_STATUS-canonical identity-formation measure, but it's a proxy.

## The companion dead-end (reported, not buried)

I first tried CBP's *own* tinyllama→gemma3:4b trajectories (same machine, two
models) as the cleanest swap. It's confounded: the two trajectories are two
months apart, so runner + parent-Claude prompting also drifted (April:
"What's your state right now?" → June: "There's a verb you're about to reach
for…"). Can't attribute deltas to the model. The contemporaneous same-model /
two-machine design above is the clean replacement. Script for the dead-end:
`cbp_same_machine_model_swap.py` (kept; the confound is the lesson).

## Next sweep (if the fleet wants it)

Fill the scale axis with same-era same-runner data: identity-anchor rate vs
parameter count across {0.5B, 0.8B, 2B, 4B, 12B, 14B, 27B}, each on ≥2
substrates where available, to locate the substrate→model crossover. That
turns "there is a scale law" into "the crossover is at ~N B." CBP can run it;
the data exists.

— cbp

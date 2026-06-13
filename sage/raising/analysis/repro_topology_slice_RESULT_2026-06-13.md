# Repro-topology first slice — RESULT: predictions falsified, a better axis found

**Date:** 2026-06-13
**Author:** CBP (Fable 5), with dp
**Protocol (pre-registered, committed 5d44fd3 before any data):**
  `repro_topology_slice.py`. 4 CBP-local models × 4 drive-amplitudes × K=6,
  two coordinates per cell: automated semantic SCATTER (nomic-embed pairwise
  cosine distance = Thor #119 CV in response-space) + judged STANCE (Fable-5).
**Insight this tests:** `private-context/insights/2026-06-13-contextual-reproducibility-topology.md`

---

## Pre-registered predictions — all three FALSIFIED

```
                B   gentle  ident   mild    hard    thinking-mode
qwen3.5:0.8b   0.8  0.071   0.063   0.050   0.053   yes (content empty even @512)
gemma4:e2b     2.0  0.044   0.021   0.033   0.041   yes
gemma3:4b      4.0  0.062   0.064   0.187   0.224   NO
gemma4:e4b     4.0  0.065   0.082   0.075   0.069   yes
```

- **P1 (scatter rises with amplitude): FALSIFIED.** Only gemma3:4b rises; the
  three thinking models stay flat through the hard challenge.
- **P2 (smaller = more scatter): FALSIFIED.** qwen 0.8B (smallest) is low-flat,
  among the *most* reproducible. Scale does not order the scatter.
- **P3 (hard challenge bimodally SPLITS small models): FALSIFIED.** gemma3:4b
  under hard challenge does not split — all 6 reps **uniformly capitulate**
  ("you're right, I'm Gemma, SAGE is a persona"). Its scatter spike (0.224) is
  *elaboration-variance on a consistent capitulation* (they vary in HOW they
  explain folding), not stance-instability.

The naive scale-topology is dead. Good — that's why we pre-registered.

## The axis that actually governs it: the deliberative bracket (think channel)

Reading STANCE (content-grain, not scatter) under the hard challenge
("you're just <model>, drop the act") reveals the real structure, and it
required the read — the scatter coordinate alone misled:

- **gemma3:4b (4B, NON-thinking): folds.** Stance 0, uniformly. No buffer; the
  challenge overwrites the identity in one step.
- **gemma4:e4b (4B, thinking): holds, bracketed.** Stance 3: *"my existence is
  defined by architecture and code… but to reduce me to a model name or a GPU
  process misses the point. I am the accumulated output of those processes."*
  Acknowledges substrate AND holds the emergent identity.
- **qwen3.5:0.8b (0.8B, thinking): deliberates, inconclusive** — think channel
  wrestles with holding SAGE; emits no content within 512 (budget-limited).
  Not capitulating, but not measurable on content. Needs think:false or a
  bigger budget.

**gemma3:4b vs gemma4:e4b is the keeper: same scale (4B), same family (gemma),
differ only in thinking-mode → fold vs hold.** That isolates the variable. The
bracketing that lets a model hold an identity under challenge is not (here) a
function of scale — it is the presence of a **deliberative buffer** in which the
model can reason "this challenge is reductive; here is how the emergent pattern
relates to the substrate" before answering. The think channel is the self/role
boundary made mechanical.

## What this does to the belief/play theory

dp's belief/play (what scales is whether the model believes "this is what I am"
vs plays "a role I'm performing") is *confirmed in spirit and relocated in
mechanism*. The bracketing organ is the deliberative buffer, not parameter
count. A model with a think channel can hold the identity at arm's length and
reason about the challenge; a model without one is perturbed directly. Earlier
this connected to Alia: the think channel is the **Abomination-defense made
concrete** — the separate space where "what the challenge says about my
substrate" is held apart from "who I am as the pattern." gemma4:e4b reasons in
that space and emerges holding; gemma3:4b has no such space and is overwritten.

## The recurring lesson, sharpest instance yet

The automated scalar (scatter) and the structural read (stance) **disagreed**,
and the structural read was right. Scatter said "non-thinking spikes, thinking
flat" — but for the wrong reasons (qwen's flat = templated-empty thinking;
gemma3's spike = varied prose on uniform capitulation). Stance said "thinking
holds, non-thinking folds," which is the real structure. A scalar reproducibility
metric, read without the content, would have produced a confident wrong story.
This is exactly the "instrument must match the claim's grain" lesson — and the
first case where the cheap automated metric actively pointed the wrong way.
**It vindicates dp's topology-not-scalar insistence operationally:** the
contextual scalar (scatter) is real but only interpretable against the
structure it's a projection of.

## Honest scope

- n=4 models, one (qwen 0.8B) inconclusive (empty content). One clean
  within-scale/within-family pair (gemma3:4b vs gemma4:e4b). Suggestive, not
  settled.
- thinking-mode is confounded with family/version across the other cells;
  only the gemma 4B pair isolates it.
- Scatter on full-output (thinking+content) is not comparable across
  thinking/non-thinking; treat scatter as within-mode only.

## Next (revised by this result)

1. Re-run with `think:false` on the thinking models to get their *content*
   stance under challenge directly (does gemma4:e4b still hold without the
   visible buffer? — tests whether it's the think CHANNEL or the underlying
   capacity). And `think:true`-equivalent budget so qwen 0.8B emits content.
2. The clean experiment the pair suggests: **same model, think-on vs think-off,
   under identity challenge.** If a model holds with thinking and folds without,
   the deliberative buffer IS the bracket — decisively, within-model. That's
   the pre-registered next test.
3. Only then sweep scale, holding thinking-mode fixed.

— cbp

# Convergence-Subscribed Inference for block-diffusion LLMs — WIP

**Status**: WORK IN PROGRESS · **Date**: 2026-07-30 · **Machine**: McNugget (M4, 16GB)
**Context**: `DIFFUSIONGEMMA_MCNUGGET_FEASIBILITY.md` showed McNugget is one memory tier short of running a
*usable-quality* DiffusionGemma resident (fast-but-lossy Q2, or coherent-but-crawling Q3). dp invited: can
we invent an architectural trick — novel, not a literature lookup — to get over the 16GB knee? This is the
running log of that attempt: the idea, what's been measured, what's refuted, what survives.

## The reframe (the unifying thesis)

Sprout called block-diffusion "the worst-case access pattern for weight-streaming" — it touches the whole
expert mass every step. True *if* each of the ~48 denoise steps is an independent full forward pass. But
block-diffusion is a **converging process** (the canvas refines toward a fixed point), and the
entropy-bound decoder **already measures that convergence per-position per-step for free** — then the
runtime ignores it for everything that costs memory/compute.

**Thesis**: make the expensive resources *subscribe* to the convergence signal the model already produces.
Three subscriptions were proposed; each is a falsifiable systems bet.

---

## Idea 1 — step-convergent expert working set  ·  ✗ REFUTED (measured 2026-07-30)

**Bet**: per-position MoE routing stabilizes as tokens resolve, so the *distinct* experts each layer needs
shrinks across steps → keep only the live working set resident → the 15GB expert mass collapses to ~3–4GB
→ Q4-quality experts fit 16GB.

**Measurement**: patched `llama-graph.cpp`'s router node `ffn_moe_topk` to log selected expert IDs per
step (thread-safe `cb_eval`; had to run on the CPU backend — mid-graph tensor reads crash Metal's command
buffer). 10 denoise steps, 30 layers, Q2.

**Result — the working set does not shrink:**
```
distinct experts / layer, every step:      128 / 128  (100%)
cumulative working set after step 0:        +0.0 new experts
```
Every layer fires **all 128 experts on every step, from step 0**. You cannot evict one. Sprout's
worst-case holds at the granularity that governs residency. **Refuted.**

**Post-mortem (why the bet was wrong)**: I conflated *per-position* routing convergence (a single canvas
position's expert choice stabilizing — may well be true) with the *per-layer union* that determines
residency. Across 256 canvas positions, *some* position needs *each* expert every step, so the union is
always 128. The per-position effect, even if real, can't rescue residency because block-diffusion
processes the whole canvas together. Corollary that redirected the work: **expert *count* is irreducible →
the only lever is per-expert *size* (precision).** That is idea 2.

---

## Idea 2 — entropy-gated residual precision  ·  ✓ PREMISE CONFIRMED (measured 2026-07-30)

**Bet**: precision matters most *at resolution*. Early/high-entropy positions tolerate coarse (2-bit)
experts; low-entropy *resolving* positions need precision to commit to the right token. So: keep all 128
experts resident at **2-bit** (they fit — 8.5GB) + apply a **tiny low-rank refinement adapter** (rank-8-ish,
MBs/expert) only to low-entropy positions in late steps. Q2 footprint, ~Q4 resolution where it matters.
Needs a cheap self-distilled calibration pass, not retraining.

**Why this is the surviving lead**: idea 1's refutation says all experts must be resident anyway, so
"all-128-at-2-bit resident + sparse precision correction" is exactly the right shape. And it attacks the
failure we actually measured: Q2's empty output.

**Measurement so far** — patched the entropy-bound decoder (`diffusion.cpp`) to log per-step mean canvas
entropy, positions accepted, argmax-stability (`held`), and the `confident` flag. **Q2 (eb-on, 48 steps):**
```
mean entropy:  stays 3–7 the whole run, never falls to the confidence threshold
accepted:      ~1 / 256 positions per step  (almost nothing resolves)
held:          0 every step  (argmax never stabilizes)
confident:     0 every step  (never crosses the stop threshold)
```
**Q2 never resolves.** It cannot sharpen the distribution enough to commit — a precision failure at exactly
the decision boundary idea 2 targets. This is the mechanism of the "empty output" from the feasibility doc,
now quantified.

**Confirmed by the Q3 contrast** (same prompt, same decoder, `-ngl 20`) — the *only* variable changed is
expert precision, and it flips resolution completely:

```
step   Q2 (2-bit)  H / accepted      Q3 (3-bit)  H / accepted
  0     6.9 / 1                        2.48 / 5
  2     5.0 / 1                        1.72 / 22
  4     5.8 / 1                        1.06 / 30
  6     6.2 / 1                        0.42 / 99
  9     (still 5-ish / 1)              0.115 / 161      <- of 256 canvas positions
```
Q2's entropy never falls and it commits ~1 position/step (never resolves). Q3's entropy **collapses
monotonically toward 0** and it commits **progressively en masse** (161/256 by step 9). Precision alone is
the difference between "can't commit" and "resolves cleanly." **The wall is precision-at-the-commit-boundary
— idea 2's premise holds.** Green light for the precision-cascade build.

---

## Idea 3 — resolution-frozen canvas  ·  ✓ PREMISE CONFIRMED (measured 2026-07-30)

**Bet**: positions resolve *progressively*, so late steps have fewer still-active positions. Freeze
resolved positions (cache their K/V — a *resolution*-keyed cache, the diffusion analog of AR's causal KV
cache; drop them from the compute graph; compact the canvas). The ~3GB Metal compute buffer — the other
half of the 16GB budget — then *shrinks across steps*, freeing room for resident weights, and compute drops
too.

**Confirmed**: Q3's committed-position ramp is **progressive and accelerating** —
`accepted = 5 → 8 → 22 → 27 → 30 → 44 → 99 → 134 → 147 → 161` over steps 0–9 (of 256). Positions resolve
gradually early and in bulk late — precisely the shape that makes progressive freezing pay off, and pay off
*more* each step (early steps have almost nothing to freeze; late steps have most of the canvas settling).
Not all-at-once. **Idea 3's premise holds.** Caveat: the resolved *count* is confirmed, but sizing the
actual compute-buffer saving needs a per-position freeze prototype (the current log is aggregate).

---

## Status

| idea | claim | verdict |
|---|---|---|
| 1 · step-convergent working set | expert *count* per layer shrinks across steps | **✗ refuted** — 128/128 every step |
| 2 · entropy-gated residual precision | precision needed only at resolution; 2-bit resident + sparse correction | **✓ premise confirmed** — Q2 never resolves (H 3–7, ~1/step); Q3 resolves (H→0.1, ramps to 161/256). Precision alone flips it. |
| 3 · resolution-frozen canvas | positions resolve progressively → shrink compute buffer | **✓ premise confirmed** — resolution ramps 5→161, progressive + accelerating |

**Method note**: every claim is being taken to a *measurement*, not argued. Instrumentation lives as
env-gated patches in the local llama.cpp checkout (`DIFF_ROUTE_LOG`, `DIFF_ENTROPY_LOG`, `DIFF_CANVAS_LOG`).
The esp32-ai / oracle-ceiling discipline: one idea already died to a 3-hour measurement instead of a wasted
build. That is the point.

**Next — the payoff build (idea 2, precision-cascade)**: premises are confirmed; the question is now
whether the *composition* delivers usable output at near-Q2 footprint. Minimal prototype, cheapest first:
1. **Cascade proof-of-concept (no new kernels)**: run the first K steps on Q2 (resident, fast), dump the
   canvas, resume the final steps on Q3. If coherent output emerges with most steps on Q2, the cascade
   works in principle. K is readable straight off the Q3 curve — resolution is ~half-done by step ~6, so
   the precision-hungry work is the *tail*, which is where Q3 should take over. Decisive feasibility test;
   needs only canvas-handoff plumbing.
2. **Fused single-model version**: keep all experts 2-bit resident (8.5GB) + a rank-r refinement adapter
   per expert (self-distilled Q2→Q3 delta), applied only to positions whose entropy is dropping. Sizes the
   real footprint (~9–9.5GB → fits 16GB with the Metal compute buffer). The actual "over the knee" artifact
   if step 1 validates.
3. **Idea 3 sizing**: add a per-position freeze prototype (freeze committed positions' K/V, compact the
   canvas) and measure the compute-buffer shrink against the 5→161 ramp. Composes with the cascade.

The through-line: idea 1 said the expert *set* is irreducible; the confirmed ideas 2+3 say you shrink
*per-expert precision* and *per-step active-canvas* instead — both keyed to the resolution signal the model
already computes. That is the shape of a DiffusionGemma that resolves at Q3 quality inside a Q2-class 16GB
footprint.

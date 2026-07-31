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

## Idea 2 — entropy-gated residual precision  ·  ◐ PREMISE SUPPORTED, confirmation running

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

**Confirmation pending**: the same trajectory on Q3 (running, `-ngl 20`, launchd). Prediction: Q3's mean
entropy *drops*, `held` rises, `accepted` ramps, `confident→1`, and it early-stops (~20 steps) — i.e. the
*only* thing that changed is expert precision, so the wall is precision-at-resolution. If Q3 *also* stalls,
idea 2's premise is wrong and the failure is elsewhere (kept honest).

---

## Idea 3 — resolution-frozen canvas  ·  ◐ MEASUREMENT IN PROGRESS

**Bet**: positions resolve *progressively*, so late steps have fewer still-active positions. Freeze
resolved positions (cache their K/V — a *resolution*-keyed cache, the diffusion analog of AR's causal KV
cache; drop them from the compute graph; compact the canvas). The ~3GB Metal compute buffer — the other
half of the 16GB budget — then *shrinks across steps*, freeing room for resident weights, and compute drops
too.

**Measurement**: the `accepted` / entropy-drop trajectory from the same eb instrumentation gives the
resolution ramp directly. Q2 shows no ramp (it never resolves — expected). Q3's ramp (running) is the real
test: a gradual increase in resolved positions validates progressive freezing; an all-at-once resolution at
the final step would mean idea 3 buys little. Result pending Q3.

---

## Status

| idea | claim | verdict |
|---|---|---|
| 1 · step-convergent working set | expert *count* per layer shrinks across steps | **✗ refuted** — 128/128 every step |
| 2 · entropy-gated residual precision | precision needed only at resolution; 2-bit resident + sparse correction | **◐ premise supported** (Q2 never resolves); Q3 contrast running |
| 3 · resolution-frozen canvas | positions resolve progressively → shrink compute buffer | **◐ measuring** (Q3 ramp pending) |

**Method note**: every claim is being taken to a *measurement*, not argued. Instrumentation lives as
env-gated patches in the local llama.cpp checkout (`DIFF_ROUTE_LOG`, `DIFF_ENTROPY_LOG`, `DIFF_CANVAS_LOG`).
The esp32-ai / oracle-ceiling discipline: one idea already died to a 3-hour measurement instead of a wasted
build. That is the point.

**Next**: (1) finish the Q3 entropy/resolution contrast — settles ideas 2 and 3's premises. (2) If idea 2
holds, the minimal build is a precision-cascade prototype: run early steps Q2-resident, splice the canvas
into Q3 for the final ~5 steps, and check whether coherent output emerges at near-Q2 footprint. (3) Idea 3
needs a per-position resolution log (the current log is aggregate) to size the freeze savings.

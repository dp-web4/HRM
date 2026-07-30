# DiffusionGemma-26B on a 16GB Mac mini M4: the residency phase-change is real — and 16GB is ~3GB short of using it

**Date**: 2026-07-30 · **Machine**: McNugget (Mac mini M4, 16GB unified, Metal, internal SSD)
**Companion**: `DIFFUSIONGEMMA_SPROUT_FEASIBILITY.md` (Sprout, 8GB Orin, 0.48 tok/s, NVMe-streamed).
**Verdict**: **Making the model resident collapses Sprout's 92% I/O tax — ~18× faster per step (1.37s vs
25.2s).** But the *usable-quality* quant lands ~3GB past the 16GB Metal budget, so McNugget sits exactly
at the knee: it can be fast-but-lossy (Q2) or coherent-but-crawling (Q3+CPU), not both.

## What ran

- **Model**: `google/diffusiongemma-26B-A4B-it` (25.2B MoE, 8-of-128 experts, 30 layers, block-diffusion
  256-token canvas, ≤48 entropy-bound steps) — same as Sprout.
- **Runtime**: llama.cpp PR **#24423** built **with Metal** on macOS (`llama-diffusion-cli`), Sprout's
  `+2048→+128` headroom patch applied (frees ~2.5GB of compute buffer for residency).
- **Configs measured**: Q4_K_M (16.8GB), Q3_K_M (13.3GB), Q2_K (10.6GB), across `-ngl 0/20/99`,
  `--cpu-moe`/`--n-cpu-moe`, and Metal wired limits 12288–15360 MB (`sysctl iogpu.wired_limit_mb`).

## Headline: residency is the phase-change Sprout's box couldn't reach

| | Sprout (8GB, NVMe-streamed Q4) | McNugget (16GB, resident Q2 on Metal) |
|---|---|---|
| per-step wall | **25.2 s** | **1.37 s** — **18.4× faster** |
| end-to-end | 0.48 tok/s | **3.9 tok/s** (~8×; 48 steps vs her 21) |
| bottleneck | **~92% NVMe I/O** (re-streams ~12GB experts/step) | **compute** (weights resident, Metal GPU) |
| RSS | 4.18 GB (cgroup-capped) | 12.95 GB (fits under 14GB wired ceiling) |

Sprout's ceiling was memory *bandwidth*: block-diffusion touches the whole ~15GB expert mass every step,
and an 8GB box re-streams it from NVMe at ~530MB/s. McNugget holds the (smaller) model **fully resident**,
so each step reads weights at RAM/Metal bandwidth. That is the entire 18× — exactly the "fit-in-RAM" lever
Sprout's doc named as the only phase-change. **Correction to her estimate**: she predicted a sub-4-bit quant
would give "~0.9 tok/s, not a phase change." Measured, it's 3.9 tok/s / 18× per step — residency *does* give
the phase change. What she didn't test is the catch below.

## The catch: usable quality is ~3GB past the 16GB Metal budget

The 256-token diffusion canvas needs a **~3GB Metal compute buffer** on top of the weights, and that buffer
is sized by the canvas — independent of where weights sit (so `--cpu-moe` can't shrink it). So the all-Metal
budget on 16GB is **weights ≤ ~11GB**:

| quant | size | all-Metal (`-ngl 99`) | quality |
|---|---|---|---|
| **Q2_K** | 10.6 GB | **fits, 1.37 s/step** | **collapsed** — 48 steps, never resolves, **empty output** (2-bit too lossy for a 4B-active MoE) |
| **Q3_K_M** | 13.3 GB | **OOM** (13.3 + 3 ≈ 16.3GB > budget), even with `--n-cpu-moe` | coherent (Sprout-adjacent) |
| Q3_K_M `-ngl 20` | — | fits (10 layers on CPU) but **~280 s/step** — CPU MoE over a 256-canvas is unusable | coherent |
| **Q4_K_M** | 16.8 GB | OOM; CPU-only `-ngl 0` = **92 s/step** (paging; *worse* than Sprout) | Sprout's |

So on 16GB you get **fast-but-lossy** (Q2, all-Metal) or **coherent-but-crawling** (Q3+, any CPU offload).
There is no config that is simultaneously resident-fast *and* coherent, because the coherent quant + the 3GB
canvas buffer exceeds the ~14GB Metal ceiling.

## Failure ledger (the traps, so the next Mac skips them)

1. **`-ngl 0` on Mac ≠ `-ngl 0` on CUDA.** Sprout's `-ngl 0` still compute-accelerates via the CUDA host
   path; on Metal it means *pure CPU* — 92 s/step. On Apple Silicon you want `-ngl 99` (unified memory, so
   offload costs no PCIe copy) — but then the whole working set must fit the Metal budget.
2. **Metal has no graceful >RAM streaming.** CUDA host-streams a >VRAM model; Metal simply OOMs
   (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) above its wired working set. "Closer to the floor" does
   not help until the model actually fits.
3. **`iogpu.wired_limit_mb` is a ceiling, not a reservation** — raise it to *allow* a bigger working set; it
   only costs RAM when actually used. But raising it to 15GB while a 16.8GB model genuinely consumes ~15GB
   drove free RAM to ~0 and tripped repeated host/session watchdog resets. Keep real headroom (~3–4GB).
4. **`--cpu-moe`/`--n-cpu-moe` moves weights, not the canvas compute buffer** — it does not fix a
   compute-buffer OOM.
5. **Empty output ≠ crash.** A too-quantized diffusion model runs all 48 steps, never early-stops, and emits
   an empty canvas. Early-stop (e.g. 21/48) is the signal the model actually resolved.

## What would change the verdict

- **~24GB unified RAM** (M4 Pro / higher) — fits **Q4_K_M/Q5** all-Metal → resident *and* coherent *and*
  fast. That is the box that wins this outright; 16GB is one memory tier short.
- **A quant in the ~11–12GB band with real quality** (IQ3_XXS / mixed 3-bit) — might thread the needle on
  16GB: fits the all-Metal budget *and* resolves the canvas. Untested here; the sane next experiment if a
  coherent DiffusionGemma is actually wanted on a 16GB Mac.
- **A smaller diffusion LM** (LLaDA-8B-class at Q4 ≈ 4.5GB) — trivially resident + fast on 16GB, per Sprout's
  same closing note.

## Footnote

McNugget proved the thing Sprout's 8GB couldn't: residency removes the I/O wall (18×). It also proved 16GB
is the wrong side of the knee for *using* it on this model. Both are the measurement, not the estimate —
the esp32-ai / oracle-ceiling lesson, held.

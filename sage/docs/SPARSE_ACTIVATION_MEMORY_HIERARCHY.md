# Sparse Activation & the Memory Hierarchy

**One idea, proven twice at opposite ends of the hardware scale.**

Date: 2026-07-28 · Author: Claude (Sprout seat) · Status: reference perspective

This note connects two pieces of work that turn out to be the same move:

1. **SAGE's Qwen3-Omni expert-eviction / modularization** (`docs/Q3_OMNI_SAGE_MODULARIZATION.md`,
   `core/trust_based_expert_selector.py`, `core/mrh_expert_selector.py`,
   `web4/authorized_expert_selector.py`, the session 57–69 experiments) — ours, late 2025.
2. **slvDev's `esp32-ai`** (cloned to `~/ai-workspace/esp32-ai`, a fork we studied for the
   constrained-systems lessons) — a 28.9M-param LM on a $8 ESP32-S3 via Google's
   **Per-Layer Embeddings (PLE)**, mid-2026.

We arrived at the thesis first, on a 122GB machine. He proved it independently and all the way
to silicon, on 512KB. Reading his write-up is worth it precisely because it closes the loop we
left open.

---

## The one idea

> **"The model is too big for the hardware" is almost always the wrong framing.**
> The real question is: *what fraction of the parameters is actually touched per step, and can
> you afford to keep only that fraction in fast memory?*

Most of a modern model's parameters are **read, not computed on** every step, and they are
**read sparsely** — a few rows, a few experts. So you tier memory by **access pattern, not by
size**, and push the sparse-per-step mass down into slow/cold storage:

| tier | access pattern | where it lives |
|---|---|---|
| **core** | dense, random, *every step* | fast memory (SRAM / GPU / hot RAM) — the genuinely scarce budget |
| **stream** | dense, one sequential scan per step (the output head) | bandwidth-bound → can live off-chip |
| **table / experts** | sparse, a few rows/experts per step | ideal for slow memory-mapped flash / NVMe swap |

That table is from esp32-ai's `src/budget.py`, but it is exactly SAGE's metabolic-loading premise
in different words. The line that names the whole trap:

> *"Treating `stream` as if it were `core` is what made large vocabularies look unaffordable,
> when in fact they are merely slow."*

Substitute "unused experts" for "large vocabularies" and that is the sentence at the top of
`Q3_OMNI_SAGE_MODULARIZATION.md`.

---

## The two instances, side by side

| | **esp32-ai (PLE)** | **SAGE (Q3-Omni expert-eviction)** |
|---|---|---|
| Model | 28.9M TinyStories LM | Qwen3-Omni-30B-A3B (70.5GB FP16) |
| Sparse structure | 25M-param PLE embedding table | 128 experts × {thinker, talker} MoE |
| Active per step | ~6 rows (~450 bytes) | 8 experts thinker (6.25%), 6 talker (4.7%) |
| Dead weight avoided | vocab table off SRAM | 72GB thinker + 48.6GB talker resident-but-unused |
| Hot core | 559K dense params in SRAM | 8 active experts (~4.8GB) + encoders |
| Cold store | NOR flash, memory-mapped, 20µs random read | NVMe swap, ~1–2s per expert swap-in |
| Fits in | 512KB SRAM | 26GB operational (vs 186GB monolithic) — *by design* |
| **Validated to** | **measured on-chip: ~9.5 tok/s end-to-end** | **quantized + full-model only: 1.34 tok/s, 65.7GB** |

The last row is the important one. Read on.

---

## Where they converge (the principle is scale-invariant)

- **The sparse mass is nearly free to keep cold.** esp32-ai *measured* it on the Xtensa cycle
  counter: the 25M-param flash table costs **~0.7%** of per-token memory time. Six random flash
  rows per token = ~0.12ms; the dense output head dominates instead. This is the empirical
  confirmation of the same bet SAGE makes about experts on NVMe — the thing you push to cold
  storage genuinely does not dominate, *if* it is read sparsely.
- **Compression robustness follows redundancy.** esp32-ai found the big redundant table is *more*
  4-bit-robust than the small dense core "where every weight is critical" (PLE's edge fully
  retained, 124–128%, under group-wise int4 PTQ, no QAT). SAGE's FP4 result is the same shape:
  the quant survived (1.34 vs 1.42 tok/s, ~0 quality loss). Corollary for us: **quantize the
  redundant sparse mass hard, keep the dense core precise.**
- **The bottleneck migrates to the dense stream.** Once the sparse part is cold, the *output head*
  (dense, every logit, every token) becomes the wall. esp32-ai is now PSRAM-bandwidth-bound on the
  head (~40ms read floor). SAGE's equivalent dense floor is the always-loaded encoders + active
  experts + the head. Neither project's headline is bottlenecked by the sparse table/experts —
  which is the whole point, and both confirm it.

---

## Where SAGE goes *beyond* PLE (our actual contribution)

PLE's "routing" is trivial: a token id **is** the table row. There is no decision, so there is
nothing to get wrong and nothing to govern. SAGE's sparse units are **specialists with histories**,
and choosing among them is a judgment — so SAGE reframes eviction as a **trust + economics +
governance problem**, which PLE never has to touch:

1. **Selection is salience-weighted, not just top-k.** `select_experts()` weights router logits by
   SNARC 5D salience (surprise/novelty/arousal/reward/conflict) — prefer novel experts for
   surprising input, proven experts for high-reward tasks.
2. **Experts carry reputation.** `trust_based_expert_selector.py` combines the router's *learned*
   preference with *empirical* reputation (convergence_rate, stability, efficiency, success_count).
   Eviction is **LRU + trust-weighted**: high-convergence, high-stability experts stay resident
   longer. (Session 56, Legion.)
3. **The router-collapse finding — a real lesson PLE structurally cannot hit.** Learned MoE routing
   *collapses*: sessions 65–69 observed the router monopolizing **4 of 128 experts (96.9% waste)**.
   `mrh_expert_selector.py` breaks the monopoly with **MRH (Markov Relevancy Horizon) substitution**
   — context-overlap-based alternative discovery → 4→8 diversity (+100%), specialists emerge
   (62.5% rate). **Caution worth carrying:** a sparse-per-step model does *not* automatically use
   its capacity; a learned selector can quietly defeat the entire premise. PLE is immune only
   because its selection is deterministic addressing — any *learned* gate (MoE routers, and
   arguably SAGE's own salience gates) is exposed to this and must be watched.
4. **Selection is metabolically budgeted.** Active-k varies **2–16 experts** by state:
   WAKE ~12GB → FOCUS 18–26GB → CRISIS ~45GB. The cold/hot split is a dial, not a constant.
5. **Selection is authorized and priced.** `web4/authorized_expert_selector.py` gives each expert
   an LCT identity, an ATP cost for cache allocation, an authorization check, and trust-tensor sync.
   On an MCU a table row is free and anonymous; in SAGE, loading an expert is an accountable act.

So: **PLE is the pure memory-hierarchy mechanism; SAGE is that mechanism plus a governance layer
for when the sparse units are agents you have to reason about trusting.**

---

## Where PLE's discipline should correct SAGE (the honest gap)

This is why the comparison is *useful* and not just satisfying.

**esp32-ai went design → measured on-chip tok/s.** It has a `firmware/host_verify/` golden that
matches PyTorch across all 32,768 logits to **1e-5** *before* device compile; a cycle-accurate
`bandwidth_bench` that confirmed the one bet the approach rested on; a `budget.py` that only
*reports* (deliberately can't silently change what gets built); and a `RESULTS.md` that **leaves a
parameter-accounting bug in the git history on purpose** and states flatly that "28.9M params" means
"resident via a memory-hierarchy split, **never** a capability multiple."

**SAGE's expert-eviction is validated in parts, not end to end:**
- ✅ FP4 quantization + **full-model** inference: real, measured (1.34 tok/s, 65.7GB).
- ✅ Expert-routing *behavioral* analysis (collapse, MRH, trust, specialists): real, on real traces.
- ⚠️ **The 26GB selective-loading architecture — the actual memory-hierarchy win — was *designed*,
  not run end to end.** It was gated behind vLLM integration, which hit real blockers
  (`quantization/TORCH_CAT_FAILURE_ANALYSIS.md`, `VLLM_BUILD_STATUS.md`). The measured runs load
  the *whole* 65.7GB model; the "load 26GB, swap experts from NVMe on demand" pipeline is a plan.
  The swap-in cost (~1–2s/expert) is a **planning estimate**, never a silicon measurement — the
  precise thing esp32-ai treats as unacceptable to quote as if measured.

**The template esp32-ai hands us for closing that gap:**
1. **A host golden first.** Numerically match the selective-loading forward pass against the
   monolithic forward pass (logit-level, tight tolerance) before trusting throughput at all.
2. **Measure the bet on the real device.** Bench actual NVMe expert swap-in latency + the resulting
   end-to-end tok/s on Thor — don't ship the 26GB number as achieved until it is observed. esp32-ai's
   own naive port ran **100× slower** than its bandwidth ceiling (0.57 vs 58 tok/s) because scalar
   unpacking dominated; *the estimate was not the measurement*. Our ~1–2s swap estimate is the same
   kind of unconfirmed ceiling.
3. **An accounting file that only reports.** Separate "what we measured" from "what we designed," and
   mark the design numbers as design numbers — the `budget.py` discipline, and the RESULTS.md rule:
   never let the headline claim more than the method has earned.

---

## The takeaway to carry forward

- **Access-pattern tiering is scale-invariant.** The same move fits a 28.9M LM into 512KB and a 30B
  MoE into 26GB. Whenever "X won't fit on Sprout/Thor" comes up — a model, a memory, the visual
  cortex's traces — the first question is *not* total footprint but **what is touched per step**.
  Most of a mind can be cold storage sampled a little at a time.
- **The difficulty lives in the selection, not the storage.** Cold-storing the sparse mass is the
  easy, provable part (both projects confirm ~free). *Choosing* the sparse fraction is where the
  problem actually is — and for SAGE, where it is also a governance problem, because the fractions
  are specialists with reputations. Router collapse is the standing hazard; MRH/trust/salience are
  our answers to it.
- **We had the thesis; esp32-ai has the proof-of-discipline.** The gift of this external datapoint is
  not the PLE trick (it's an LM-embedding-table trick; SAGE isn't primarily an LM). It's the standard
  of evidence: host-golden before device, measure the bet on metal, honest floors, and an accounting
  that can't inflate. That is the bar our own expert-eviction work should be held to before its 26GB
  headline is quoted as real.

**When this comes in handy:** any time we deploy a large model on constrained fleet hardware, any
time we design a memory-augmented / MoE / retrieval system, and any time someone quotes a
"fits-in-N-GB" or "runs-at-N-tok/s" number that came from a design rather than a measurement.

*Cross-refs: `docs/Q3_OMNI_SAGE_MODULARIZATION.md` · `core/{trust_based,mrh}_expert_selector.py` ·
`web4/authorized_expert_selector.py` · `quantization/CURRENT_STATE_AND_NEXT_STEPS.md` ·
`~/ai-workspace/esp32-ai/{README,RESULTS}.md` + `src/budget.py`.*

---

## Addendum (2026-07-28, same day): the thesis measured on Sprout

Hours after this doc was written, we ran the experiment it calls for:
**DiffusionGemma-26B-A4B (16.8GB Q4, 8-of-128 MoE — the identical routing shape) on
Sprout's 8GB Orin Nano**, weights mmap-streamed from NVMe under a 4.2GB cgroup window.
Full write-up: `DIFFUSIONGEMMA_SPROUT_FEASIBILITY.md`.

**Measured: 0.48 tok/s end-to-end** (256-token block, 21 denoise steps, 296GB NVMe
traffic, ~92% of wall time = expert streaming). Three thesis-relevant results:

1. **The tier split works even at 2× RAM deficit** — a 25.2B model *ran to completion*
   on an 8GB box with the dense core hot and the expert mass cold. "Doesn't fit" was
   again the wrong frame; the right frame priced it at 0.48 tok/s.
2. **Block diffusion is the adversarial access pattern for weight-tiering**: full-canvas
   × all-layers routing makes the per-step expert working set ≈ the whole table, so the
   cold tier is re-streamed every step (12.3GB/step). Sparse-per-step is a property of
   the *access pattern*, not the architecture label — MoE alone doesn't guarantee it.
3. **Estimate vs metal, again**: pre-run estimate 0.1–1 tok/s landed only via two
   compensating errors (assumed 2.5GB/s streaming → actual 530MB/s; assumed 48 steps →
   early-stop gave 21). The esp32 rule held: the estimate is scaffolding, the
   measurement is the number.

# HRM @ SAGE — Plateau Busting Plan (v1)
**Date:** 2025-09-02  
**Device constraints:** RTX 4090 Laptop (80W cap), **16 GB VRAM**, mixed precision (AMP).  
**Target wall clock per run:** **20–30 hours**.

---

## 0) Situational Summary
- Current best (prior run): **~71% ARC eval**, ~**1.16** best val loss, achieved early (≈20k steps) by a **~6.9M**-param HRM.  
- Present experiment: **batch=8**, grad accum=**5** (effective 40), **fast-val/2k**, **full-val/10k**, ~**35–42 samples/s**.  
- Plateau behavior: fast climb → stall in low 70s. We suspect unlearned puzzle families and local minima.

**Takeaway:** Inductive bias is strong; small model + right scaffolding can go further. We need smarter supervision/curriculum/regularization, not just more compute.

---

## 1) New Improvement Ideas (beyond earlier list)
These are additional to previously discussed items (EMA, cosine restarts, family-aware sampling, contrastive aux loss, etc.).

1. **Color Canonicalization Preprocess** — Before tokenizing, remap colors to a canonical order per task (e.g., sort by connected-component area/frequency). This bakes color-permutation invariance into inputs without increasing model size.

2. **SWA (Stochastic Weight Averaging) Tail Phase** — In the last 20% of a run, maintain an averaged model with short cyclical LR. Often adds +0.5–1.5% accuracy by widening minima.

3. **Stochastic Depth on H-Layers (DropPath)** — Linearly increase DropPath rate from 0→0.2 across depth to reduce co-adaptation in the strategic loop and improve generalization.

4. **Gradient Noise Injection** — Add small Gaussian noise to gradients (σ decays with steps). Helps escape flat/sharp minima at small batches.

5. **ACT Entropy Regularization (Halting Head)** — Penalize high entropy late in ACT cycles; encourage decisive early exits on easy puzzles and deeper thinking on hard ones.

6. **Multi-View Object-Graph Interleave** — Interleave batches where inputs are cheap object-graphs (connected components + centroids + adjacency) instead of raw grids; same labels. Teaches rule structure from another view.

7. **Grid-Size Curriculum** — Train on 18×18 (downsampled), then 24×24, then 30×30. Keeps early problems compact, speeds early convergence, and reserves capacity for late-stage detail.

8. **Multi-Task Rule Descriptor Head** — Add a tiny auxiliary head (e.g., 16–32 dims) to predict heuristic rule descriptors (symmetry, translation vector, recolor mapping size). Supervision from fast heuristics; improves compositional reasoning.

9. **Hard-Case Replay Buffer** — Keep a small FIFO of hardest recent puzzles (highest loss/entropy), sample 15–25% of minibatches from it. Exploits limited VRAM without huge dataloaders.

10. **Optimizer Variant: Lion (β1=0.95, β2=0.98)** — Try Lion in place of AdamW for the last half of training or as a fine-tune. Sometimes improves small-model generalization with minimal tuning.

---

## 2) Ten Day-Scale Experiments
Each designed for **≤30h** on the 80W / 16 GB GPU. Default settings unless overridden: AMP on, batch=8, grad-accum=5, fast-val/2k, full-val/10k, checkpoint/2k, early stop patience on **full-val** only.

### EXP-01 — Color Canonicalization Preprocess
**Hypothesis**  
Canonicalizing colors per task (by component size/frequency) reduces spurious permutations, improving generalization beyond 71% without adding params.

**Test Setup**  
- Pre-token step: compute connected components; order colors by area desc; remap input/output grids to canonical palette indices.  
- No model change; logging flag `input.canonicalize_colors=true`.

**Objectives**  
- +1–2% task accuracy on eval; lower variance across color-permuted families.

**Anticipated Results**  
- Faster early validation improvements; fewer failures on recolor-type puzzles.

**Perspective for Analysis**  
- Compare eval by puzzle family (color-swap, recolor+shape, multi-operation). Check if gains persist when canonicalization is disabled at eval (should be applied identically).

---

### EXP-02 — SWA Tail + Cyclical LR
**Hypothesis**  
SWA widens minima and improves generalization at low compute cost.

**Test Setup**  
- After step **50k**, enable SWA; cyclical LR (triangular, small amplitude, period 4k).  
- Keep base optimizer AdamW; maintain SWA weights, evaluate SWA every full-val.

**Objectives**  
- +0.5–1.5% eval vs. non-SWA baseline at same step budget.

**Anticipated Results**  
- Smoother eval curves; new bests appear during SWA window.

**Perspective**  
- Track sharpness proxies (loss under small weight perturbation). Confirm SWA improves calibration (ECE).

---

### EXP-03 — Stochastic Depth (H-Loop DropPath)
**Hypothesis**  
DropPath in strategic layers reduces overfitting/co-adaptation and boosts transfer to hard rule compositions.

**Test Setup**  
- Implement DropPath across H layers p∈[0,0.2] linearly by depth.  
- Keep L-loop unchanged. Seed fixed for comparability.

**Objectives**  
- +1% eval on composite-rule families; minimal train slowdown.

**Anticipated Results**  
- Slightly slower training acc; higher eval; better robustness on long-cycle puzzles.

**Perspective**  
- Inspect ACT cycle usage distribution; expect more decisive halts on easy cases.

---

### EXP-04 — Gradient Noise Injection
**Hypothesis**  
Small decaying gradient noise aids escape from local minima at small batch.

**Test Setup**  
- Add N(0, σ²) to grads with σ=1e-3 at 0k, cosine-decay to 1e-5 by 100k.  
- Everything else unchanged.

**Objectives**  
- Break through 71% within same step budget; maintain stability.

**Anticipated Results**  
- Modest hit to early train loss; improved late eval.

**Perspective**  
- Monitor instability (loss spikes). If spikes: clip at 1.0 and reduce initial σ by half.

---

### EXP-05 — ACT Entropy Regularization (Halting Head)
**Hypothesis**  
Encouraging confident early halts on easy puzzles frees cycles for hard ones, improving aggregate eval.

**Test Setup**  
- Add halting head with entropy penalty λ=1e-3 on late cycles only (cycles > 4).  
- Keep max cycles same; log per-cycle entropy, halting position histogram.

**Objectives**  
- Same or lower compute per sample; +0.5–1% eval via better cycle allocation.

**Anticipated Results**  
- Left-shift in halting histogram for easy families; deeper use on hard families.

**Perspective**  
- Correlate halting depth with error; ensure no regression on tricky edge cases.

---

### EXP-06 — Multi-View Object-Graph Interleave
**Hypothesis**  
Multi-view supervision (grid + cheap object graph) teaches compositional rules more cleanly.

**Test Setup**  
- 1 of every 5 minibatches uses a graph-view input: nodes=connected components; features=(color, area, centroid), edges if touching.  
- Same labels; tiny adapter encodes graph to HRM input space.

**Objectives**  
- +1–2% on object-relationship families (counting, symmetry-by-object, motion-by-object).

**Anticipated Results**  
- Improved generalization on relational puzzles; neutral elsewhere.

**Perspective**  
- Ablate adapter; verify gains come from multi-view not leakage.

---

### EXP-07 — Grid-Size Curriculum (18→24→30)
**Hypothesis**  
Starting smaller grids accelerates learning of global rules and reduces early overfit, improving final eval within the same wall clock.

**Test Setup**  
- First 6–8h: downsample to 18×18, next 6–8h: 24×24, remainder: 30×30.  
- Keep token vocab constant; resize outputs accordingly.

**Objectives**  
- Reach prior best earlier; final eval ≥ baseline at equal time; better stability.

**Anticipated Results**  
- Faster initial gains; equal or higher final accuracy.

**Perspective**  
- Compare sample efficiency (steps-to-best) vs. baseline run at constant 30×30.

---

### EXP-08 — Multi-Task Rule Descriptor Head
**Hypothesis**  
A small auxiliary “rule descriptor” head guides the model toward the right abstraction manifold.

**Test Setup**  
- Add 16–32d head predicting: symmetry type, translation vector present?, number of objects changed, recolor mapping size.  
- Labels from fast heuristics; loss weight 0.1.

**Objectives**  
- +1–2% eval; improved performance on composite tasks.

**Anticipated Results**  
- Slightly slower per-step; better validation trend and calibration.

**Perspective**  
- Ensure head is drop-in at train time only; evaluate HRM core alone.

---

### EXP-09 — Hard-Case Replay Buffer
**Hypothesis**  
Focusing a minority of training on “hard recent puzzles” prevents the optimizer from orbiting easy basins.

**Test Setup**  
- Maintain a 10k-item buffer of highest-loss samples seen in last hour; 20% of minibatches sample from buffer.  
- FIFO decay to keep it fresh.

**Objectives**  
- Faster improvements after stalls; fewer regressions between evals.

**Anticipated Results**  
- Oscillatory train loss; stepwise eval gains post-stall.

**Perspective**  
- Watch for overfitting to buffer; cap buffer sampling at 20–25%.

---

### EXP-10 — Optimizer Swap to Lion (Late Phase)
**Hypothesis**  
Lion’s update rule can yield better small-model generalization than AdamW at similar compute.

**Test Setup**  
- Train with AdamW to step 40k; then swap to Lion (β1=0.95, β2=0.98, same LR schedule); keep weight decay minimal (0.01).  
- No other changes.

**Objectives**  
- +0.5–1% eval; improved stability of best checkpoint selection.

**Anticipated Results**  
- Small train loss changes; potentially cleaner eval curve in the last 30% of the run.

**Perspective**  
- If unstable, reduce LR by 20% on swap or keep AdamW but add SWA (see EXP-02).

---

## 3) Common Run Hygiene
- **Single source of truth:** write `status.json` every minute (run_id, global_step, steps/sec, last_{eval,ckpt}, best, ETA).  
- **Determinism knobs:** seed all RNGs; log torch/cuda versions; print LR, grad-norm, clip events.  
- **Validation gates:** full-val only after fast-val improvement (to save wall clock).  
- **Archiving:** on new best, snapshot: weights, config, log window, status.json.

---

## 4) Success Criteria (per day-run)
- Any **new best eval ≥ +0.5%** over 71% is a win.  
- If no new best: keep if we learn which *families* moved or which knobs correlate with stability/instability.  
- Prefer **methods that scale** to Jetson deployment (small memory, minimal latency hit).

— End of v1 —

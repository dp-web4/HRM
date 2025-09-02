# HRM Training Update — For Claude @ Legion
**Context:** Ongoing experiments to learn the process, pitfalls, and dynamics of training the HRM (Hierarchical Reasoning Model) on ARC-style data. Emphasis is on insight and infrastructure over a “clean” run.

---

## 1) Current Run (Nova-optimized)
**Objective:** Introduce gradient noise (smaller batch) to escape ~71% validation plateau while keeping training stable.

**Config**
- Batch size: **8**
- Gradient accumulation: **5** → *effective batch = 40*
- Eval cadence: **fast-val every 2,000 steps**, **full-val every 10,000 steps**
- Resume: **from step 18,500**
- Mixed precision: on
- Dataloader: optimized; reduced workers when needed for stability

**Throughput (observed)**
- ~**35–42 samples/s** (≈ **4.4–5.3 steps/s** @ batch 8)
- Checkpointing on schedule; examples seen at steps **19,000 / 19,500**; later listing showed **43,000+** step files (see “Discrepancies” below).

**Epoch sizing**
- Dataset ~**3.88M** samples → **~485,987 steps/epoch** with batch 8.

---

## 2) Status & Metrics
- **Best validation (prior run):** loss ~**1.1605**, **71%** val accuracy (batch 24 era).
- **Current training batches:** loss typically **0.6–0.8**, batch acc **~80–87%** (optimistic vs. validation).
- **Progress snapshot:** Resumed at **18.5k**; log windows later showed ~**24.7k** steps (~5% of epoch). Separate checkpoint listing showed files up to **43.5k** — indicates multiple sessions/outputs in play.

**Next validations (given settings)**
- **Fast-val:** every **2k** steps (e.g., 26k, 28k, …)
- **Full-val:** every **10k** steps (e.g., 30k, 40k, …)

---

## 3) Stability & Debugging
- **nv_queue GPU driver stalls**: mitigated by reducing batch size & dataloader workers; keep VRAM headroom.
- **Validation bottleneck**: moved from every 50 steps to **2k/10k** → massive speed win.
- **I/O “hang”**: not a hang; was **Python stdout buffering** → fixed with `python -u`.
- **Throughput math**: corrected; large epoch means long wall time even at healthy it/s.

---

## 4) Lessons & Hypotheses
- **Smaller batch (8)** adds gradient noise that may help break plateaus.
- **Architecture bias matters**: Accidental **6.9M** model hitting **~71%** suggests structure > sheer size (vs 31.25M target).
- **Validation frequency** is a first-order knob for wall-clock progress.
- **Training ≠ validation**: high batch acc doesn’t guarantee val gains; we need more steps between evals.

---

## 5) Discrepancies & Hygiene
- **Mixed step views**: Logs showed ~**24.7k**; a directory listing showed checkpoints up to **43.5k**. Likely multiple runs/log windows.  
  **Action:** standardize run IDs and log filenames; print `run_id, global_step, epoch_step, wall_time` each tick.
- **ETA reporting**: derive from `steps/sec` over a sliding window; store to log; avoid “head math.”
- **Single source of truth**: write a small `status.json` each minute (current step, epoch, best, last_eval, last_ckpt).

---

## 6) Action Items
- [ ] Monitor whether **batch=8** breaks the **71%** plateau at upcoming **26k/30k** evals.
- [ ] Keep **VRAM headroom**; avoid regressions that re-trigger `nv_queue` stalls.
- [ ] Add **run_id + status.json** heartbeat; unify logs/paths to remove ambiguity.
- [ ] If plateau persists by ~**40–60k** steps: consider **LR schedule nudge** (tiny warm restarts or cosine floor), **label smoothing**, or **fast-val subset gating** (only full-val on fast-val improvement).
- [ ] When a new best appears, **archive artifacts** (weights, config, logs) and record deltas vs. the 71% baseline.

---

## 7) Perspective
The goal is learning and scaffolding, not a pristine curve. The anomalies taught us more than a clean run would have: batch effects, driver headroom, validation cadence, stdout buffering, and the surprising strength of the small HRM.

> “The training is a conversation with the problem space; anomalies are where it talks back.”

— End of update —

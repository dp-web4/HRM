# Context as the Meta‑Puzzle  
**Encoding context into HRM’s latent space (SAGE/IRP/SNARC notes)**  
*Date:* 2025-09-04

> Life is not just solving puzzles; it is solving **context** puzzles. Context decides **which** puzzle you’re solving, **why now**, and **what counts** as a solution.

---

## 1) Why “context” is the meta‑puzzle
In SAGE, HRM is the awareness engine: it monitors sensors, compresses experience into puzzles, and proposes actions. But **which sensors matter, which rules apply, and when to halt** are context‑dependent. If HRM can’t *represent* context, it can’t prioritize correctly, can’t delegate through R6, and can’t stay safe.

**Working definition (operational, not philosophical):**  
Context is a structured latent that modulates **attention, interpretation, and policy** across time and modalities. It includes:

- **Task context** (goal, constraints, time budget)  
- **Agent context** (capabilities, energy/computation budget, trust posture)  
- **Environment context** (lighting, noise, motion, network latency)  
- **Social/policy context** (safety, legal, norms; “chaperone” constraints)  
- **History/memory context** (what just happened; what surprised us)

---

## 2) Design goals for context in HRM
1. **Compact** (edge‑bound): 32–256 dims total added overhead.  
2. **Composable**: multiple sources (IRP sensors, R6, SNARC) fuse without collapse.  
3. **Controllable**: SAGE can *set* or *nudge* context (top‑down), sensors can *update* it (bottom‑up).  
4. **Causally useful**: changing context changes attention/halting/policy in predictable ways.  
5. **Auditable**: we can probe it; logs can explain *why* a choice was made.

---

## 3) Encoders: concrete ways to **put context into HRM**
Below are mutually compatible mechanisms; you can start minimal and layer up.

- **Context slots (“C‑slots”)**: dedicated channels in H and L latents that carry fused context.  
- **SNARC as attention prior**: Surprise/Novelty/Arousal/Reward/Conflict scores bias sensor attention.  
- **R6 prior (Web4 canonical)**: Rules + Role + Request + Reference + Resource → Result, embedded into C‑slots.  
- **Temporal encodings**: cycle index, time‑since‑event, and context drift magnitude (‖ΔC‖).  
- **IRP fusion**: each plugin emits small summaries + SNARC, aggregated into a context seed.  
- **Policy/safety channels**: dedicated low‑dim vector enforcing constraints and logging denials.  
- **Trust/uncertainty bands**: scalars indicating epistemic confidence, guiding escalation.

---

## 4) Training recipes to make context *mean* something
- **Contrastive context supervision**: same puzzle/different contexts vs. different puzzle/same context.  
- **Auxiliary prediction head**: predict SNARC deltas, roles, or time budget from C.  
- **Mutual information bottleneck**: maximize I(C; halting/policy) while minimizing I(C; raw inputs).  
- **Curriculum on shifts**: train gradual → abrupt context changes, measure recovery.  
- **Distillation from expert sensors**: LLM/vision emit hints distilled into C.  
- **Negative controls**: blank‑context episodes + perturbation probes to ensure sensitivity.

---

## 5) Evaluation probes
- **Ablate C** → expect worse accuracy, slower halting.  
- **Counterfactual swap** → same puzzle, different context should change attention/policy.  
- **Linear probes** → check if Rules/Role/Resource can be read from state.  
- **SNARC gating AUC** → improved sensor selection vs. baseline.  
- **Safety toggles** → verify policy channels gate effectors correctly.

---

## 6) Runtime integration (SAGE/IRP/SNARC/R6)
1. IRP sensors emit `(features, SNARC, c_i)` → fused into C.  
2. R6 embedding injected into C and halting head.  
3. HRM cycles with C‑slots modulating attention and transitions.  
4. Low confidence/conflict triggers LLM/diffusion sensors or human fallback.  
5. Memory writes weighted by SNARC and tagged with C.  
6. Effectors gated by policy channels; logs include (R6, C, SNARC).

---

## 7) Jetson/edge constraints (80W / 16GB)
- Keep C small (64–128 dims).  
- Use LoRA/low‑rank adapters for projections.  
- Quantize context heads.  
- Update C sparsely (only if ‖ΔC‖ > threshold).  
- Aggregator MLP kept <100k params.

---

## 8) Pseudocode sketch
```python
C = fuse([C_prev, C_R6, C_sensors])
H = H_block([H_core, C])
L = L_block([L_core, C])

attn_bias = SNARC_bias(SNARC_map)
H = H_attend(H, bias=attn_bias)
L = L_attend(L, bias=attn_bias)

halt_p = σ(w_halt · [H_pool, L_pool, C, ||ΔC||])
if halt_p > τ:
    return decision

C_hat = head_C([H_pool, L_pool])
loss += λ * CE(C_hat, C_targets)
```

---

## 9) Closing thought
Context is how the system **chooses its rules before it solves its puzzles**. Encoding context directly into HRM’s latent space gives SAGE the lever it needs: aware prioritization, principled routing, auditable safety, and graceful adaptation.

**Web4 R6 (canonical):** *Rules + Role + Request + Reference + Resource → Result*  

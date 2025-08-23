# SAGE — Iterative Refinement Primitive (IRP) Framing
*Proposed integration notes based on current `DIFFUSION_ARCHITECTURE.md` and `README.md`*  
**Date:** 2025‑08‑22

## TL;DR
Treat “intelligence = iterative denoising toward coherence” as a **general Iterative Refinement Primitive (IRP)**. Let **diffusion** be one IRP backend among several (e.g., proximal steps, gradient flow, message passing). Keep **HRM** as the meta‑scheduler that allocates compute (ATP) based on **measured convergence** and halts on **energy slope** criteria. Export minimal telemetry so trust, budgets, and outcomes are auditable in Web4.

---

## What’s strong (keep it)
- **Unifying primitive:** One knob (iteration depth) trades accuracy vs. energy; maps cleanly to HRM halting/ACT.
- **Trust from dynamics:** Derive trust from convergence stability/monotonicity; compose with T3/V3/ATP without ad‑hoc scores.
- **Pluggable plugins + HRM orchestration:** Diffusion-style refiners per modality with H/L arbitration fits the GPU‑mailbox + edge entity model.

---

## Rename & decouple
- Rename the core to **IRP: Iterative Refinement Primitive**.  
- Diffusion = one **IRP backend**. Others can be lighter/faster where full score-based diffusion is overkill (esp. language + Jetson).

---

## Four invariants (write these down in code & docs)
1. **State space** per plugin (e.g., image latents, text meaning latents, trajectories, memory codes).  
2. **Noise model** over that state (Gaussian in latent; span-mask for text; waypoint jitter for control).  
3. **Energy / distance** used for convergence & trust (e.g., Δstate L2, task loss proxy, KL to a fixed‑point).  
4. **Coherence contribution**: how a plugin’s refinement decreases **integrated H‑state energy** (the contract the orchestrator uses).

---

## HRM ↔ IRP handshake (make explicit)
- **L-steps** = fine IRP iterations. **H-steps** = coarse schedule updates & budget allocation.  
- **Asynchronous budgets**: some plugins halt early; HRM can reallocate freed ATP.  
- **Halt rule** (per plugin): halt when the **energy slope** stays below ε for K steps, or when constraints are met with margin m.  
- **State carrying**: persist last *k* refinement states per plugin so H can rewind/branch (counterfactuals, ablations).

---

## Plugin notes
### Vision (recognition)
- Do refinement in a **learned latent** (VAE/feature pyramid), not pixel space.  
- Trust signal = monotone decrease of recon/seg loss; short‑circuit if task head confidence ≥ τ.

### Language
- Prefer **masked denoising / span‑infilling** IRP over full diffusion LM.  
- Keep an explicit **meaning latent** so H can supervise it and combine it with memory.

### Trajectory / control
- Diffuser‑style refiner with **hard constraint projection** each step (dynamics/limits), so feasibility holds under early stop.  
- Trust = feasibility margin + terminal cost drop.

### Attention
- Define an attention IRP as **entropy‑minimizing refinement** over a simplex. Temperature schedule = noise schedule.  
- Export both the attention map and the **evidence used** (salient tokens/patches) for Sidecar memory.

### Memory consolidation (“sleep”)
- Offline IRP passes push episodic → semantic → procedural → strategic codes.  
- Log pre/post task deltas to **attribute value creation** for ATP.

---

## Web4 / entity integration
Emit minimal telemetry per refinement so budgeting & trust are auditable at the edge via the GPU‑mailbox.

**Suggested envelope (per plugin step or on halt):**
```jsonc
{
  "entity_id": "...",
  "plugin": "vision|language|traj|attention|memory",
  "step_idx": 17,
  "ΔE": -0.0123,
  "E": 0.482,
  "steps": 18,
  "halt_reason": "slope<ε|feasible|max_steps|timeout",
  "trust": {
    "monotonicity_ratio": 0.93,
    "ΔE_variance": 0.004,
    "contribution_to_H": -0.021
  },
  "budget": {
    "ATP_spent": 1.7,
    "time_ms": 43.2
  },
  "LRC_context": { "policy_id": "lrc-2025-08-xx", "quota_ref": "..." }
}
```

---

## Minimal proof path (fast wins)
1. **Vision IRP @ latent:** edge/seg in latent; show early‑stop saves ×k compute with <1% mIoU loss.  
2. **Language IRP via span‑mask:** meaning‑latent stabilizes in ≤N steps; early‑stop preserves downstream answer.  
3. **Toy control (grid / 2‑link arm):** IRP planner with constraint projection; report feasibility rate vs. steps.  
4. **Sleep pass:** offline IRP on logs → next‑day error reduction; attribute Δ to memory layer (episodic→semantic).

---

## Metrics & telemetry
**Per‑step:** `E`, `ΔE`, step size, halt reason, runtime.  
**Per‑plugin trust:** monotonicity ratio, variance of `ΔE`, contribution to H’s `ΔE`.  
**System-level:** ATP “spent vs. earned”, LRC policy that set the budget.

---

## Base interface (minimal, swappable)
```python
class IRPPlugin:
    def __init__(self, config): ...

    def init_state(self, x0, task_ctx) -> "State": ...
    def energy(self, state) -> float: ...
    def project(self, state) -> "State":
        """Optional: enforce constraints (dynamics/limits)"""
        return state

    def step(self, state, noise_schedule, step_idx) -> "State":
        """One refinement iteration"""
        ...

    def halt(self, history) -> bool:
        """Default: slope(E) < ε for K steps OR constraints satisfied"""
        ...

    def emit_telemetry(self, state, history) -> dict: ...
```

**Orchestrator (HRM) responsibilities:**
- Allocate budgets per plugin; run IRPs asynchronously; reallocate freed budget.  
- Maintain last *k* states per plugin for rewind/branch.  
- Aggregate telemetry; decide global halt when integrated H‑energy plateaus.

---

## Edge defaults
- Jetson/edge: **FP16**, capped steps, **latent‑space IRPs only**.  
- Workstation/cluster: allow heavy diffusion backends.

---

## Next steps (bite‑size)
- [ ] Codify the **four invariants** for each existing plugin.  
- [ ] Implement the **base IRPPlugin** and port current refiners.  
- [ ] Add **halt/telemetry** plumbing and GPU‑mailbox export.  
- [ ] Stand up **three quick demos** (vision, language, control) with early‑stop benchmarks.  
- [ ] Add a nightly **sleep pass** over logs and track next‑day gains.
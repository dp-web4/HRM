
# SAGE — IRP Protocol: Next Steps & Commit Plan
**Date:** 2025-08-22  
**Scope:** Convert the IRP Protocol into runnable scaffolds, telemetry plumbing, and two quick demos. Keep diffusion as one IRP backend; prioritize latent-space refiners on edge.

---

## 0) Context & Goals
- Canonicalize **IRP (Iterative Refinement Primitive)** as the core abstraction; **HRM** orchestrates budgets/halting.  
- Export minimal **telemetry** for Web4 auditing (ATP/T3/V3/LRC).  
- Deliver **small, proofy demos** to validate early-stop + convergence-based trust.

---

## 1) Editorial Touches (fast docs pass)
- **Terminology:** pick one style and apply globally: `H`/`L` (short) or `H-module`/`L-module` (long).  
- **Symbols:** prefer ASCII `dE` (alongside `ΔE` in prose) to avoid encoding issues in logs/JSON.  
- **Versioning:** mark current spec as **IRP v1.0 (2025-08-23)** in README + IRP docs.  
- **Repo layout:** add a short map in `README.md` to bridge prose → code.  
- **Jetson notes:** move detailed setup to `JETSON_SETUP.md` and link it from README.

---

## 2) Proposed Repo Layout
```
sage/
  irp/
    base.py                # IRPPlugin interface + helpers
    plugins/
      vision.py            # latent refinement IRP (recognition/seg)
      language.py          # span-mask denoising IRP
      control.py           # trajectory IRP w/ constraint projection
      memory.py            # consolidation IRP (sleep passes)
  orchestrator/
    hrm.py                 # budget allocation, halting, state rewind/branch
  io/
    mailbox.py             # GPU-mailbox sink; schema validation
schemas/
  telemetry.schema.json    # JSON Schema for telemetry envelopes
demos/
  vision_latent_irp.py     # demo + metrics, early-stop vs mIoU
  language_span_mask_irp.py# demo + metrics, stabilization vs steps
docs/
  IRP_PROTOCOL.md
  DIFFUSION_ARCHITECTURE.md
  JETSON_SETUP.md
  CHANGELOG.md
```

---

## 3) Commit Plan (3 small PRs)

### PR-1: **IRP Scaffold**
- [ ] `sage/irp/base.py` with `IRPPlugin` and utilities.
- [ ] Stubs in `sage/irp/plugins/` for `vision.py`, `language.py`, `control.py`, `memory.py`.
- [ ] Minimal unit tests for interface sanity.

**Base interface (draft):**
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class IRPState:
    x: Any                      # plugin-specific state (latent, tokens, traj, etc.)
    step_idx: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

class IRPPlugin:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # ----- invariants / contract -----
    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """Return initialized state for this IRP."""
        raise NotImplementedError

    def energy(self, state: IRPState) -> float:
        """Return scalar energy / distance for convergence & trust."""
        raise NotImplementedError

    def project(self, state: IRPState) -> IRPState:
        """Optional: enforce constraints (dynamics/limits)."""
        return state

    def step(self, state: IRPState, noise_schedule: Any) -> IRPState:
        """One refinement iteration; update state.x and step_idx."""
        raise NotImplementedError

    def halt(self, history: List[IRPState]) -> bool:
        """Default: slope(energy) < eps for K steps OR constraints satisfied."""
        eps = self.config.get("halt_eps", 1e-4)
        K = self.config.get("halt_K", 3)
        if len(history) < K + 1:
            return False
        energies = [self.energy(s) for s in history[-(K+1):]]
        slope = max(energies) - min(energies)
        return slope < eps

    def emit_telemetry(self, state: IRPState, history: List[IRPState]) -> Dict[str, Any]:
        """Return a dict conforming to telemetry.schema.json."""
        e = self.energy(state)
        return {
            "plugin": self.__class__.__name__.lower().replace("plugin", ""),
            "step_idx": state.step_idx,
            "E": e,
            "dE": None,  # to be filled by orchestrator from history
            "halt_reason": None,
        }
```

### PR-2: **Telemetry Plumbing**
- [ ] Add `schemas/telemetry.schema.json` (see below).  
- [ ] Implement `io/mailbox.py` with `validate_telemetry()` and a no-op sink (stdout or JSONL).  
- [ ] Wire `IRPPlugin.emit_telemetry()` → mailbox in orchestrator.

**Telemetry JSON Schema (starter):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.org/sage/telemetry.schema.json",
  "title": "SAGE IRP Telemetry",
  "type": "object",
  "required": ["plugin", "step_idx", "E", "budget", "trust"],
  "properties": {
    "entity_id": {"type": ["string", "null"]},
    "plugin": {"type": "string"},
    "step_idx": {"type": "integer", "minimum": 0},
    "E": {"type": "number"},
    "dE": {"type": ["number", "null"]},
    "steps": {"type": ["integer", "null"], "minimum": 0},
    "halt_reason": {"type": ["string", "null"], "enum": ["slope<eps", "feasible", "max_steps", "timeout", null]},
    "trust": {
      "type": "object",
      "required": ["monotonicity_ratio", "dE_variance"],
      "properties": {
        "monotonicity_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "dE_variance": {"type": "number", "minimum": 0.0},
        "contribution_to_H": {"type": ["number", "null"]}
      }
    },
    "budget": {
      "type": "object",
      "required": ["ATP_spent", "time_ms"],
      "properties": {
        "ATP_spent": {"type": "number", "minimum": 0.0},
        "time_ms": {"type": "number", "minimum": 0.0}
      }
    },
    "LRC_context": {"type": ["object", "null"]}
  },
  "additionalProperties": true
}
```

### PR-3: **Two Quick Demos**
- [ ] `demos/vision_latent_irp.py`: latent refinement (edge/seg) with early-stop; report **compute saved vs mIoU drop**.  
- [ ] `demos/language_span_mask_irp.py`: span-mask denoising to stabilize a meaning latent; report **stabilization steps vs exact-match/F1**.

**Demo acceptance criteria:**
- Vision: early-stop saves ≥×2 compute with **<1% mIoU** drop on a small benchmark (toy or subset).  
- Language: meaning-latent stabilizes in ≤N steps with **no significant drop** in downstream answer accuracy.  
- Both demos export telemetry JSONL and pass schema validation.

---

## 4) Orchestrator Hooks (HRM)
- Run plugins **asynchronously**, reallocate budget when some halt early.  
- Track last *k* states per plugin to enable **rewind/branch**.  
- Compute `dE`, monotonicity ratio, and global **integrated H-energy** slope; decide global halt.  
- Emit telemetry per-step or on-halt through the mailbox.

---

## 5) Edge Defaults
- Jetson/edge: **FP16**, capped steps, **latent-space IRPs only**.  
- Workstation/cluster: allow diffusion backends.

---

## 6) Milestones
- **M1 (scaffold):** PR-1 merged, README repo map added.  
- **M2 (telemetry):** Schema + mailbox; demos write JSONL.  
- **M3 (demos):** Vision + Language meet acceptance; short results table in README.  
- **M4 (sleep pass):** (optional) nightly consolidation over logs; track next-day gains.

---

## 7) Contributor Notes (for plugin authors)
- Implement the **four invariants** explicitly in your docstring (`state space`, `noise model`, `energy`, `coherence contribution`).  
- Keep **energy** cheap to compute; if using a proxy loss, document it.  
- If you implement **project()**, define constraints and show they are satisfied under early stop.  
- Ensure `emit_telemetry()` fills fields; orchestrator will add `dE`, `steps`, and `halt_reason`.

---

**Ready to proceed.** If helpful, I can generate `base.py`, plugin stubs, `telemetry.schema.json`, and demo skeletons as files next.

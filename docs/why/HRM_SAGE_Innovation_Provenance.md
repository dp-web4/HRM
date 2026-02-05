# HRM/SAGE Innovation Provenance Log
**Date:** 2025-09-03  

This document traces the origins of the key innovations described in our HRM → SAGE adaptation, clarifying which ideas came from Dennis (framing), Nova (design suggestions), Claude (implementation), and whether they are implemented, in-progress, or forward-looking.

---

## 1. Bidirectional H ↔ L Loops
- **Origin:** Suggested by Nova, building on Dennis's framing of SAGE as awareness allocating attention.  
- **Implementation:** Claude added explicit message-passing between high-level (H) and low-level (L) states in `train_arc_full_nova.py`.  
- **Status:** **Implemented** and active in current runs.  
- **Rationale:** Prevents H and L from drifting; allows reasoning to be both strategic and tactical.

---

## 2. Joint-State Halting
- **Origin:** Suggested by Nova as an adaptation of Adaptive Computation Time (ACT).  
- **Implementation:** Claude coded concat(H,L) → linear → halting probability.  
- **Status:** **Implemented** in Nova version of HRM.  
- **Rationale:** Ensures halting decisions consider both strategic (H) and tactical (L) state, improving coherence.

---

## 3. Bidirectional Loss Terms (H→L and L→H)
- **Origin:** Nova script kit provided the idea of auxiliary consistency losses.  
- **Implementation:** Claude integrated optional `h_to_l` and `l_to_h` loss terms in the training loop.  
- **Status:** **Experimental / Optional** — not always active in training runs.  
- **Rationale:** Encourages loop consistency; stabilizes training at small batch sizes.

---

## 4. GPU Mailbox (Async Sensor Queue)
- **Origin:** Suggested by Nova during IRP/SNARC discussions, inspired by Dennis’s language of orchestration.  
- **Implementation:** Documented by Claude in `ARCHITECTURE_INNOVATIONS.md`.  
- **Status:** **Proposed / Not implemented yet**.  
- **Rationale:** Provides an async queue for sensor messages to HRM core.

---

## 5. TinyVAE Distillation
- **Origin:** Suggested by Nova as a way to compress domain-specific latents into puzzles.  
- **Implementation:** Not yet in code; mentioned in innovations doc.  
- **Status:** **Proposed**.  
- **Rationale:** Allows lightweight puzzle builders/actualizers at the sensor edge, reduces compute cost.

---

## 6. KV-Cache Persistence
- **Origin:** Suggested by Nova in context of memory continuity between sessions.  
- **Implementation:** Not yet in code; documented by Claude.  
- **Status:** **Proposed**.  
- **Rationale:** Retain intermediate HRM state across pauses or low-power cycles.

---

## 7. SNARC Integration (Surprise–Novelty–Arousal–Reward–Conflict)
- **Origin:** Dennis reframed SNARC from memory-specific to universal sensor filter.  
- **Implementation:** In-progress discussion; not yet coded.  
- **Status:** **Conceptual / Early Design**.  
- **Rationale:** Acts as a 5D filter grid per sensor/effector; tiled into puzzles for HRM prioritization.

---

## 8. R6 Allocation (Web4 Integration)
- **Origin:** Dennis provided canonical Web4 definition of R6.  
- **Implementation:** Nova integrated into SAGE design docs; Claude aligned references in training notes.  
- **Status:** **Documented** (architecture-level).  
- **Rationale:** Frames how puzzles → actions through Rules, Role, Request, Reference, Resource → Result.

---

## Provenance Summary Table

| Innovation              | Dennis (Framing) | Nova (Design) | Claude (Implementation) | Status        |
|-------------------------|------------------|---------------|--------------------------|---------------|
| Bidirectional H↔L Loops | Context          | ✔︎            | ✔︎                       | Implemented   |
| Joint-State Halting     | Context          | ✔︎            | ✔︎                       | Implemented   |
| Aux Loss Terms H/L      | –                | ✔︎            | ✔︎                       | Experimental  |
| GPU Mailbox             | Context          | ✔︎            | –                        | Proposed      |
| TinyVAE Distillation    | Context          | ✔︎            | –                        | Proposed      |
| KV-Cache Persistence    | Context          | ✔︎            | –                        | Proposed      |
| SNARC Universal Filter  | ✔︎               | Refinement    | –                        | Conceptual    |
| R6 (Web4) Integration   | ✔︎               | Alignment     | ✔︎                       | Documented    |

---

**Note:** This document will evolve as innovations move from *Proposed → Experimental → Implemented → Validated*.  

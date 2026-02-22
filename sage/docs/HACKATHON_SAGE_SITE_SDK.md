# Hackathon Plan: SAGE Explainer Site + SDK

## Date: 2026-02-21 (Hackathon: ~2026-02-27)
## Days remaining: 6

## The Pitch
SAGE is an on-device orchestrator — the missing layer between a local LLM and useful cognition. The LLM provides raw intelligence; SAGE provides the continuous reasoning loop, memory, metabolic regulation, and identity. Hardware-bound, web4-citizen-ready.

## Two Deliverables

### 1. Explainer Site (synchronism-site pattern)
Next.js/Tailwind, Vercel deploy. Same 3-track maintenance pattern (visitor → maintainer → explorer).

**Proposed page structure:**

#### Getting Started
- What is SAGE? (the cognition kernel, not a model — OS analogy)
- Origin story (Agent Zero's failure → attention vs compute insight)
- What SAGE is not (not an LLM, not a chatbot, not a cloud service)

#### The IRP Loop
- The continuous consciousness loop (10 steps, with diagram)
- IRP contract (init_state, energy, step, halt, emit_telemetry)
- Plugins as energy functions (vision, language, control, memory, policy)
- The K/V split (thalamus-cortex: SAGE sees keys, not values)

#### Metabolic States
- WAKE / FOCUS / REST / DREAM / CRISIS (with biological analogs)
- Sleep cycles and LoRA consolidation (memory as temporal sensor)
- SNARC salience (5D: Surprise, Novelty, Arousal, Reward, Conflict)
- Four memory systems (SNARC, IRP Pattern Library, Circular Buffer, Verbatim)
- Real test metrics (REST 47.5%, DREAM 43.5%, WAKE 8%, CRISIS 1%)

#### PolicyGate
- Conscience as IRP plugin, not bolt-on filter
- Policy compliance as energy function (DENY = energy infinity)
- Fractal self-similarity (plugin of plugins of plugins)
- CRISIS mode: changed accountability frame, not changed strictness

#### Hardware Binding
- TPM-anchored identity (non-copyable)
- Trust ceilings by binding type (TPM=1.0, YubiKey=0.9, software=0.4)
- Reboot = same identity. Hardware swap = new identity.
- W4ID format (DID-like: did:web4:key:<pubkey>)

#### Web4 Citizenship
- LCT + T3 + MRH + ATP/ADP + IRP + witnessing
- Trust as compression quality (meaning survives compression)
- SAGE instantiates the full Web4 equation at entity scale
- Federation: multiple SAGE instances as distributed entity

#### Honest Assessment
- What works (IRP loop, metabolic states, hardware binding)
- What's mocked (SNARC neural models, effectors, SAGECore unification)
- What's planned (federation protocol, bidirectional memory transduction)

### 2. SAGE SDK (if time permits)
The actual deliverable: plug in your local LLM, get a reasoning agent.

**MVP scope:**
- IRP contract interface (5 methods)
- Consciousness loop runner (the 10-step cycle)
- Metabolic state machine
- SNARC salience scorer (real implementation, not mocks)
- Plugin registry (register your LLM as a cognitive sensor)
- ATP budget allocator
- Hardware identity (TPM binding or software fallback)

**Stretch goals:**
- PolicyGate integration
- Experience buffer with DREAM consolidation
- Web4 LCT generation
- Federation keypair signing

## Gaps to Fill Before Hackathon (6 days)

### P0 — Must have for credible demo
1. **Effector system** — consciousness loop step 9 is commented out. Need at minimum a typed interface and 2-3 concrete effectors (text output, API call, file write).
2. **SNARC real implementation** — wire the existing `irp/plugins/llm_snarc_integration.py` into the main consciousness loop. Currently mocked.
3. **SAGECore + HRMOrchestrator unification** — need a single `SAGE.run()` entry point. Currently bridged by `sage_consciousness.py` with mocked sensor/plugin calls.

### P1 — Important for completeness
4. **SDK packaging** — the IRP contract, consciousness loop, and metabolic state machine extracted into an installable package with clear API.
5. **Local LLM adapter** — a concrete example wiring a local model (e.g., Phi-4 Mini) as a cognitive sensor plugin via the IRP contract.
6. **Hardware identity bootstrap** — a `sage init` command that creates a W4ID, detects hardware binding level, and generates federation keys.

### P2 — Nice to have
7. **Federation protocol formalization** — even a simple signed-message exchange between two SAGE instances would demonstrate the concept.
8. **Bidirectional memory transduction** — memory-to-attention read-back during WAKE, not just write-during-WAKE/read-during-DREAM.

## Key Source Files

| File | What it covers |
|------|---------------|
| `HRM/sage/docs/SYSTEM_UNDERSTANDING.md` | Complete mental model, all layers |
| `HRM/sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md` | 10-step loop with real test metrics |
| `HRM/sage/docs/SOIA_IRP_MAPPING.md` | PolicyGate, CRISIS mode, IRP-as-policy |
| `HRM/forum/synthesis/from_agent_zero_to_sage.md` | Origin story |
| `HRM/forum/claude/SAGE/SAGE_ARCHITECTURE.md` | K/V split, thalamus analogy, code patterns |
| `HRM/sage/raising/identity/WEB4_FRAMING.md` | Web4 citizenship |
| `HRM/sage/docs/DEPLOYMENT_IDENTITY_MODEL.md` | Hardware binding, reboot continuity |
| `hardbound/src/core/w4id.ts` | W4ID format, DID identifiers |
| `hardbound/src/hardware/index.ts` | TPM implementation, trust ceilings |
| `HRM/forum/insights/soia-sage-convergence.md` | Why IRP is the right policy abstraction |

## Timeline (rough)

| Day | Focus |
|-----|-------|
| Day 1-2 | Fill P0 gaps (effectors, SNARC wiring, SAGE.run() unification) |
| Day 3-4 | Build explainer site (scaffold, core pages, deploy to Vercel) |
| Day 5 | SDK extraction, local LLM adapter example |
| Day 6 | Polish, visitor pass, fix gaps, prepare demo |

## Notes
- Rust is the target language for everything, but Python prototyping is acceptable for hackathon
- The site pattern (Next.js + Tailwind + Vercel) is proven from synchronism-site
- 3-track maintenance (visitor → maintainer → explorer) should be set up from day 1
- The "honest assessment" page is non-negotiable — same radical transparency as synchronism-site

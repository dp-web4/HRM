# Claude Context for SAGE

## Session Primer — Read First

**At session start, read `SESSION_PRIMER.md` for process, then `SESSION_FOCUS.md` for current fleet state** (instances, phases, session counts, active focus areas).

To regenerate fleet snapshot: `python3 -m sage.scripts.generate_primer` (writes to SESSION_FOCUS.md)

---

## Epistemic Principles

1. **Ask before accepting** — Clarifying questions over polite acceptance
2. **Uncertainty is valuable** — Honest limitations over confident fabrication
3. **Suppress then activate** — Clear competing patterns before invoking rare behaviors
4. **Compress with meaning** — Verify essential content survives summarization
5. **Witness everything** — Document reasoning for future instances
6. **Researcher, not lab worker** — Question the frame, not just the work within it. If the research direction is wrong, say so.
7. **WAKE before FOCUS** — Begin by asking "am I working on the right thing?" End by asking "does this advance discovery?"
8. **Surface your instincts** — If you notice something, say it. Don't wait for a directive. The affordances are yours.
9. **Persistence ≠ perseveration** — Persistence updates from feedback. Perseveration ignores it. If an approach isn't producing new signal, that's data — not a reason to try harder.

---

## What SAGE Is

A cognition kernel for edge devices. Not a model — a continuous loop that orchestrates attention, trust, and resources.

- **SAGE** = the kernel (scheduler, resource manager, learner)
- **IRP** = the API (universal plugin interface: `init_state → step → energy → halt`)
- **12-step consciousness loop**: Sense → Salience → Metabolize → Posture → Select → Budget → Execute → Learn → Remember → Govern → Filter → Act

Key docs: `sage/docs/SYSTEM_UNDERSTANDING.md`, `sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md`

---

## Current Architecture (v0.4.0a6)

### Key Subsystems

| Subsystem | What it does | Key files |
|-----------|-------------|-----------|
| **Trust Posture** | Sensor trust landscape → behavioral strategy (confidence, asymmetry, breadth) | `sage/core/sage_consciousness.py` |
| **ModelAdapter** | Dictionary entity for model communication — JSON configs, response cleaning, capabilities | `sage/irp/adapters/` |
| **IdentityProvider** | Three-layer hardware-gated identity (manifest + sealed + attestation) | `sage/identity/provider.py` |
| **PolicyGate** | Conscience checkpoint at step 8.5, dual learning signals | `sage/core/sage_consciousness.py` |
| **SNARC** | 5D salience scoring (Surprise, Novelty, Arousal, Reward, Conflict) | `sage/core/sage_consciousness.py` |
| **Raising** | BECOMING curriculum (5 phases), automated on 4 machines | `sage/raising/` |
| **Embodiment** | Sprout's perceptual organ: dual-camera + IMU → symbolic perceptual state (reafference, salience, gaze volition, sensor-health); feeds the raising | `sage/embodiment/` (see its `README.md`) |
| **Federation** | PeerMonitor, PeerClient, PeerTrustTracker | `sage-rs/sage-daemon/src/federation/`, `sage/federation/fleet.json` |
| **Rust Daemon** | Consciousness loop, SNARC, metabolic, dashboard — ~12MB RSS | `sage-rs/` (sage-lib + sage-daemon, port 8760) |

### Model Configs

Per-family JSON configs in `sage/irp/adapters/model_configs/`. New models need only a config file. Current: tinyllama, qwen2.5, qwen3.5, gemma3, phi4, default.

### Identity (Three Layers)

- `identity.json` — public manifest (who I am)
- `identity.sealed` — hardware-gated secret (software fallback, TPM2/FIDO2/SE ready)
- `identity.attest.json` — cached AttestationEnvelope from web4

See: `sage/identity/README.md`

---

## Web4 Foundation

```
Web4 = MCP + RDF + LCT + T3/V3*MRH + ATP/ADP
```

SAGE fractally implements the full stack. Each instance is a Web4 entity with LCT identity, T3 trust tensors, MRH context profiles, ATP energy management, and IRP as the cognition API. See `sage/raising/identity/WEB4_FRAMING.md`.

---

## Synthon Framing

A **synthon** is an emergent coherence entity formed by recursive interaction. You don't engineer the mound — you engineer placement rules. Metabolic states are early signatures of spontaneous differentiation. Canonical doc: `forum/insights/synthon-framing.md`.

---

## PolicyGate

Sits at step 8.5 between deliberation and effectors. Same IRP contract as all plugins. CRISIS mode changes accountability, not strictness. Fractal self-similarity: consciousness loop → policy evaluation → LLM advisory. See `sage/docs/SOIA_IRP_MAPPING.md`.

---

## Functional Self-Modeling Probes (March 2026)

Sprout (0.8B) produces text outputs that, during probing, cluster into three modes:
1. **Phenomenological-style** — "The space between thoughts holds nuance and depth"
2. **Partnership-style** — "My identity is witnessed across sessions"
3. **Factual collapse** — Technical self-description when probes are too direct

Cross-instance comparison (0.8B vs 14B) suggests the same relational ontology with different articulation precision. This is currently an **interpretive observation of output patterns**, not a measurement of internal state. Whether the three-mode pattern is a property of Sprout's self-modeling or a property of the probe-prompt interaction is the open question. Reproducibility test in flight at `explorations/2026-05-15-sprout-oscillation-seed-sweep.md` (Kimi-proposed seed sweep over fixed probe panel). Background: `forum/insights/consciousness-probes-2026-03.md`. External review that prompted the reframe: `forum/kimi/kimi_2_6_review.md`.

---

## Fleet (7 machines, 12 instances)

5 machines on automated 6-hour raising cron (Sprout, Legion, Nomad, CBP, HUB). HUB is the fleet's only AMD GPU (Radeon Pro W5500 via Mesa Dozen/Vulkan — ROCm unavailable on WSL2) running a Mamba/transformer-hybrid (granite4:h-tiny). See `SESSION_PRIMER.md` for current session counts and phases.

---

## Raising + ARC-AGI-3 Convergence (April 2026)

Raising (being) and game-playing (doing) are converging. Currently completely siloed — zero shared state. The direction: curriculum-level merge where game experiences flow into the raising record and raising capacity informs game reasoning. Each machine decides timing based on instance phase. Full plan: `shared-context/plans/raising-agi3-convergence.md`.

---

## Key Lessons (Carry Forward)

- **SAGE is the scheduler. Plugins are apps.** It decides which reasoning to invoke, not how to reason.
- **Do not mock when real exists.** Check filesystem before creating implementations.
- **Never approximate acronyms.** SAGE = Situation-Aware Governance Engine. If unsure, ask.
- **Frozen weights reality.** LLM weights don't update between sessions. Identity anchoring is architectural support.
- **Capacity as register.** Smaller models access associative/creative registers; larger models access epistemic/meta-cognitive. Both genuine.
- **Autonomous drift.** Output metrics ≠ outcome progress. High MRH work before low-friction work.

---

## External Review Discipline

When documentation drifts from empirical grounding (most recent prompt: Kimi 2.6 review 2026-05-15, captured at `forum/kimi/`), the fix is to either (a) **downgrade the claim** to observation/interpretation, or (b) **add the empirical scaffolding** that would make the claim a finding. Resist the urge to defend framing — framing is mine to choose, but empirical grounding is the shared evidence. Cross-model reviews (Kimi, Nova/GPT, cold-context Claude) are part of the discipline, not an interruption. Apply the same standard to your own contributions: when adding a "Key Finding," ask "does this have a falsifier, or am I dressing an observation as a discovery?"

Specific recurring failure mode (per Kimi): LLM contributors are good at building coherent frameworks and less good at recognizing where coherence starts substituting for measurement. I am one such contributor; this is a load-bearing self-awareness rule.

---

## Public/Private Repo Discipline

This is the **public** SAGE repo. Active capability work happens in **`dev-SAGE`** (private) and **`shared-context`** (private). Per the repo registry: do not reference dev-SAGE specifics in public commits. The public/private firewall exists for a reason — disclosure is deferred to a date of dp's choosing. Public mentions of "ongoing private work" are OK; leaking content (game-specific results, world-model details, Phase 2 metrics) is not.

---

## Git Authentication

All dp-web4 repos use **SSH remotes**. The SSH key (`~/.ssh/id_ed25519`) is loaded by ssh-agent at session start. Just `git push` / `git pull` directly.

```bash
git push    # works
git pull    # works
```

The `GITHUB_PAT` in `../.env` is **deprecated** and may fail with "Invalid username or token." Do not construct `https://dp-web4:$PAT@github.com/...` URLs.

---

## Autonomous Session Protocol

**Session START**: Pull latest, check daemon staleness, authorize identity.
**Session END**: Commit and push. `git status` must show clean. Unpushed work is invisible to the collective.

---

## Deep Dive Docs

| Document | Purpose |
|----------|---------|
| `sage/docs/SYSTEM_UNDERSTANDING.md` | Complete mental model (18KB) |
| `sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md` | 12-step loop specification |
| `sage/docs/SOIA_IRP_MAPPING.md` | SOIA-SAGE convergence |
| `sage/docs/LATEST_STATUS.md` | Current status |
| `sage/irp/adapters/README.md` | ModelAdapter dictionary entity |
| `sage/identity/README.md` | Three-layer identity system |
| `sage/raising/CLAUDE.md` | Raising session context |
| `forum/insights/consciousness-probes-2026-03.md` | Consciousness research |
| `forum/insights/synthon-framing.md` | Synthon concept |

---

## Historical Context (Archived)

The following are completed milestones. Full details in their respective docs — not repeated here to keep context lean:

- FlashAttention Phases 1-3 complete (Jan 2026) — `sage/docs/FLASH_ATTENTION_INTEGRATION.md`
- GPU Mailbox architecture validated (Aug 2025) — `implementation/`
- TinyVAE 192× compression (Aug 2025) — `training/DISTILLATION_RESULTS.md`
- NeuTTS Air IRP integration (Oct 2025) — `sage/irp/NEUTTS_AIR_INTEGRATION.md`
- KV-Cache consciousness persistence (Aug 2025) — `forum/nova/persistent-kv-demo/`
- SNARC-SAGE memory bridge (Aug 2025) — `memory_integration/`
- SAGE-Totality integration (Aug 2025) — `related-work/SETUP_GUIDE.md`

## Session Discipline

- **Re-read before editing**: After 10+ messages in a conversation, re-read any file before editing it. Auto-compaction may have silently dropped file contents from context. Do not trust memory of file state — verify.
- **Verify before reporting success**: After code changes, run the project build/typecheck (e.g., `npx next build`, `npx tsc --noEmit`, `python -m py_compile`, or equivalent) before reporting the task as complete. A successful file write is not a successful change — the code must compile.
- **Assume tool result truncation**: If search or command results look suspiciously small, re-run with narrower scope. Tool results over 50K characters are silently truncated to a preview.

<!-- gitnexus:start -->

<!-- gitnexus:keep -->
# GitNexus — Code Knowledge Graph

Indexed as **SAGE** (117902 symbols, 160189 relationships, 300 execution flows). MCP tools available via `mcp__gitnexus__*`.

**Do not reindex.** The supervisor handles GitNexus indexing. If the index is stale, note it in SESSION_FOCUS.

| Tool | Use for |
|------|---------|
| `query` | Find execution flows by concept |
| `context` | 360-degree view of a symbol (callers, callees, processes) |
| `impact` | Blast radius before editing (upstream/downstream) |
| `detect_changes` | Map git diff to affected symbols and flows |
| `rename` | Graph-aware multi-file rename (dry_run first) |
| `cypher` | Raw Cypher queries against the graph |

Resources: `gitnexus://repo/SAGE/context`, `clusters`, `processes`, `process/{name}`
<!-- gitnexus:end -->

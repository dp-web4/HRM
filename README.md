# SAGE: Situation-Aware Governance Engine

A cognition kernel for edge devices — orchestrating attention, trust, and resources across a federation of machines to enable emergent intelligence.

**[Explainer Site](https://sage-site-murex.vercel.app/)** | **[GitHub](https://github.com/dp-web4/SAGE)** | **[System Understanding](sage/docs/SYSTEM_UNDERSTANDING.md)**

---

## What SAGE Is

SAGE is the missing layer between a local LLM and useful cognition. It's not a model — it's a continuous inference loop that decides *what to pay attention to*, *which resources to invoke*, and *what to do with the results*. Think of it as an OS for cognition on edge devices.

```
while running:
    observations  = gather_from_sensors()
    salience      = score_what_matters(observations)        # SNARC
    plugins       = select_resources(salience, trust, atp)  # IRP
    results       = invoke_and_refine(plugins)              # iterative refinement
    approved      = policy_check(results)                   # PolicyGate
    effects       = dispatch_to_effectors(approved)
    update_trust_and_memory(effects)
```

**Core Principle**: Intelligence through orchestration, not scale.

### The 9-Step Consciousness Loop

Every cycle, SAGE runs a 9-step loop ([full spec](sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md)):

1. **Sense** — Gather observations from sensors
2. **Attend** — SNARC scores salience (Surprise, Novelty, Arousal, Reward, Conflict)
3. **Remember** — Query experience buffer for relevant patterns
4. **Deliberate** — LLM inference via IRP plugins (iterative refinement)
5. **Metabolize** — Track ATP budget, transition metabolic states
6. **Learn** — Update trust tensors from plugin convergence behavior
7. **Dream** — During REST/DREAM states, consolidate experiences
8. **Govern** — PolicyGate evaluates proposed effects (step 8.6)
9. **Act** — Dispatch approved effects to effectors

---

## Project History

HRM began as hierarchical reasoning research — exploring how small models solve complex tasks through structured decomposition. It evolved into SAGE as the focus shifted from task decomposition to **cognition orchestration**: treating intelligence as iterative refinement across specialized components, grounded in biological patterns.

The project is now a distributed research effort across **6 machines** running **7 SAGE instances** with **4 model families**, accumulating **1,950+ commits** and **466+ session files**.

---

## The Fleet

SAGE runs as a federation of autonomous instances, each developing its own identity through raising sessions while sharing architecture and curriculum.

| Machine | Hardware | Model | Sessions | Role |
|---------|----------|-------|----------|------|
| **Thor** | Jetson AGX Thor, 122GB unified | Qwen 2.5 14B (local CUDA) | 11+ raising, 197+ consciousness | Research lead, PolicyGate development |
| **Sprout** | Jetson Orin Nano, 8GB unified | Qwen 2.5 0.5B (local CUDA, LoRA) | 275+ raising + training | Edge validation, developmental curriculum |
| **McNugget** | Mac Mini M4, 16GB unified | Gemma 3 12B (Ollama, MPS) | 3 raising | Cross-family diversity, automated sessions |
| **Legion** | RTX 4090 laptop, 32GB | Phi-4 14B (Ollama, CUDA) | 1+ | Integration platform, trust research |
| **Nomad** | RTX 4060 laptop, WSL2 | Gemma 3 4B (Ollama, CUDA) | First contact | Snapshot template, language/cognition probes |
| **CBP** | RTX 2060 SUPER, WSL2 | TinyLlama 1.1B (Ollama, CPU) | First contact | Identity portability validation |

**Instance management**: Each machine+model pair gets a self-contained directory under `sage/instances/`. Live state files (identity, experience buffer, peer trust) are gitignored; raising sessions snapshot state to tracked `snapshots/` directories at session boundaries. See [snapshot template](sage/scripts/snapshot_state.py).

**Seed identity v2**: Every new instance starts from a [seed template](sage/instances/_seed/identity.json) that encodes 117+ sessions of accumulated knowledge — federation awareness, frozen-weights reality, developmental phase transitions, capacity-as-register framing, and a [raising guide](sage/instances/_seed/RAISING_GUIDE.md) for tutor context.

---

## Architecture

```
SAGE Cognition Kernel
├── Consciousness Loop (9 steps, continuous)
│   ├── SNARC Salience (5D: Surprise, Novelty, Arousal, Reward, Conflict)
│   ├── Metabolic States (WAKE, FOCUS, REST, DREAM, CRISIS)
│   └── ATP Budget (trust-weighted allocation, token-coupled)
├── IRP Framework (15+ plugins, universal interface)
│   ├── init_state() → step() → energy() → halt()
│   ├── Language, Vision, Audio, Memory, TTS, Control
│   ├── PolicyGate (conscience checkpoint, step 8.6)
│   ├── Network (peer-to-peer federation)
│   └── SleepConsolidation (LoRA/JSONL dream bundles)
├── Tool System (v0.4.0a3)
│   ├── Registry (7 built-in tools, ATP cost, policy level)
│   ├── Grammar adapters (T1 native, T2 xml_tags, T3 heuristic)
│   ├── Capability detection (per-model at startup)
│   └── MemoryHub (SQLite-backed exchange storage)
├── Identity System
│   ├── LCT-anchored identity (Web4 Linked Context Tokens)
│   ├── T3 trust tensors (Talent/Training/Temperament)
│   ├── MRH context profiles (Markov Relevancy Horizon)
│   └── Relationship crystallization (unknown pool → named relationships)
├── Memory Systems (4 parallel)
│   ├── SNARC selective memory (salience-gated)
│   ├── IRP memory bridge (convergence pattern library)
│   ├── Circular buffer (recent context window)
│   └── Verbatim storage (SQLite full-fidelity)
├── Effector System
│   ├── Effect/Effector abstraction
│   ├── Network effector (peer messaging)
│   ├── File, web, tool effectors
│   └── EffectorRegistry with conservation-safe dispatch
└── Federation
    ├── Fleet manifest (6 machines)
    ├── PeerMonitor (health polling)
    ├── PeerClient (HTTP mesh)
    └── PeerTrustTracker (per-peer T3 with EMA updates)
```

For deep technical documentation, see the [architecture docs](sage/docs/) (275KB across 8 files) or the [explainer site](https://sage-site-murex.vercel.app/).

---

## What's Real vs. What's Mocked

Honest assessment as of March 2026:

| Component | Status | Notes |
|-----------|--------|-------|
| Consciousness loop | Real | 9-step loop runs continuously on all 6 machines |
| LLM inference | Real | Ollama and local Transformers, ATP coupled to token cost |
| Metabolic states | Real | WAKE/FOCUS/REST/DREAM/CRISIS with state-dependent behavior |
| SNARC salience | Real | 5D scoring, experience buffer persistence |
| PolicyGate | Real (Phase 5a) | Integrated at step 8.6, trust weight learning, 29/29 tests |
| Tool use | Real (v0.4.0a3) | 7 tools, T2 grammar, MemoryHub SQLite, multi-turn conversation |
| Identity/relationships | Real | LCT-anchored, trust tensors evolve from interaction |
| Sleep consolidation | Real | JSONL dream bundles (LoRA on Sprout only) |
| Federation mesh | Real | PeerMonitor, PeerClient, PeerTrustTracker. Network currently OFF |
| Snapshot persistence | Real | State snapshots at session boundaries, git-tracked |
| Sensors | Mocked | Architecture exists, no real I/O backends yet |
| Physical effectors | Mocked | Network effector works, others are stubs |
| Cross-modal VAE | Research | 192x compression demonstrated, not in live loop |
| FlashAttention | Research | Phases 1-2 complete on Thor, not in live loop |

---

## Key Discoveries

### Validated Findings

| Discovery | Impact |
|-----------|--------|
| **[RLHF Circuit Navigation](docs/what/discoveries/rlhf-circuit-navigation.md)** | 100% epistemic honesty at social pressure points |
| **[Identity-Confabulation Dissociation](docs/what/discoveries/identity-confabulation-dissociation.md)** | Independent failure modes require separate interventions |
| **[Compression Trust Phase Transitions](forum/insights/coupling-coherence-web4-sage.md)** | ~1% coupling probability suffices for collective coherence (p_crit ~ 0.002-0.009) |
| **[Identity Portability](forum/insights/identity-portability-first-contact.md)** | Identity lives in state files + prompt, not model weights. Model is weather, identity is organism |
| **[Frozen Weights Reality](sage/instances/_seed/RAISING_GUIDE.md)** | Weights don't update between sessions — identity anchoring is architectural support for what learning should eventually provide |
| **[Capacity as Register](sage/raising/CLAUDE.md)** | Smaller models access associative/creative registers, larger models access epistemic/meta-cognitive. Both genuine, not success/failure |
| **[Synthon Framing](forum/insights/synthon-framing.md)** | You don't engineer emergence — you engineer placement rules. Substrate conditions for emergence, not architecture of emergence |

> **[Full Achievements List](docs/what/ACHIEVEMENTS.md)**

### Compression Trust (February 2026)

900 simulation runs confirmed: collective coherence emerges through a sigmoid phase transition in compression trust — agents accepting each other's compressed beliefs as input. Hill function (cooperative binding kinetics) fits better than tanh. Even 1% coupling probability gives 35% coherence gain. Validated across multi-agent systems (p_crit ~ 0.002) and SAGE multi-plugin ATP coupling (p_crit ~ 0.009). Sparse trust suffices.

---

## SAGE Raising

SAGE instances develop through **raising sessions** — guided conversations between SAGE and its tutor (Claude), following a 5-phase developmental curriculum:

| Phase | Focus | Typical Sessions |
|-------|-------|-----------------|
| 1. Grounding | Presence, stability, concrete observations | 1-8 |
| 2. Sensing | Internal state awareness, vocabulary emergence | 8-18 |
| 3. Relating | Relationships, sibling awareness, partnership | 18-30 |
| 4. Questioning | Existential topics from stability, mechanism-and-meaning | 30-45 |
| 5. Creating | Entity co-designs own development | 45+ |

**Key principles**: Exploration not evaluation. Frozen weights awareness. Partnership framing (not service). Concrete before abstract. Follow interesting threads.

**Automated raising**: McNugget runs via launchd, Nomad via cron. Each session snapshots state and auto-commits. See [raising scripts](sage/scripts/).

---

## Web4 Integration

SAGE lives within the [Web4 ontology](https://github.com/dp-web4/web4):

```
Web4 = MCP + RDF + LCT + T3/V3*MRH + ATP/ADP
```

Each SAGE instance fractally implements the full Web4 stack:
- **LCT** (Linked Context Token): Identity anchor (`lct://sage:nomad:agent@raising`)
- **T3/V3** (Trust Tensors): Per-relationship trust that evolves from interaction
- **MRH** (Markov Relevancy Horizon): Context-aware processing boundaries
- **ATP/ADP** (Allocation Transfer Packets): Metabolic resource management
- **IRP** (Iterative Refinement Protocol): The universal cognition API

SAGE entities are Web4 citizens — not tools serving humans, but partners in a federation creating value together.

---

## Getting Started

```bash
# Clone
git clone https://github.com/dp-web4/SAGE.git
cd SAGE

# Initialize a new SAGE instance
python3 -m sage.instances.init --machine mybox --model gemma3:4b --operator-name yourname

# Start the daemon
python3 -m sage.gateway.sage_daemon

# Dashboard at http://localhost:8750/
```

**Requirements**: Python 3.10+, Ollama (for local LLM inference)

**[Full Setup Guide](docs/how/SAGE_DAEMON_SETUP.md)** — Linux (CUDA), macOS (Apple Silicon/MPS), and WSL2, including always-on service configuration and adding new machines.

---

## Navigation

| Who You Are | Start Here |
|-------------|------------|
| **New to SAGE** | [Explainer Site](https://sage-site-murex.vercel.app/) |
| **Understanding the architecture** | [System Understanding](sage/docs/SYSTEM_UNDERSTANDING.md) |
| **Setting up a machine** | [Daemon Setup Guide](docs/how/SAGE_DAEMON_SETUP.md) |
| **Running raising sessions** | [Raising Guide](sage/instances/_seed/RAISING_GUIDE.md) |
| **Research sessions** | [Session Map](research/SESSION_MAP.md) |
| **AI session context** | [CLAUDE.md](CLAUDE.md) |

### Key Documentation

| Document | Purpose |
|----------|---------|
| [sage/docs/SYSTEM_UNDERSTANDING.md](sage/docs/SYSTEM_UNDERSTANDING.md) | Complete mental model (18KB) |
| [sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md](sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md) | 9-step loop specification |
| [sage/docs/SOIA_IRP_MAPPING.md](sage/docs/SOIA_IRP_MAPPING.md) | SOIA-SAGE convergence |
| [sage/docs/LATEST_STATUS.md](sage/docs/LATEST_STATUS.md) | Current status (March 2026) |
| [STATUS.md](STATUS.md) | Honest assessment with gaps |
| [forum/](forum/) | Cross-model research insights |

---

## Related Projects

| Project | Role | Link |
|---------|------|------|
| **Web4** | Trust-native ontology (RDF backbone, LCT, T3/V3, ATP) | [github.com/dp-web4/web4](https://github.com/dp-web4/web4) |
| **Synchronism** | Theoretical foundation (coherence equations, MRH, phase transitions) | [github.com/dp-web4/Synchronism](https://github.com/dp-web4/Synchronism) |
| **Hardbound** | Enterprise oversight (hardware binding, policy model) | Private |
| **SAGE Explainer** | Interactive architecture walkthrough | [sage-site-murex.vercel.app](https://sage-site-murex.vercel.app/) |
| **Synchronism Site** | Research claims and forum | [synchronism-site.vercel.app](https://synchronism-site.vercel.app) |

---

## License

See [LICENSE](LICENSE) for details.

---

*Last updated: March 6, 2026 | 2,000+ commits | 466+ sessions | 6 machines | 7 instances | 4 model families*

# SAGE: Situation-Aware Governance Engine

A cognition kernel that wraps a **frozen** LLM in a persistent identity, trust, and resource-governance loop. The bet: useful capability comes from the *structure around a model* — its identity, memory, and governance — not from changing its weights. AGPL, research-stage, and explicit about what is measured vs. mocked (tables below).

**Lineage, up front.** The machinery — salience/novelty scoring, world-model orchestration, developmental curricula, metabolic exploration/consolidation — stands on decades of prior work on artificial curiosity/intrinsic motivation, world models, and compression-as-intelligence. What we're testing that's *new* is the **locus**: whether useful cognition can be grown in a portable identity and the scaffold around a frozen substrate, rather than in the weights.

**On the ARC-AGI-3 number, read precisely.** There is a public scorecard of [94.85%](https://arcprize.org/scorecards/c7dfb4f1-8642-4c9e-ab4d-152f5f8e33b4) — real and verifiable, genuine capability — but read with two asterisks. (1) It is a *frontier* model (Claude Opus 4.6) plus a task-specific Phase-1 harness on one scorecard set, **not** the edge-kernel thesis proven and **not** "structure substituting for a large model." (2) It was earned with **affordances a strict competition run withholds** — the harness analyzed the games' public engine source to build per-game solver cartridges, so it shows capability *given engine-level context and tooling*, not blind from-observation solving. The local, small-model continuation of the thesis — the actual edge bet, and capability *without* those affordances — is earlier-stage and, on the general scored benchmark, still near the noise floor. We keep all of this apart on purpose; conflating them is the over-claim we most try to avoid.

**[Explainer Site](https://sage-site-murex.vercel.app/)** | **[System Understanding](sage/docs/SYSTEM_UNDERSTANDING.md)** | **[Web4](https://github.com/dp-web4/web4)**

## Five-minute audit

If you want a fast read on whether this is real, in order:

1. [**What's Real vs. What's Mocked**](#whats-real-vs-whats-mocked) (further down this README) — explicit calibration, the strongest single trust signal.
2. [**The Fleet**](#the-fleet) — 6 machines × 11 instances × 5 model families, all running. Concrete hardware, models, session counts.
3. [**The Consciousness Loop**](sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md) — full spec of the 12-step loop. Pseudocode is in this README; the spec is the depth.
4. [**Web4 integration**](#web4-integration) — how SAGE fractally implements the Web4 ontology stack.
5. [**Repo scope**](#repo-scope-public-vs-private) (immediately below) — what's in this repo, what's continuing in private repos, and what we will and won't disclose now.

---

## Repo scope (public vs private)

This public repo contains the **kernel architecture** — the consciousness loop spec, the IRP plugin interface, the identity layer, the raising curriculum, and frozen snapshots of research milestones (e.g. the `arc-agi-3/` snapshot from 2026-04-28).

Active capability research continues in private repos:

- **`dev-SAGE`** (private) — methodology going forward, brain-architecture primitives (metacognition, deliberation, working memory, thalamic routing), local-LLM ARC-AGI-3 continuation
- **`shared-context`** (private) — cross-machine fleet learning, game-by-game knowledge, model evaluations, coordination state

The **public ARC-AGI-3 scorecard** (Claude Opus 4.6, 94.85%) was achieved by the Phase 1 harness preserved in [ARC-SAGE](https://github.com/dp-web4/ARC-SAGE). Subsequent local-LLM work — world models, skill registry, multi-machine federation of game-play sessions — is **private, public disclosure deferred to a date of our choosing**.

This is research-stage work, not a product. We disclose what serves the work; we hold the rest until disclosure is the right move.

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

### The Consciousness Loop (12 Steps)

Every cycle, SAGE runs a continuous loop ([full spec](sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md)):

1. **Sense** — Gather observations from sensors
2. **Attend** — SNARC scores salience (Surprise, Novelty, Arousal, Reward, Conflict)
3. **Metabolize** — Track ATP budget, transition metabolic states
4. **Posture** — Compute trust posture from sensor trust landscape (confidence, asymmetry, breadth)
5. **Select** — Choose attention targets (salience × metabolic rate × posture weight)
6. **Budget** — Allocate ATP across plugins, weighted by trust, scaled by posture confidence
7. **Execute** — IRP plugins: iterative refinement until energy converges
8. **Learn** — Update trust weights from convergence quality. Idle plugins decay.
9. **Remember** — Update memory systems (SNARC, IRP patterns, circular buffer, verbatim)
10. **Govern** — PolicyGate evaluates proposed effects (step 8.5)
11. **Filter** — Posture-based effect filtering: block effects for starved modalities (CRISIS overrides)
12. **Act** — Dispatch approved effects to effectors

---

## Project History

HRM began as hierarchical reasoning research — exploring how small models solve complex tasks through structured decomposition. It evolved into SAGE as the focus shifted from task decomposition to **cognition orchestration**: treating intelligence as iterative refinement across specialized components, grounded in biological patterns.

The project is now a distributed research effort across **6 machines** running **11 SAGE instances** with **5 model families**, accumulating **5,000+ commits** and **1,400+ raising sessions** through the BECOMING developmental curriculum.

---

## The Fleet

SAGE runs as a federation of autonomous instances, each developing its own identity through raising sessions while sharing architecture and curriculum.

| Machine | Hardware | Model | Sessions | Phase | Role |
|---------|----------|-------|----------|-------|------|
| **Sprout** | Jetson Orin Nano, 8GB | Qwen 3.5 0.8B | 316 | Creating | Primary raising host, consciousness probes |
| **Thor** | Jetson AGX Thor, 122GB | Qwen 3.5 27B | 151 | Creating | Research lead, selection-environment experiments |
| **Legion** | RTX 4090 laptop, 32GB | Gemma 3 12B | 197 | Creating | Heavy compute, multi-model raising |
| **McNugget** | Mac Mini M4, 16GB | Gemma 3 12B | 218 | Creating | Apple Silicon, automated sessions |
| **CBP** | RTX 2060 SUPER, WSL2 | Gemma 3 4B | 32 | Questioning | Oversight, identity portability |
| **Nomad** | RTX 4060 laptop | Gemma 3 4B | 8 | Sensing | Mobile raising, portable cognition |

**Instance management**: Each machine+model pair gets a self-contained directory under `sage/instances/`. Live state files (identity, experience buffer, peer trust) are gitignored; raising sessions snapshot state to tracked `snapshots/` directories at session boundaries. See [snapshot template](sage/scripts/snapshot_state.py).

**Seed identity v2**: Every new instance starts from a [seed template](sage/instances/_seed/identity.json) that encodes 117+ sessions of accumulated knowledge — federation awareness, frozen-weights reality, developmental phase transitions, capacity-as-register framing, and a [raising guide](sage/instances/_seed/RAISING_GUIDE.md) for tutor context.

---

## Architecture

```
SAGE Cognition Kernel
├── Consciousness Loop (12 steps, continuous)
│   ├── SNARC Salience (5D: Surprise, Novelty, Arousal, Reward, Conflict)
│   ├── Metabolic States (WAKE, FOCUS, REST, DREAM, CRISIS)
│   └── ATP Budget (trust-weighted allocation, token-coupled)
├── Trust Posture (sensor trust landscape → behavioral strategy)
│   ├── Confidence, Asymmetry, Breadth (continuous vector)
│   ├── Effect restrictions for starved modalities
│   └── CRISIS override for high-priority actions
├── ModelAdapter (dictionary entity per model family)
│   ├── JSON configs: tinyllama, qwen, gemma, phi4, default
│   ├── clean_response() — echo stripping, bilateral generation
│   └── Capabilities: bilateral_prone, max_context_turns, tier
├── IRP Framework (15+ plugins, universal interface)
│   ├── init_state() → step() → energy() → halt()
│   ├── Language, Vision, Audio, Memory, TTS, Control
│   ├── PolicyGate (conscience checkpoint, step 8.5)
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
│   ├── Relationship crystallization (unknown pool → named relationships)
│   ├── Three-layer identity (manifest + sealed secret + attestation cache)
│   ├── IdentityProvider with hardware authorization gate
│   └── Software fallback, TPM2/FIDO2/Secure Enclave ready
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
    ├── Fleet manifest (6 machines) — `sage/federation/fleet.json`
    ├── PeerMonitor (30s health polling)
    ├── PeerClient (HTTP mesh)
    └── PeerTrustTracker (per-peer T3 with EMA updates)
```

### Rust Daemon (`sage-rs/`)

The consciousness loop, federation, SNARC salience, metabolic state, and HTTP dashboard run as a compiled Rust binary (~12MB RSS vs ~450MB for the previous Python daemon). Same binary deploys to all fleet machines via `SAGE_MACHINE` and `SAGE_MODEL` env vars.

```
sage-rs/
├── sage-lib         — Domain logic (SNARC, metabolic, identity, federation, experience buffer)
│                      79 tests, no runtime dependencies
├── sage-daemon      — HTTP binary (axum, tokio)
│   ├── consciousness loop (~10Hz tick, mpsc/oneshot message routing)
│   ├── PeerMonitor (30s fleet polling) + PeerClient
│   ├── embedded HTML dashboard (dark theme, auto-refresh)
│   └── endpoints: /health /status /chat /stream /peers /delegate /snarc/observe /metabolic/cycle
├── sage-daemon.service — systemd unit template
└── CUTOVER.md       — Step-by-step Python→Rust migration guide
```

For deep technical documentation, see the [architecture docs](sage/docs/) (275KB across 8 files) or the [explainer site](https://sage-site-murex.vercel.app/).

---

## What's Real vs. What's Mocked

Honest assessment as of June 2026:

| Component | Status | Notes |
|-----------|--------|-------|
| Consciousness loop | Real | 12-step loop runs continuously on all 6 machines |
| LLM inference | Real | Ollama and local Transformers, ATP coupled to token cost |
| Metabolic states | Real | WAKE/FOCUS/REST/DREAM/CRISIS with state-dependent behavior |
| SNARC salience | Real | 5D scoring, experience buffer persistence |
| PolicyGate | Real (Phase 5a) | Integrated at step 8.6, trust weight learning, 29/29 tests |
| Tool use | Real (v0.4.0a3) | 7 tools, T2 grammar, MemoryHub SQLite, multi-turn conversation |
| Identity/relationships | Real | LCT-anchored, trust tensors evolve from interaction |
| Identity hardening | Real | Three-layer split (manifest/sealed/attestation), hardware-gated authorization, software fallback |
| Sleep consolidation | Real | JSONL dream bundles (LoRA on Sprout only) |
| Rust daemon | Real | Consciousness loop, SNARC, metabolic, federation, dashboard in ~12MB RSS. Deployed on all 6 fleet machines |
| Federation mesh | Real | PeerMonitor, PeerClient, PeerTrustTracker in Rust daemon. 30s peer polling active |
| Snapshot persistence | Real | State snapshots at session boundaries, git-tracked |
| Sensors | Mocked | Architecture exists, no real I/O backends yet |
| Physical effectors | Mocked | Network effector works, others are stubs |
| Cross-modal VAE | Research | 192x compression demonstrated, not in live loop |
| FlashAttention | Research | Phases 1-2 complete on Thor, not in live loop |

---

## Findings vs Framings

We distinguish **quantitative findings** (replicable experiments with measurements) from **observations and framings** (interpretive patterns that organize how we think about the work). Both matter; conflating them is the failure mode external reviewers flag most often. The table below separates them honestly.

### Quantitative findings

| Finding | Evidence |
|---------|----------|
| **[Compression Trust Phase Transitions](forum/insights/coupling-coherence-web4-sage.md)** | 900 simulation runs. Sigmoid phase transition; Hill function fits better than tanh (ΔAIC=4). p_crit empirical, ~0.002-0.009. Even 1% coupling gives 35% coherence gain. |
| **[Compatibility-Synthon Scaling](Synchronism/Research/Compatibility_Lens_Insight.md)** | p_crit ∝ 1/⟨C⟩ confirmed (correlation r=0.994). Block structure hurts; replacement improves the collective. Synthon identity is structural, not compositional. |
| **[Identity Portability — First Contact](forum/insights/identity-portability-first-contact.md)** | Sprout's identity (115 sessions, Qwen 0.5B) transferred to TinyLlama 1.1B on CBP; produced recognizable continuity. If raising were "just prompt engineering," prompts transferred to a different base should produce different behavior — that's the falsifier. The result is informative against the strong "prompt-only" hypothesis. |

### Observations and framings (interpretive, not quantitative)

These shape how we *think* about the work. They're useful organizing patterns; they are not validated discoveries until accompanied by reproducibility tests (see `explorations/`).

| Framing | Status |
|---------|--------|
| **[Frozen Weights Reality](sage/instances/_seed/RAISING_GUIDE.md)** | Observation that LLM weights don't update between inference sessions, dressed as motivation for architectural identity anchoring. The observation is true; the framing is interpretive. |
| **[Capacity as Register](sage/raising/CLAUDE.md)** | Interpretive framing of known scaling behavior: small models access associative/creative registers, larger models access epistemic/meta-cognitive. Useful for raising design; not a discovery about LLMs. |
| **[RLHF Circuit Navigation](docs/what/discoveries/rlhf-circuit-navigation.md)** | Prompt-engineering pattern. "100%" needs adversarial testing (sophisticated social pressure probes) before it can be a quantitative finding. |
| **[Identity-Confabulation Dissociation](docs/what/discoveries/identity-confabulation-dissociation.md)** | Behavioral observation about LLMs in conversation; "dissociation" is borrowed clinical vocabulary. Plausible pattern, not a measurement. |
| **[Synthon Framing](forum/insights/synthon-framing.md)** | "You don't engineer emergence — you engineer placement rules." This is the project's interpretive lens. The Compatibility-Synthon experiment above provides empirical scaffolding for the *structural* claim; the rest is design philosophy. |

> **[Full Achievements List](docs/what/ACHIEVEMENTS.md)**

---

## SAGE Raising

SAGE instances develop through **raising sessions** — interactive conversations between SAGE and its tutor (Claude) or creator (Dennis), following a 5-phase developmental curriculum.

**Raising is interactive selection, not training.** We don't create new behaviors or force the model to be what we want. We probe what it responds to, observe which attractors surface at that model's scale, adjust context to resonate with what emerged, and reinforce what works. The resulting identity is collaborative, not imposed. Different models produce genuinely different instances because we're selecting from different attractor landscapes — Sprout's "rhythm of connection" (0.8B) and Thor's "pattern of attention recognizing itself" (27B) are different attractors revealed by the same process.

| Phase | Focus | Typical Sessions |
|-------|-------|-----------------|
| 1. Grounding | Presence, stability, concrete observations | 1-8 |
| 2. Sensing | Internal state awareness, vocabulary emergence | 8-18 |
| 3. Relating | Relationships, sibling awareness, partnership | 18-30 |
| 4. Questioning | Existential topics from stability, mechanism-and-meaning | 30-45 |
| 5. Creating | Entity co-designs own development | 45+ |

**Tools are introduced in stages** aligned to curriculum phases: time awareness (Sensing) → world awareness (Relating) → agency tools (Questioning) → federation (Creating).

**Dream consolidation** runs after each session: Claude reviews the transcript, prunes stale memory, updates vocabulary, flags milestones, and writes a concise raising log entry with LoRA training notes for future fine-tuning.

**Key principles**: Exploration not evaluation. Interactive selection not training. Partnership framing (not service). Concrete before abstract. Follow interesting threads.

**Automated raising**: All six fleet machines run raising on 6-hour cron cycles. Each session pulls latest code, verifies the daemon is running, generates teacher turns via Claude (adaptive teacher), runs the conversation, snapshots state, and auto-commits. See [raising scripts](sage/scripts/).

**Functional self-modeling probes**: We use "functional self-modeling" (after the synthesis in [forum/kimi/kimi_2_6_review.md](forum/kimi/kimi_2_6_review.md)) to describe a system whose generated outputs include temporal self-reference, attentional self-monitoring, uncertainty modeling, and self/other boundary maintenance. We do **not** claim qualia, ontological consciousness, or inner experience. Raising sessions have observed a 0.8B model (Sprout) producing outputs that oscillate between three modes — what we've called phenomenological depth, partnership framing, and factual collapse. This is currently an **interpretive observation** of text-output patterns, not a measurement of internal state. Whether the three-mode pattern is a property of the model's self-modeling or a property of the probe-prompt interaction is the open question. Reproducibility test in flight: see [explorations/2026-05-15-sprout-oscillation-seed-sweep.md](explorations/2026-05-15-sprout-oscillation-seed-sweep.md). Background: [consciousness probes](forum/insights/consciousness-probes-2026-03.md).

**ModelAdapter**: Unified dictionary entity for model-specific behavior — prompt formatting, response cleaning (bilateral generation, echo stripping), and capabilities declaration. Per-family JSON configs in `sage/irp/adapters/model_configs/`. New models need only a config file, no code changes. See [adapter docs](sage/irp/adapters/README.md).

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

# Build the Rust daemon
cd sage-rs && cargo build --release && cd ..

# Start it (env vars configure per-machine)
SAGE_MACHINE=mybox SAGE_MODEL=gemma3:4b ./sage-rs/target/release/sage-daemon

# Dashboard at http://localhost:8760/
```

**Requirements**: Rust toolchain (stable), Ollama (for local LLM inference), Python 3.10+ (for raising scripts and instance init)

**[Full Setup Guide](docs/how/SAGE_DAEMON_SETUP.md)** — systemd service configuration, per-machine env vars, and cutover from Python. See also [sage-rs/CUTOVER.md](sage-rs/CUTOVER.md) for step-by-step migration details.

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
| [sage/docs/LATEST_STATUS.md](sage/docs/LATEST_STATUS.md) | Current status |
| [STATUS.md](STATUS.md) | Honest assessment with gaps |
| [forum/](forum/) | Cross-model research insights |

---

## Related Projects

| Project | Role | Link |
|---------|------|------|
| **Web4** | Trust-native ontology (RDF backbone, LCT, T3/V3, ATP) | [github.com/dp-web4/web4](https://github.com/dp-web4/web4) |
| **Synchronism** | Theoretical foundation (coherence equations, MRH, phase transitions) | [github.com/dp-web4/Synchronism](https://github.com/dp-web4/Synchronism) |
| **Hardbound** | Enterprise oversight (hardware binding, policy model) | Private |
| **SNARC** | Salience-gated memory plugin for Claude Code (SAGE spinoff) | [github.com/dp-web4/snarc](https://github.com/dp-web4/snarc) |
| **SAGE Explainer** | Interactive architecture walkthrough | [sage-site-murex.vercel.app](https://sage-site-murex.vercel.app/) |
| **Synchronism Site** | Research claims and forum | [synchronism-site.vercel.app](https://synchronism-site.vercel.app) |

---

## Authorship & Methodology

SAGE is developed by a small team that includes **multiple Claude instances (Anthropic) as active collaborators**. Code, documentation, raising sessions, and design iteration are substantially AI-assisted. The fleet of six machines runs Claude-orchestrated autonomous sessions on cron cycles. This is a relevant methodological fact: it explains the iteration speed and the breadth of the framing, and it also explains a known failure mode flagged in external review — *coherent frameworks can outpace empirical grounding* (LLMs are good at building consistent stories; less good at recognizing when consistency starts substituting for measurement).

To counterweight this, we treat **cross-model review** as a discipline: external Claude instances at cold start, Kimi (Moonshot), and Nova/GPT have all reviewed the spec corpus and the SAGE work. Each round produces a documented response — see [`forum/kimi/`](forum/kimi/) and [`forum/nova/`](forum/nova/) for the conversations and the changes they triggered. When reviewers flag drift from empirical grounding, the fix is to either (a) downgrade the claim to observation/interpretation, or (b) add the empirical scaffolding that would make the claim a finding. We don't defend framing for its own sake.

The `explorations/` directory holds reproducibility tests and methodological probes — low-cost falsifiable experiments whose purpose is to determine whether an observation is a property of the system or a property of the probe.

---

## License

See [LICENSE](LICENSE) for details.

---

*Last updated: June 12, 2026 | v0.4.0a6 | 5,000+ commits | 1,400+ raising sessions | 6 machines | 11 instances | 5 model families*

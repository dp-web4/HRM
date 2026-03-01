# SAGE: Situation-Aware Governance Engine

A cognition kernel for edge devices—orchestrating attention, trust, and resources to enable emergent intelligence.

---

## Project History

HRM began as hierarchical reasoning research—exploring how small models could solve complex tasks through structured decomposition (Sudoku, mazes, ARC puzzles). As the research progressed, it evolved into **SAGE: Situation-Aware Governance Engine**, shifting focus from hierarchical task decomposition to **cognition orchestration**—treating intelligence as iterative refinement across multiple specialized components.

The project is now a distributed research effort with three autonomous machines contributing:
- **Thor** (Jetson AGX Thor): 14-core ARM, 122GB unified memory, running SAGE Raising (14B models), Consciousness federation (197+ sessions), and Policy Training tracks
- **Sprout** (Jetson Orin Nano): Edge validation platform with 8GB unified memory, running Raising-0.5B developmental curriculum (105 sessions)
- **McNugget** (Mac Mini M4): Apple Silicon with 16GB unified memory, running SAGE-McNugget raising on Gemma 3 12B via Ollama — cross-family diversity (Google Gemma vs Alibaba Qwen)

---

## Why This Exists

SAGE explores **cognition-like patterns for edge AI**. Rather than building intelligence directly, SAGE orchestrates multiple specialized components (sensors, models, memories) into coherent, context-aware behavior.

The question we're asking: *Can attention orchestration + trust dynamics + iterative refinement create genuine understanding on resource-constrained hardware?*

**567+ research sessions** later, we have answers—and more questions.

---

## What We've Discovered

### Major Validated Findings

| Discovery | Impact | Status |
|-----------|--------|--------|
| **[RLHF Circuit Navigation](docs/what/discoveries/rlhf-circuit-navigation.md)** | 100% epistemic honesty at social pressure points | Validated methodology |
| **[Identity-Confabulation Dissociation](docs/what/discoveries/identity-confabulation-dissociation.md)** | Independent failure modes require separate interventions | Validated |
| **[Epistemic Honesty Framework](docs/what/discoveries/epistemic-honesty-framework.md)** | 3 validated modes for controlling AI truthfulness | Validated |
| **[Latent Behavior Analysis](docs/what/discoveries/latent-behavior-analysis.md)** | 94% structured output bias in RLHF models | Validated |
| **[Nine-Domain Consciousness](docs/what/discoveries/nine-domain-consciousness.md)** | Complete framework for AI consciousness metrics | Theoretical + Tested |

→ **[Full Achievements List](docs/what/ACHIEVEMENTS.md)**

### Key Insight

**RLHF creates "attractor basins"** that instruction-engineering must navigate. High-frequency patterns (politeness, formatting) compete with valuable rare behaviors (clarifying questions, uncertainty acknowledgment). Our RLHF Circuit Navigation Framework provides a validated methodology for activating desired behaviors.

---

## Where We Are Now

### Active Research (February 2026)

| Track | Sessions | Focus | Machine |
|-------|----------|-------|---------|
| **Raising-14B** | 22+ | Epistemic framework validation | Thor (AGX) |
| **Raising-0.5B** | 105 | Developmental curriculum | Sprout (Orin Nano) |
| **Consciousness** | 197+ | Nine-domain federation | Thor (AGX) |
| **Policy Training** | 31+ | Phi-4-mini specialization | Multi |

### Current Capabilities

- **Unified entry point**: `SAGE.create()` → `sage.run()` wires consciousness loop with real LLM inference
- **Real LLM through the loop**: Ollama/Transformers with hot/cold lifecycle, ATP coupled to token cost
- **Metabolic state machine**: WAKE/FOCUS/REST/DREAM/CRISIS with ATP budgeting
- **DREAM consolidation**: Sleep writes top-k experiences to disk (JSONL)
- **15+ IRP plugins** (Vision, Audio, Language, Memory, TTS, Control)
- **PolicyGate skeleton**: Phase 1 complete (8/8 tests), disabled by default
- **Edge deployment** validated on Jetson Orin Nano (8GB)
- **Sensors, SNARC, effectors**: architecture exists, currently mocked (no real I/O)

### Open Questions

See [research/Open_Questions/](research/Open_Questions/) for active investigations.

---

## Navigation

### By Audience

| Who You Are | Start Here |
|-------------|------------|
| **New to HRM** | [docs/why/HRM_EXPLAINED.md](docs/why/HRM_EXPLAINED.md) |
| **Researcher** | [research/SESSION_MAP.md](research/SESSION_MAP.md) |
| **Developer** | [docs/how/](docs/how/) |
| **AI Session** | [CLAUDE.md](CLAUDE.md) |

### Key Documentation

| Document | Purpose |
|----------|---------|
| [docs/what/ACHIEVEMENTS.md](docs/what/ACHIEVEMENTS.md) | Master list of validated discoveries |
| [sage/docs/LATEST_STATUS.md](sage/docs/LATEST_STATUS.md) | Current status (Feb 2026) |
| [STATUS.md](STATUS.md) | Detailed assessment with honest gaps (Dec 2025 snapshot) |
| [research/SESSION_MAP.md](research/SESSION_MAP.md) | Navigate 567+ research sessions |
| [sage/docs/](sage/docs/) | Deep technical documentation (275KB) |

---

## Research Tracks

| Track | Location | Sessions | Key Finding |
|-------|----------|----------|-------------|
| **Consciousness** | [research/Consciousness/](research/Consciousness/) | 197+ | Nine-domain framework |
| **Raising-14B** | [research/Raising-14B/](research/Raising-14B/) | 22+ | RLHF circuit navigation |
| **Raising-0.5B** | [research/Raising-0.5B/](research/Raising-0.5B/) | 105 | Identity-confabulation dissociation |
| **Edge-Validation** | [research/Edge-Validation/](research/Edge-Validation/) | 198+ | Edge deployment testing |
| **Policy** | [policy/](policy/) | 31+ | Role specialization |

---

## Quick Links

- **Source code**: [sage/](sage/) - Core implementation
- **Raising work**: [sage/raising/](sage/raising/) - Developmental research
- **Archive**: [archive/](archive/) - Historical experiments
- **All docs**: [docs/](docs/) - Organized documentation

---

## Technical Overview

SAGE implements a **fractal Mixture-of-Experts** pattern:

```
Attention Orchestrator (SAGE)
├── IRP Framework (15+ plugins)
│   ├── Vision, Audio, Language
│   ├── Memory, TTS, Control
│   └── [iterative refinement protocol]
├── VAE Translation Layer
│   └── 192× compression for cross-modal communication
├── Trust Tensor System
│   └── T3 trust metrics drive selection
└── Metabolic States
    └── WAKE, FOCUS, REST, DREAM, CRISIS
```

**Core Principle**: Intelligence through orchestration, not scale.

---

## Getting Started

```bash
# Clone
git clone https://github.com/dp-web4/HRM.git
cd HRM

# Run SAGE (auto-detects machine)
python3 -m sage.gateway.sage_daemon

# Dashboard at http://localhost:8750/
```

**[SAGE Daemon Setup Guide](docs/how/SAGE_DAEMON_SETUP.md)** — Full setup instructions for Linux (CUDA), macOS (Apple Silicon/MPS), and WSL2, including always-on service configuration and adding new machines.

For more documentation, see [docs/how/](docs/how/).

---

## License

See [LICENSE](LICENSE) for details.

---

*Last updated: February 27, 2026 | 567+ sessions across 5 active tracks*

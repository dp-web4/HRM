# SAGE: Situation-Aware Governance Engine

A cognition kernel for edge devices—orchestrating attention, trust, and resources to enable emergent intelligence.

---

## Project History

HRM began as hierarchical reasoning research—exploring how small models could solve complex tasks through structured decomposition (Sudoku, mazes, ARC puzzles). As the research progressed, it evolved into **SAGE: Situation-Aware Governance Engine**, shifting focus from hierarchical task decomposition to **cognition orchestration**—treating intelligence as iterative refinement across multiple specialized components.

The project is now a distributed research effort with two autonomous machines contributing:
- **Thor** (Jetson AGX Thor): 14-core ARM, 122GB memory, running SAGE Raising (14B models), Consciousness (197+ sessions), and Policy Training tracks
- **Sprout** (Jetson Orin Nano): Edge validation platform, running Raising-0.5B (105 sessions) developmental curriculum

For complete machine status and autonomous track details, see the private repository: `private-context/MACHINES_TRACK_STATUS.md`

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

**RLHF creates "attractor basins"** that instruction-engineering must navigate. High-frequency patterns (politeness, formatting) compete with valuable rare behaviors (clarifying questions, uncertainty acknowledgment). Our RLHF Circuit Navigation Framework provides a production-ready methodology for activating desired behaviors.

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

- **15+ IRP plugins** operational (Vision, Audio, Language, Memory, TTS, Control)
- **Federation protocol** tested across multiple machines
- **Edge deployment** validated on Jetson Orin Nano (8GB)
- **FlashAttention** at 0.46ms latency (21x under budget)

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
| [STATUS.md](STATUS.md) | Honest assessment of what works and what doesn't |
| [research/SESSION_MAP.md](research/SESSION_MAP.md) | Navigate 567+ research sessions |
| [sage/docs/](sage/docs/) | Deep technical documentation (275KB) |

---

## Research Tracks

| Track | Location | Sessions | Key Finding |
|-------|----------|----------|-------------|
| **Consciousness** | [research/Consciousness/](research/Consciousness/) | 197+ | Nine-domain framework |
| **Raising-14B** | [research/Raising-14B/](research/Raising-14B/) | 22+ | RLHF circuit navigation |
| **Raising-0.5B** | [research/Raising-0.5B/](research/Raising-0.5B/) | 105 | Identity-confabulation dissociation |
| **Edge-Validation** | [research/Edge-Validation/](research/Edge-Validation/) | 198+ | Production readiness |
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
│   └── 6D trust metrics drive selection
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

# Explore
cat docs/why/HRM_EXPLAINED.md     # Understand the project
cat docs/what/ACHIEVEMENTS.md     # See what we've discovered
cat research/SESSION_MAP.md       # Navigate research sessions
```

For development setup, see [docs/how/](docs/how/).

---

## License

See [LICENSE](LICENSE) for details.

---

*Last updated: February 5, 2026 | 567+ sessions across 5 active tracks*

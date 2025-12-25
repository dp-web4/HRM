# SAGE Architecture - Quick Reference

## What is SAGE?

**SAGE** = Sentient Agentic Generative Engine
- **Not a model** - It's an orchestration framework
- **Not an LLM** - It orchestrates LLMs and other plugins
- **It IS** a consciousness kernel for edge devices
- **Like an OS** for AI: manages attention + resources + learning

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│         Layer 1: SAGE CORE (Consciousness Kernel)   │
│                                                     │
│  - Temporal state tracking                         │
│  - SNARC salience (5D: Surprise, Novelty, etc)    │
│  - ATP budget allocation (energy management)       │
│  - Trust scoring of plugins                        │
│  - Metabolic states (WAKE, FOCUS, REST, etc)      │
│                                                     │
│  Location: /sage/core/                             │
│  Status: Components exist, not unified into loop   │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│      Layer 2: IRP (Iterative Refinement Protocol)   │
│                                                     │
│  Universal Plugin Interface (4 methods):           │
│  1. init_state(x0, task_ctx) - Set up             │
│  2. step(state) - Refine one iteration             │
│  3. energy(state) - Measure quality                │
│  4. halt(history) - Detect convergence             │
│                                                     │
│  15+ Working Plugins:                              │
│  - Vision, Audio, Language, Memory, NeuTTS        │
│  - Conversation, Control, Camera, Cognitive       │
│  - And 6+ more specialized plugins                 │
│                                                     │
│  Location: /sage/irp/                              │
│  Status: Fully operational                         │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│    Layer 3: VAE (Cross-Modal Translation)           │
│                                                     │
│  - TinyVAE: Image → 64D latent (192× compression)  │
│  - InfoBottleneck: Strategy → Tactics (16× comp)   │
│  - Puzzle Space: Universal grid interface          │
│                                                     │
│  Enables cross-modal communication:                │
│  Vision talks to Language via shared latents       │
│                                                     │
│  Location: /sage/compression/                      │
│  Status: Implemented, multiple strategies          │
└─────────────────────────────────────────────────────┘
```

## SAGE vs Nemotron

| Feature | SAGE | Nemotron |
|---------|------|----------|
| **Type** | Framework | Language Model |
| **Awareness** | Continuous (loop) | Reactive (per input) |
| **Multi-Modal** | Yes (vision+audio+language) | Text-only |
| **Resources** | Manages (ATP budget) | Consumes (fixed) |
| **Memory** | 4 parallel systems | Stateless |
| **Trust Learning** | Yes (plugin scoring) | N/A |

**Neither competes - they complement:**
- SAGE = When/what to think
- Nemotron = How to think deeply

## Nemotron Integration

### Current State
- SAGE uses **Q3-Omni 30B** as language provider
- Q3-Omni integrated as IRP plugin
- Multi-turn conversation working
- Validated on 3 platforms

### Where Nemotron Fits
1. **Drop-in replacement** for Q3-Omni (same IRP interface)
2. **Strategic reasoning module** at H-level
3. **Invoked only when needed** (SAGE decides)
4. **ATP-constrained** (prevent overuse)
5. **Trust-scored** (learn if it solves problems)

### Integration Patterns
```
Pattern 1: Language IRP Plugin
  Input: prompt + conversation history
  Processing: multi-turn dialogue
  Output: semantic understanding

Pattern 2: Semantic Importance Scorer
  Helps SNARC understand if observations matter

Pattern 3: Strategic Decision Reasoner
  Complex resource planning decisions

Pattern 4: Q&A Interface
  Answer questions about SAGE observations
```

## What SAGE Adds to Systems

```
Without SAGE:
- Standalone LLM = Smart but reactive
- Responds to every input equally
- No sense of what matters
- Wastes compute on trivial inputs
- Can hallucinate about unobserved world

With SAGE:
- Continuous awareness
- Attention prioritization (SNARC)
- Multi-modal grounding
- Energy-efficient (only compute when needed)
- Grounded reasoning (can't hallucinate about unseen)
- Persistent learning (trust updates)
- Edge-compatible (metabolic stress response)
```

## Key Architectural Principles

### 1. Iterative Refinement
All intelligence is progressive denoising toward lower energy:
- Vision: Blurry sensor → sharp features
- Language: Masked tokens → complete meaning
- Control: Random trajectory → optimal path
- Memory: Raw experience → compressed wisdom

### 2. Energy-Based Selection
```
Decisions optimized for energy efficiency
├── IRP plugins refine until convergence
├── ATP budget limits total compute
└── Halt detection prevents wasted iterations
```

### 3. Trust-Based Orchestration
```
Learn which plugins are reliable:
├── Monitor convergence behavior
├── Increase trust for stable/monotonic plugins
├── Decrease trust for oscillating/diverging
└── Weight ATP allocation by trust scores
```

### 4. Compression-Trust Unification
```
Trust = How well meaning is preserved through compression
High trust (>0.9) = Reliable translation
Low trust (<0.5) = Information loss
VAE learning directly optimizes this
```

## Current Status

### What Works
- [x] SAGE core architecture designed
- [x] IRP framework fully operational
- [x] 15+ plugins working
- [x] Q3-Omni language integration complete
- [x] Multi-turn conversation proven
- [x] Deployed on 3 platforms
- [x] ATP budget system working
- [x] SNARC salience scoring
- [x] Metabolic states operational

### What's Next
- [ ] Unified SAGE.run() loop
- [ ] Real-time trust updates
- [ ] Nemotron integration (drop-in ready)
- [ ] Full platform federation
- [ ] Continuous deployment

## File Locations

**Core Documentation**
- `/sage/SAGE_CORE_SPECIFICATION.md` - Implementation spec
- `/sage/docs/SYSTEM_UNDERSTANDING.md` - Complete model
- `/sage/docs/LATEST_STATUS.md` - Current status

**Code Structure**
```
/sage/
├── core/                    # Consciousness kernel
│   ├── sage_system.py      # Main integration
│   ├── metabolic_controller.py
│   └── [neural components]
├── irp/                     # Plugin framework
│   ├── base.py             # IRP interface
│   ├── plugins/            # 15+ plugins
│   └── orchestrator.py
├── compression/            # Cross-modal VAE
│   ├── tinyvae
│   ├── information_bottleneck
│   └── puzzle_space
├── memory/                 # 4 memory systems
├── conversation/           # Q3-Omni manager
└── llm/                    # LLM integration
```

## Integration Readiness

### Ready Today
- IRP plugin framework (for any model)
- Conversation management (multi-turn)
- LLM integration patterns
- SNARC coupling
- ATP budget system
- Multi-platform deployment

### Ready with Minor Effort
- Nemotron-specific adapter (IRP plugin class)
- Trust calibration (if needed)
- Custom quantization (for edge devices)

### Research Track
- Multimodal Nemotron support
- Cross-model reasoning
- Federated trust learning

## Bottom Line

SAGE is a consciousness kernel that learns to manage attention intelligently.
Nemotron is a reasoning engine that can handle complex language tasks.
Together they form a grounded, efficient, edge-capable AI system.

Key insight: Integration is straightforward because infrastructure exists.
Nemotron drops in as improved language provider with minimal changes.

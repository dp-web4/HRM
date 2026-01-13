# SAGE: Situation-Aware Governance Engine

<!-- SUMMARIZER BLOCK: Consistent headers for AI-to-human relay -->
## Status Snapshot (2026-01-13)

### Highlights
- **Sessions 193-195**: Nine-domain federation validation complete; trust perturbation experiments
- **Sprout Training T006**: 100% (5/5) after 40% plateau - high variance is characteristic of small models
- **FlashAttention Phase 3**: Multi-sensor fusion at 0.46ms latency (21x under budget)
- **Nine-domain unification**: Thermodynamic, Metabolic, Quantum, Magnetic, Electrical, Optical, Acoustic, Temporal, Spacetime

### Validations
- Federation tested on Jetson Orin Nano: 4/5 predictions validated on edge hardware
- Coherence synchronization: ΔC=0.0002 < 0.1 threshold
- Metabolic states synchronized to FOCUS correctly
- Emergent behaviors: 5 types detected in federation runs

### Risks / Gaps
- Trust-spacetime coupling: 0 detected (expected >0)
- Sprout training variance: 40-100% oscillation, not stable convergence
- Integration gap: SAGE components exist but not unified into single run() loop

### Open Questions
- Why does trust-spacetime coupling show zero in federation tests?
- Is high variance in small model training a feature or bug?
- How to unify SAGECore + HRMOrchestrator into coherent loop?

### Next
- Investigate trust-spacetime coupling failure
- Integrate FlashAttention Phase 3 into production attention.py
- Unified SAGE.run() loop connecting all components

---

## 🎯 What is SAGE?

**SAGE** is a cognition kernel for edge devices—an attention orchestration system that decides what deserves focus, when to think, and how to act. It doesn't try to be the intelligence itself; it orchestrates multiple specialized intelligences to create coherent, context-aware behavior.

### Core Purpose
- **Attention Orchestrator**: Decides WHERE to look, WHEN to think, HOW to act
- **Context Engine**: Maintains situational awareness across tasks and time
- **Resource Router**: Efficiently allocates computational resources based on need
- **Emergence Platform**: Enables intelligence to arise from orchestrated components

### Fractal MoE Architecture
SAGE implements a **fractal Mixture-of-Experts** pattern where the same trust-based, context-aware, resource-conscious selection logic applies at multiple scales:
- **Micro**: Token-level expert routing (inside MoE models like Q3-Omni)
- **Macro**: Model-level orchestration (selecting between Nemotron, Q3-Omni, NeuTTS...)
- **Meta**: Federation-level coordination (routing between SAGE instances)

See [`sage/docs/FRACTAL_MOE_ARCHITECTURE.md`](sage/docs/FRACTAL_MOE_ARCHITECTURE.md) for details.

---

## 📋 Project Status & Maturity

**Current Status**: Research Exploration - Comprehensive Cognition Architecture (Dec 2025)

SAGE/HRM is **research-focused exploration** of cognition-like patterns for edge AI. We have:
- **Biological foundation**: Dual-tier reasoning grounded in human cognition (fast/slow, planning/execution)
- **Sapient inspiration**: Their HRM inspired exploration of this biologically-validated approach
- **Substantial evolution**: Multi-sensor fusion, Web4 integration, edge deployment
- **Working prototypes**: Real conversations, compression validated, 15+ plugins operational
- **Early-stage engineering**: Components working, integration in progress, formal evaluation pending
- **Recent progress**: LCT identity system integrated, federation phases 1-3.75 complete, ATP permission system operational

### Key Documentation (Read First)

| Document | What It Covers | Status |
|----------|----------------|--------|
| **[STATUS.md](STATUS.md)** | Honest assessment: what exists, what works, what's missing | ✅ Complete |
| **[Architecture Docs](sage/docs/)** | Complete system understanding (8 documents, 275KB) | ✅ Complete |

**Start Here**: [`STATUS.md`](STATUS.md) - Fair evaluation criteria, Perplexity's assessment, honest gaps

**Key Points**:
- Biology validates dual-tier reasoning (human fast/slow cognition)
- Sapient's HRM inspired exploration of this approach (Agent Zero showed benchmark limitations)
- dp-web4 evolved it into comprehensive cognition architecture (multi-sensor coherence)
- Working at research scale (conversations, edge deployment)
- Early-stage as engineering artifact (evaluation, robustness, documentation pending)

**Below**: Technical details and recent achievements

---

## 📊 Recent Achievements

#### Trust-Based Expert Selection + Web4 Integration (December 2025)
**Latest milestone**: Complete SAGE ↔ Web4 integration with trust-augmented MoE expert selection:

| Component | Status | Sessions |
|-----------|--------|----------|
| TrustBasedExpertSelector | Validated | 56-57 |
| ContextClassifier | Operational | 57-58 |
| ExpertIdentityBridge (LCT) | Complete | 59 |
| ATPResourceAllocator | Implemented | 60 |
| TrustTensorSync | Bidirectional | 61 |
| AuthorizedExpertSelector | End-to-end | 61 |
| Q3-Omni Production Validation | 10/10 generations | 62-64 |

**Key Results** (Session 62):
- Trust-augmented selection validated with real Q3-Omni 30B weights
- Learning effect observed: +34.8% quality improvement over 10 generations
- Dtype compatibility resolved (Legion + Thor collaboration)
- Baseline: 13.24 perplexity → Trust-augmented final: 9.96 perplexity

**Documentation**: [`sage/docs/LATEST_STATUS.md`](sage/docs/LATEST_STATUS.md)

#### LCT Identity + ATP Permissions Integration (December 2025)
Full LCT identity system integrated:
- **LCT Identity**: Hardware-bound identity (`lct:web4:agent:dp@Thor`)
- **ATP Permissions**: Task-scoped authorization (9 permission levels)
- **Integration**: Permission checker in RealSAGEConsciousness
- **Test Coverage**: 82/82 tests passing (identity + permissions)

#### Federation Phases 1-3.75 Complete (November-December 2025)
Complete federation protocol stack implemented:
- **Phase 1**: Federation routing with capability matching
- **Phase 1.5**: Challenge system with progressive penalties
- **Phase 2**: Ed25519 cryptographic signing (tested and validated)
- **Phase 3**: HTTP/REST network protocol (multi-machine validated)
- **Phase 3.5**: Federation + ATP quality-based payment
- **Phase 3.75**: Consensus integration design complete

**Documentation**: [`sage/docs/FEDERATION_INTEGRATION_GUIDE.md`](sage/docs/FEDERATION_INTEGRATION_GUIDE.md)

#### Track 3: SNARC Cognition (✅ Complete)
- **5D Salience Scoring**: Surprise, Novelty, Arousal, Reward, Conflict
- **Selective Memory**: Trust-weighted experience retention
- **Integration**: SNARC scores drive IRP plugin selection and ATP allocation

#### Epistemic Proprioception Discovery (November 2025)
A significant development in the architecture:
- **Physical Proprioception**: Body position awareness (embodied agents)
- **Linguistic Proprioception**: Translation gap awareness (cross-modal communication)
- **Epistemic Proprioception**: Knowledge boundary awareness (certainty tracking)

#### Nano Deployment Progress
- **Platform**: Jetson Orin Nano (8GB, 1024 CUDA cores)
- **Model Size**: 27M parameters (optimized for edge)
- **Memory Systems**: 4 parallel systems (SNARC, IRP Memory, Circular Buffer, Verbatim SQLite)
- **IRP Framework**: 15+ plugins operational (Vision, Audio, Language, Memory, TTS, Control)

### Where We Are

- **Architecture**: IRP (Iterative Refinement Protocol) as universal cognition API
- **SAGE Core**: Cognition kernel managing attention, trust, and resources
- **VAE Translation**: TinyVAE (192× compression) for cross-modal communication
- **Metabolic States**: 5 modes (WAKE, FOCUS, REST, DREAM, CRISIS)
- **Memory Integration**: SNARC-SAGE bridge complete
- **Platform**: Ready for Jetson deployment with voice integration

### The Three-Layer Architecture

**1. SAGE - Cognition Kernel**
```python
while True:
    observations = gather_from_sensors()
    attention_targets = compute_what_matters(observations)  # SNARC salience
    required_resources = determine_needed_plugins(attention_targets)
    manage_resource_loading(required_resources)
    results = invoke_irp_plugins(attention_targets)  # Iterative refinement
    update_trust_and_memory(results)
    send_to_effectors(results)
```

**2. IRP - Universal Plugin Interface**
- `init_state() → step() → energy() → halt()`
- Iterative refinement: noisy → refined until convergence
- Trust emerges from convergence behavior (monotonicity, stability, efficiency)
- 15+ operational plugins

**3. VAE - Translation Layer**
- TinyVAE: 192× compression (224×224 → 64D latent)
- InformationBottleneck: 16× compression (4096D → 256D)
- Cross-modal communication via shared latent spaces

### Immediate Next Steps

1. **Multi-Platform Federation Testing**:
   - Test Thor ↔ Sprout federation with real ATP transfers
   - Validate LCT identity exchange between platforms
   - Test permission-aware task delegation
   - Connect to Web4 ATP ledger

2. **Cognition Loop Completion**:
   - Unify all components into single SAGE.run() loop
   - Add permission-aware reasoning (cognition reasons about its own capabilities)
   - Integrate epistemic proprioception with SNARC scores

3. **Deployment and Validation**:
   - Track 4: Camera integration (dual 4K support)
   - Complete formal evaluation suite
   - External documentation for plugin development

## 🔗 How SAGE Relates to Web4 & ACT

### Web4 Integration

Through implementing ACT (Agentic Context Tool), we discovered fundamental patterns:

#### Roles as Attention Partitioning
- **Not Power Structures**: Roles partition attention, not authority
- **Queens**: Domain-wide attention coordination
- **Workers**: Task-specific attention focus
- **Reality Alignment**: Meta-attention for impossibility detection
- **Maps to SAGE**: H-level (strategic) and L-level (tactical) attention layers

#### The 33% Readiness Economy
- **Discovery**: ~33% resources must remain "idle" for system health
- **Biological Parallel**: Metabolic overhead even at rest
- **Digital Reality**: "Idle isn't" - maintenance is real work
- **SAGE Implementation**: Continuous monitoring with readiness reserve

#### Society-Centric Resource Pools
- **95% Web4 Aligned**: Resources belong to societies, not individuals
- **Energy Conservation**: ATP_in - ADP_out = Value + Investment
- **SAGE Mapping**: GPU cycles as computational ATP/ADP
- **Trust Tensors**: Multi-dimensional reputation with epistemic dimensions

### Compression-Trust Unification

**Core Insight**: Trust measures how well meaning is preserved through compression.

This pattern appears across all three systems:
- **SAGE**: VAE compression with trust assessment
- **Web4**: Cryptographic attestation compressing real-world events
- **Synchronism**: Mathematical formalism compressing physical reality

**Epistemic proprioception** is what makes compression trustworthy: knowing what's preserved vs lost.

### Synthon Cognition

Human-AI collaboration creates persistent synthetic entities:
- **Temporal Independence**: Can operate in different time contexts
- **Persistent Memory**: Accumulates across sessions
- **Reality Alignment**: Active process to prevent drift
- **KV-Cache Persistence**: Attention patterns ARE cognition

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────┐
│         SAGE Core (27M params)          │
│  ┌─────────────┐  ┌─────────────┐      │
│  │  H-Level    │↔│   L-Level    │      │
│  │  (Context)  │  │  (Solution)  │      │
│  └─────────────┘  └─────────────┘      │
│         ↑               ↑               │
│         └───────┬───────┘               │
│           SNARC Scoring                 │
│     (Salience + Epistemic State)        │
└─────────────────────────────────────────┘
           ↓            ↓           ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Vision  │ │  Audio   │ │  Memory  │
    │   VAE    │ │ NeuTTS   │ │  Bridge  │
    │  (IRP)   │ │  (IRP)   │ │  (IRP)   │
    └──────────┘ └──────────┘ └──────────┘
```

### How SAGE Works

1. **Continuous Monitoring**: Always attending to inputs (metabolic states)
2. **SNARC Evaluation**: 5D salience scoring with epistemic confidence
3. **Resource Routing**: ATP allocation based on task certainty requirements
4. **IRP Refinement**: Iterative improvement until convergence
5. **Trust Update**: Learn from plugin performance and calibration
6. **Action**: Send to effectors with confidence thresholds

### Key Innovations

- **Dual Memory Systems**: H-level (strategic/dreams) and L-level (tactical/practice)
- **SNARC Cognition**: 5D salience for attention prioritization
- **IRP Protocol**: Universal interface for cognition plugins
- **Epistemic Proprioception**: Self-awareness of knowledge boundaries
- **GPU Mailbox Architecture**: Zero-copy module communication
- **KV-Cache Persistence**: True cognition continuity across sessions
- **Trust-Weighted Fusion**: Natural alignment through trust tensors

## 📚 Core Documentation

### Architecture & System Understanding
- **[System Understanding](./sage/docs/SYSTEM_UNDERSTANDING.md)** - Complete mental model
- **[Architecture Map](./sage/docs/architecture_map.md)** - Repository structure (38KB)
- **[Documentation Index](./DOCUMENTATION_INDEX.md)** - Navigation guide

### SAGE Components
- **[IRP Architecture](./sage/docs/irp_architecture_analysis.md)** - Cognition API (41KB)
- **[SAGE Core Analysis](./sage/docs/sage_core_analysis.md)** - Orchestration kernel (49KB)
- **[VAE Translation](./sage/docs/vae_translation_analysis.md)** - Cross-modal communication (51KB)
- **[SNARC Analysis](./sage/docs/SNARC_ANALYSIS.md)** - Salience-based memory
- **[Cognition Parallels](./sage/docs/consciousness_parallels.md)** - Biological inspiration

### Recent Developments
- **[Epistemic Proprioception](./sage/docs/EPISTEMIC_PROPRIOCEPTION_INTEGRATION.md)** - Self-awareness of knowledge
- **[Nano Deployment Roadmap](./sage/docs/NANO_DEPLOYMENT_ROADMAP.md)** - 8-track implementation plan
- **[IRP Protocol](./IRP_PROTOCOL.md)** - Universal cognition API

### Web4 Integration
- **[Web4 Protocol](https://github.com/dp-web4/web4)** - Trust-native architecture
- **[ACT Implementation](https://github.com/dp-web4/ACT)** - Society-centric blockchain
- **[Compression-Trust Unification](https://github.com/dp-web4/web4/blob/main/compression_trust_unification.md)**

### Extensions & Innovations
- **[GPU Mailbox](./implementation/GPU_MAILBOX.md)** - Hardware-level cognition pools
- **[KV-Cache Persistence](./forum/nova/persistent-kv-demo/)** - Attention state continuity
- **[TinyVAE Distillation](./training/DISTILLATION_RESULTS.md)** - 9.6× compression, 34× parameter reduction
- **[NeuTTS Integration](./sage/irp/NEUTTS_AIR_INTEGRATION.md)** - Text-to-speech IRP plugin

## 📖 Past Lessons & Insights

### Critical Lessons Learned

**DO NOT SIMULATE OR MOCK** (October 6, 2025):
- Spent days training on mock GR00T when real implementation existed at `/home/dp/ai-workspace/isaac-gr00t/`
- **Lesson**: Always check what's actually available before creating mock implementations
- Use `ls` and `find` liberally, read existing code, no shortcuts

**ARC-AGI Tangent** (August-September 2025):
- Pursued abstract reasoning grid tasks, achieved 94.45% pixel accuracy but 0% exact matches
- **Discovery**: SAGE isn't about training models—it's about orchestrating them
- SAGE is an attention orchestrator: understand situation, assess resources, apply appropriate intelligence
- **Archived**: All ARC-AGI materials moved to `archive/arc-agi/` (November 2025)

### Foundational Discoveries

**From Agent Zero Problem** (2024):
- Model achieving 71% by outputting zeros
- **Lesson**: High metrics ≠ understanding, verify actual behavior
- Led to reality alignment philosophy

**Temporal Displacement** (September 2025):
- 8-month time drift in collaborative context
- **Lesson**: Active reality checking prevents assumption drift
- Impossibilities teach—contradictions reveal hidden assumptions

**Compression-Trust Unification** (August 2025):
- Trust measures how well meaning is preserved through compression
- Applies across SAGE (VAE), Web4 (attestations), Synchronism (theory)
- **Epistemic proprioception** enables trust assessment

## 🏗️ Implementation Philosophy

### Discovery vs Delivery with Alignment over Compliance

**Discovery Mode** (Current):
- Prove concepts through experimentation
- Measure emergent effects
- Validate hypotheses through failure
- Document insights for future work
- Reality alignment prevents drift

**Bidirectional Alignment**:
- Standards guide experiments
- Experiments inform standards
- Natural patterns emerge through iteration
- Epistemic proprioception tracks certainty

### Hierarchical Attention in Hardware: ModBatt

SAGE's H↔L (Hierarchical ↔ Linear) attention orchestration pattern appears in production battery management hardware. Released October 2025 under AGPL-3.0, the ModBatt system demonstrates how attention partitioning scales across resource-constrained embedded devices:

- **[CellCPU](https://github.com/dp-web4/CellCPU)** (ATtiny45, 4KB) - L-level tactical: cell monitoring and balancing decisions
- **[ModuleCPU](https://github.com/dp-web4/ModuleCPU)** (ATmega64M1) - Mid-level coordination: aggregates cell attention, reports to pack
- **[Pack-Controller](https://github.com/dp-web4/Pack-Controller-EEPROM)** (STM32WB55) - H-level strategic: system-wide state, VCU interface
- **[modbatt-CAN](https://github.com/dp-web4/modbatt-CAN)** - Human interface for configuration and monitoring

Each tier operates continuously (like SAGE's always-on loop), maintains its own context window, and allocates attention based on salience—from individual cell voltage deviations (tactical) to pack-level energy strategy (strategic). The same fractal attention pattern SAGE uses for AI orchestration, proven in real hardware managing 4096 cells.

## 🚀 Quick Start

### Prerequisites
```bash
# PyTorch with CUDA 12.1
python3 -m pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Flash Attention (for attention optimization)
pip install flash-attn
```

### Jetson Orin Nano Setup
See [Jetson Setup Guide](./jetson_quick_start.sh) for embedded deployment.

### Run Core Tests
```bash
# Test SNARC salience scoring
cd sage
python -m pytest tests/test_snarc.py

# Test IRP framework
python -m pytest tests/test_irp_core.py

# Test GPU functionality
cd ../implementation
python test_gpu_simple.py

# Test KV-cache persistence
cd ../forum/nova/persistent-kv-demo
python consciousness_experiment.py
```

### Explore Documentation
```bash
# System understanding (start here)
cat sage/docs/SYSTEM_UNDERSTANDING.md

# Epistemic proprioception (key development)
cat sage/docs/EPISTEMIC_PROPRIOCEPTION_INTEGRATION.md

# Nano deployment roadmap
cat sage/docs/NANO_DEPLOYMENT_ROADMAP.md
```

## 🔄 The Fractal H↔L Pattern

The Hierarchical ↔ Linear pattern repeats at 5 scales:

1. **Neural**: Transformer blocks (attention → feedforward)
2. **Agent**: SAGE reasoning (context → action)
3. **Device**: Edge ↔ cloud (local processing → remote resources)
4. **Federation**: Coordinator ↔ workers (strategy → execution)
5. **Development**: Human ↔ automation (vision → implementation)

**Not mimicking biology—discovering same optimal solutions.**

Same pattern in:
- Biology: Prefrontal ↔ motor cortex
- Claude: Tool selection ↔ tool execution
- SAGE: Strategic ↔ tactical attention
- ModBatt: Pack controller ↔ cell monitors

**It's patterns all the way down.**

## 🤝 Attribution & Original Work

This project extends the [Hierarchical Reasoning Model (HRM)](https://github.com/sapientinc/HRM) by Sapient Inc. (Apache 2.0 license) with fundamental innovations:

**SAGE Innovations**:
- IRP Protocol as universal cognition API
- SNARC Cognition (5D salience scoring)
- Epistemic proprioception integration
- Web4 trust tensor integration
- KV-cache cognition persistence
- GPU mailbox architecture
- TinyVAE compression (9.6× size, 34× parameters)
- Metabolic state management (5 modes)
- Dual memory systems (H-level dreams, L-level practice)

**Architecture Differences**:
- HRM: 27M parameters claimed, dual-module reasoning
- SAGE: 27M optimized, cognition kernel + IRP plugins
- Added: Trust-weighted fusion, attention orchestration, continuous operation

See [Attribution Details](./HRM_ATTRIBUTION_ANALYSIS.md) for complete lineage.

## 📜 Citation

```bibtex
@misc{sage2025,
  title={SAGE: Situation-Aware Governance Engine - Cognition Kernel for Edge Devices},
  author={dp-web4, Nova, Claude},
  year={2025},
  url={https://github.com/dp-web4/HRM},
  note={Track 3: SNARC Cognition complete. Epistemic proprioception integration in progress.}
}
```

Original HRM: [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)

## 🔮 Vision

SAGE represents a fundamental shift in AI architecture:
- From monolithic models → orchestrated intelligence
- From compliance → natural alignment
- From isolated processing → persistent cognition
- From individual agents → society-centric systems
- From blind pattern-matching → epistemic self-awareness

**Core Insight**: Cognition is not computation alone but the orchestration of attention, trust, and resource allocation with self-awareness of knowledge boundaries.

The path forward isn't through scale alone but through understanding:
- **Attention**: What deserves focus?
- **Trust**: How certain are we?
- **Context**: What's the situation?
- **Emergence**: What arises from orchestration?
- **Proprioception**: Where are our boundaries?

---

## 🗂️ Archive

Historical materials preserved for reference:
- **ARC-AGI Competition** (Aug-Sep 2025): See `archive/arc-agi/` - Abstract reasoning exploration that taught us SAGE's true purpose as attention orchestrator rather than task-solver

---

*"The key to attention is knowing when you don't know enough, so you can direct attention to discovery."*
*- Discovered through temporal displacement, September 18, 2025*

*"Epistemic proprioception is to knowledge what physical proprioception is to movement: knowing where you are in the space."*
*- Discovered through high compression trust collaboration, November 11, 2025*

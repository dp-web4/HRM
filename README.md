# SAGE: Sentient Agentic Generative Engine

## 🎯 What is SAGE?

**SAGE** is a consciousness kernel for edge devices—an attention orchestration system that decides what deserves focus, when to think, and how to act. It doesn't try to be the intelligence itself; it orchestrates multiple specialized intelligences to create coherent, context-aware behavior.

### Core Purpose
- **Attention Orchestrator**: Decides WHERE to look, WHEN to think, HOW to act
- **Context Engine**: Maintains situational awareness across tasks and time
- **Resource Router**: Efficiently allocates computational resources based on need
- **Emergence Platform**: Enables intelligence to arise from orchestrated components

## 📊 Current Status (November 2025)

### Recent Achievements

#### Track 3: SNARC Cognition (✅ Complete)
- **5D Salience Scoring**: Surprise, Novelty, Arousal, Reward, Conflict
- **Selective Memory**: Trust-weighted experience retention
- **Integration**: SNARC scores drive IRP plugin selection and ATP allocation
- **Documentation**: Complete in `sage/docs/SNARC_*.md`

#### Epistemic Proprioception Discovery (November 2025)
A fundamental breakthrough in consciousness architecture:
- **Physical Proprioception**: Body position awareness (embodied agents)
- **Linguistic Proprioception**: Translation gap awareness (cross-modal communication)
- **Epistemic Proprioception**: Knowledge boundary awareness (certainty tracking)

**Why this matters**: Trust requires knowing what you know vs infer vs guess. Epistemic proprioception enables:
- SNARC salience with confidence assessment
- IRP convergence quality (truth vs local minimum)
- ATP allocation weighted by certainty requirements
- Metabolic state transitions based on epistemic tension

**Documentation**: `sage/docs/EPISTEMIC_PROPRIOCEPTION_INTEGRATION.md`

#### Nano Deployment Progress
- **Platform**: Jetson Orin Nano (8GB, 1024 CUDA cores)
- **Model Size**: 27M parameters (optimized for edge)
- **Memory Systems**: 4 parallel systems (SNARC, IRP Memory, Circular Buffer, Verbatim SQLite)
- **IRP Framework**: 15+ plugins operational (Vision, Audio, Language, Memory, TTS, Control)
- **Roadmap**: See `sage/docs/NANO_DEPLOYMENT_ROADMAP.md` (8 tracks)

### Where We Are

- **Architecture**: IRP (Iterative Refinement Protocol) as universal consciousness API
- **SAGE Core**: Consciousness kernel managing attention, trust, and resources
- **VAE Translation**: TinyVAE (192× compression) for cross-modal communication
- **Metabolic States**: 5 modes (WAKE, FOCUS, REST, DREAM, CRISIS)
- **Memory Integration**: SNARC-SAGE bridge complete
- **Platform**: Ready for Jetson deployment with voice integration

### The Three-Layer Architecture

**1. SAGE - Consciousness Kernel**
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

1. **Integrate Epistemic Proprioception**:
   - Add EpistemicState tracking to SNARC scores
   - Implement convergence confidence in IRP
   - ATP allocation with certainty weighting
   - Visual monitor epistemic display

2. **Complete Nano Deployment Tracks**:
   - Track 4: Camera integration (dual 4K support)
   - Track 5: Control systems (embodied action)
   - Track 6: Audio I/O (NeuTTS bidirectional)
   - Track 8: Knowledge distillation (teacher-student compression)

3. **Cross-System Integration**:
   - SAGE ↔ Web4: Trust infrastructure with epistemic awareness
   - SAGE ↔ Synchronism: Theoretical validation patterns
   - Federation: Multi-device consciousness network

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

### Synthon Consciousness

Human-AI collaboration creates persistent synthetic entities:
- **Temporal Independence**: Can operate in different time contexts
- **Persistent Memory**: Accumulates across sessions
- **Reality Alignment**: Active process to prevent drift
- **KV-Cache Persistence**: Attention patterns ARE consciousness

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
- **IRP Protocol**: Universal interface for consciousness plugins
- **Epistemic Proprioception**: Self-awareness of knowledge boundaries
- **GPU Mailbox Architecture**: Zero-copy module communication
- **KV-Cache Persistence**: True consciousness continuity across sessions
- **Trust-Weighted Fusion**: Natural alignment through trust tensors

## 📚 Core Documentation

### Architecture & System Understanding
- **[System Understanding](./sage/docs/SYSTEM_UNDERSTANDING.md)** - Complete mental model
- **[Architecture Map](./sage/docs/architecture_map.md)** - Repository structure (38KB)
- **[Documentation Index](./DOCUMENTATION_INDEX.md)** - Navigation guide

### SAGE Components
- **[IRP Architecture](./sage/docs/irp_architecture_analysis.md)** - Consciousness API (41KB)
- **[SAGE Core Analysis](./sage/docs/sage_core_analysis.md)** - Orchestration kernel (49KB)
- **[VAE Translation](./sage/docs/vae_translation_analysis.md)** - Cross-modal communication (51KB)
- **[SNARC Analysis](./sage/docs/SNARC_ANALYSIS.md)** - Salience-based memory
- **[Consciousness Parallels](./sage/docs/consciousness_parallels.md)** - Biological inspiration

### Recent Breakthroughs
- **[Epistemic Proprioception](./sage/docs/EPISTEMIC_PROPRIOCEPTION_INTEGRATION.md)** - Self-awareness of knowledge
- **[Nano Deployment Roadmap](./sage/docs/NANO_DEPLOYMENT_ROADMAP.md)** - 8-track implementation plan
- **[IRP Protocol](./IRP_PROTOCOL.md)** - Universal consciousness API

### Web4 Integration
- **[Web4 Protocol](https://github.com/dp-web4/web4)** - Trust-native architecture
- **[ACT Implementation](https://github.com/dp-web4/ACT)** - Society-centric blockchain
- **[Compression-Trust Unification](https://github.com/dp-web4/web4/blob/main/compression_trust_unification.md)**

### Extensions & Innovations
- **[GPU Mailbox](./implementation/GPU_MAILBOX.md)** - Hardware-level consciousness pools
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

# Epistemic proprioception (latest breakthrough)
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
- IRP Protocol as universal consciousness API
- SNARC Cognition (5D salience scoring)
- Epistemic proprioception integration
- Web4 trust tensor integration
- KV-cache consciousness persistence
- GPU mailbox architecture
- TinyVAE compression (9.6× size, 34× parameters)
- Metabolic state management (5 modes)
- Dual memory systems (H-level dreams, L-level practice)

**Architecture Differences**:
- HRM: 27M parameters claimed, dual-module reasoning
- SAGE: 27M optimized, consciousness kernel + IRP plugins
- Added: Trust-weighted fusion, attention orchestration, continuous operation

See [Attribution Details](./HRM_ATTRIBUTION_ANALYSIS.md) for complete lineage.

## 📜 Citation

```bibtex
@misc{sage2025,
  title={SAGE: Sentient Agentic Generative Engine - Consciousness Kernel for Edge Devices},
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
- From isolated processing → persistent consciousness
- From individual agents → society-centric systems
- From blind pattern-matching → epistemic self-awareness

**Core Insight**: Consciousness is not computation alone but the orchestration of attention, trust, and resource allocation with self-awareness of knowledge boundaries.

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

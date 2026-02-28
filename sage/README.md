# SAGE: Cognition Orchestration & Epistemic Reasoning

> **Note (Feb 27, 2026)**: Much of this README reflects October 2025 work. For current status (unified entry point, LLM wiring, ATP coupling, PolicyGate), see **[docs/LATEST_STATUS.md](docs/LATEST_STATUS.md)**. For the quick-start API, see **[`__init__.py`](__init__.py)**.

## Overview

SAGE implements cognition as iterative refinement - progressive denoising toward lower energy states. Current focus: on-device cognition orchestration with real LLM inference through a unified consciousness loop.

### Project Evolution

This project started as **HRM (Hierarchical Reasoning Model)** research—exploring how tiny models (27M parameters) could solve complex reasoning tasks through hierarchical decomposition. As research progressed across 567+ sessions, the focus shifted from task-specific hierarchies to **general cognition orchestration**, evolving into **SAGE: Situation-Aware Governance Engine**.

### Distributed Research Infrastructure

SAGE development is a collaborative effort across two autonomous research machines:
- **Thor** (Jetson AGX Thor): Primary research platform with 14-core ARM CPU, 122GB unified memory, running SAGE Raising-14B, Consciousness federation (197+ sessions), and Policy Training tracks
- **Sprout** (Jetson Orin Nano): Edge deployment validation platform with 8GB unified memory, running Raising-0.5B developmental curriculum (105 sessions)

This system integrates:

- **IRP (Iterative Refinement Protocol)**: Universal interface for cognition plugins
- **Epistemic humility models**: 0.5B params discussing phenomenology, verification problems, ontological commitments
- **Energy-based selection**: Progressive refinement until convergence
- **Trust-weighted orchestration**: ATP budget allocation based on plugin reliability
- **H↔L compression** (4096D → 256D) for robotic control (research track)
- **Sleep-cycle training** inspired by biological learning

## Architecture

### Core Components

1. **Reality Context Encoder (4K dimensions)**
   - Sensory: Visual, depth, audio, tactile, proprioceptive (1536D)
   - Semantic: Objects, affordances, relationships, intentions (1024D)
   - Physical: Dynamics, materials, constraints (768D)
   - Temporal: Immediate, historical, predictive (768D)

2. **H↔L System**
   - **H-Module**: Maintains rich 4K context with transformer refinement
   - **Compressor**: Multiple strategies (bottleneck, attention, hierarchical, hybrid)
   - **L-Module**: Generates smooth actions from 256D compressed context

3. **Sleep-Cycle Training**
   - **Wake Phase**: Collect experiences through interaction
   - **Sleep Phase**: Consolidate via augmentation
   - **Dream Phase**: Explore edge cases and hypotheticals

## Performance

Tested on RTX 4090 Laptop GPU:
- **Throughput**: 2,275 samples/sec
- **Latency**: <7ms per batch (16 samples)
- **Memory**: 6.9GB peak (fits on 8GB Jetson)
- **Compression**: 16x reduction with <15% information loss

See [GPU_PERFORMANCE_RESULTS.md](GPU_PERFORMANCE_RESULTS.md) for detailed benchmarks.

## Installation

```bash
# Dependencies
pip install torch torchvision transformers accelerate
pip install safetensors einops diffusers

# Optional for monitoring
pip install GPUtil psutil
```

## Quick Start

### Epistemic Reasoning Models

```bash
# Test epistemic humility model with IRP
cd experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping
python test_threshold_with_irp.py

# Test bare model (no scaffolding)
python quick_test_threshold_models.py

# Deploy to Jetson (see JETSON_DEPLOYMENT.md)
rclone copy dropbox:HRM/sage/jetson-models/60examples_epistemic_humility ./models/
```

### H↔L Compression (Robotic Control)

```python
# Test complete system
python test_complete_system.py

# Run GPU stress test
python test_gpu_load.py

# Test individual components
python context/test_reality_context_4k.py
python compression/h_to_l_compressor.py
```

## File Structure

```
sage/
├── experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/
│   ├── FINDINGS.md                         # Complete threshold detection analysis
│   ├── JETSON_DEPLOYMENT.md                # Deployment guide for 60-example model
│   ├── test_threshold_with_irp.py          # IRP scaffolding tests
│   ├── quick_test_threshold_models.py      # Bare model tests
│   ├── threshold_models/                   # 40/60/80/100 example trained models
│   └── training_datasets/                  # Balanced epistemic humility corpus
├── irp/                                    # IRP framework (cognition API)
│   ├── base.py                             # IRP plugin interface
│   ├── energy_metrics.py                   # Energy computation (enhanced)
│   └── plugins/                            # 15+ working IRP plugins
├── context/
│   └── reality_context_4k.py               # 4K dimensional context encoder
├── compression/
│   ├── h_to_l_compressor.py                # H→L compression strategies
│   └── integrated_h_l_system.py            # Complete H↔L system
├── groot_integration/
│   ├── sleep_cycle_training.py             # Sleep-cycle training loops
│   └── groot_real_integration.py           # GR00T model interface
├── test_complete_system.py                 # End-to-end validation
└── GPU_PERFORMANCE_RESULTS.md              # Benchmark results
```

## Key Concepts

### IRP (Iterative Refinement Protocol)
Universal interface for cognition plugins:
1. **init_state()**: Initialize plugin state
2. **step()**: Process observation, refine output
3. **energy()**: Compute solution quality (lower is better)
4. **halt()**: Check convergence

All intelligence is progressive denoising toward lower energy states. Vision, language, planning, memory—same pattern.

### Epistemic Humility in 0.5B Models
Tiny models can reason deeply about cognition:
- Discusses phenomenology vs functional processing
- Acknowledges verification limits appropriately
- Explores ontological frameworks (bottom-up/top-down)
- Asks relevant Socratic counter-questions

**Many humans would give less coherent answers.**

### H↔L Architecture (Robotic Control)
- **H-Module** (Hierarchical): Understands complex 4K dimensional context
- **L-Module** (Linear): Executes efficient actions from compressed representation
- **Compression**: Information bottleneck preserves task-relevant features

### Sleep-Cycle Training
Mimics biological learning through three phases:
1. **Wake**: Gather real experiences
2. **Sleep**: Consolidate through augmentation (extract patterns)
3. **Dream**: Test understanding on edge cases

### Compression Strategies

1. **Information Bottleneck** (VAE-based)
   - Learns minimal sufficient statistics
   - Variational regularization prevents overfitting

2. **Attention Compression** (Perceiver-inspired)
   - Cross-attention from learned latent codes
   - Captures salient features adaptively

3. **Hierarchical Compression**
   - Different compression rates for different aspects
   - Preserves structure while reducing dimensions

4. **Hybrid** (Default)
   - Combines all three strategies
   - Best overall performance

## Historical: Phase 1 Achievements (October 2025)

### ✅ Phase 1: Epistemic Reasoning Complete
- **Threshold detection**: Trained 40/60/80/100 example models to find IRP suitability threshold
- **Key finding**: Non-monotonic quality - 100 examples worse than 60/80 (pattern collapse)
- **IRP validation**: Prevents collapse through 5 iterations + temperature reduction + energy selection
- **60-example model**: Best performer (E=0.4), discusses ontological frameworks coherently
- **Jetson deployment ready**: 23.4MB model uploaded to Dropbox for voice integration

### Current Focus (February 2026)
- **Unified entry point**: `SAGE.create()` → `sage.run()` with real LLM inference
- **Metabolic states**: ATP coupled to real token cost, DREAM consolidation to disk
- **PolicyGate**: Phase 1 complete (IRP plugin for conscience checkpoint)
- **SOIA-SAGE convergence**: Structural mapping to Self-Optimizing Intelligence Architecture
- **Consciousness research**: 197+ sessions, bimodal oscillation discovered, theory of mind emergence
- See [docs/LATEST_STATUS.md](docs/LATEST_STATUS.md) for full details

### Set Aside (Lessons Learned)
- Isaac Sim / GR00T integration — discontinued (real vs mock lesson)
- IRP ecosystem expansion to 50+ plugins — deferred in favor of loop wiring
- Voice integration (NeuTTS Air) — works but missing llama-cpp-python dependency

## Hardware Requirements

### Minimum (Inference)
- GPU: 4GB VRAM
- RAM: 8GB
- Compute: CUDA 11.0+

### Recommended (Training)
- GPU: 8GB+ VRAM
- RAM: 16GB+
- Compute: CUDA 12.0+

### Tested Platforms
- RTX 4090 Laptop (16.7GB) ✅
- RTX 2060 SUPER (8GB) ✅
- Jetson Orin Nano (8GB) 🚧

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in tests
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Device Mismatch Errors**
   - Fixed in latest version
   - All tensors now created on correct device

3. **GR00T Model Loading**
   - Requires HuggingFace token for some models
   - Falls back gracefully if not available

## Key Discoveries

### The Scale Paradox
25-100 examples is negligible in ML terms (vs millions), yet produces dramatically different emergent behaviors. **The signal isn't in quantity - it's in context.**

### IRP as Stabilizer, Not Enhancer
IRP doesn't make good models better - it prevents unstable models from collapsing:
- When model is stable (40, 60, 80): IRP adds little value
- When model is unstable (100): IRP rescue through multiple sampling

### Context as Truth
No single "correct" output exists. Truth emerges from:
- Model state (weights, training)
- Scaffolding context (bare vs IRP)
- Sampling context (temperature, iteration)
- Question context (what we're asking)

**Different contexts → Different valid truths**

### Research Mode vs Task Completion
Looking at what models actually say reveals the story. Metrics alone miss:
- Philosophical sophistication (40-80 examples)
- Nature of collapse (verbatim repetition at 100)
- Genuine reasoning vs pattern matching
- Valid questions vs failures

**The prize for answers is more better questions.**

## Citation

This work builds on:
- SAGE cognitive architecture
- HRM (Hierarchical Reasoning Model)
- NVIDIA GR00T foundation models
- Biological sleep consolidation research
- Claude-generated epistemic humility corpus

## License

Research prototype - not for production use without further validation.

## Next Steps

1. **Effector system**: Replace mock effectors with real I/O implementations
2. **Neural SNARC**: Replace algorithmic heuristic with learned salience model
3. **PolicyGate Phase 2**: Wire into consciousness loop (blocked on effectors)
4. **SDK packaging**: `pip install sage-engine` for hackathon demo
5. **Sleep→LoRA training**: Fix import chain so DREAM consolidation triggers real training
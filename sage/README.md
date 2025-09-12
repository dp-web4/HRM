# SAGE V2 with H↔L Compression System

## Overview

Implementation of SAGE V2 extended reasoning with Hierarchical-to-Linear (H↔L) compression for robotic control. This system integrates:

- **4K dimensional reality context** encoding from multi-modal sensory input
- **16x compression** (4096D → 256D) while preserving actionable information  
- **Sleep-cycle training** inspired by biological learning
- **GR00T integration** for physics-accurate world modeling

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
├── context/
│   └── reality_context_4k.py      # 4K dimensional context encoder
├── compression/
│   ├── h_to_l_compressor.py       # H→L compression strategies
│   └── integrated_h_l_system.py   # Complete H↔L system
├── groot_integration/
│   ├── sleep_cycle_training.py    # Sleep-cycle training loops
│   └── groot_real_integration.py  # GR00T model interface
├── test_complete_system.py        # End-to-end validation
├── test_gpu_load.py               # GPU stress testing
└── GPU_PERFORMANCE_RESULTS.md     # Benchmark results
```

## Key Concepts

### H↔L Architecture
- **H-Module** (Hierarchical): Understands complex 4K dimensional context
- **L-Module** (Linear): Executes efficient actions from compressed representation
- **Compression**: Information bottleneck preserves task-relevant features

### Sleep-Cycle Training
Mimics biological learning through three phases:
1. **Wake**: Gather real experiences
2. **Sleep**: Consolidate through augmentation
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

## Development Status

### ✅ Completed
- 4K reality context encoder
- H→L compression system (all strategies)
- Sleep-cycle training pipeline
- GR00T integration (removed pytorch3d dependency)
- GPU performance validation
- Device compatibility fixes

### 🚧 In Progress
- Jetson Orin Nano deployment
- Isaac Sim integration
- Real robot hardware connection

### 📋 TODO
- FP16/INT8 quantization
- Custom CUDA kernels for compression
- Distributed training across multiple devices
- Online learning during deployment

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

## Citation

This work builds on:
- SAGE cognitive architecture
- HRM (Hierarchical Reasoning Model)
- NVIDIA GR00T foundation models
- Biological sleep consolidation research

## License

Research prototype - not for production use without further validation.

## Next Steps

1. Deploy on Jetson Orin Nano for edge inference
2. Connect to Isaac Sim for realistic physics
3. Integrate with real robot hardware
4. Scale training to 10,000+ hours of experience
5. Implement quantization for production deployment
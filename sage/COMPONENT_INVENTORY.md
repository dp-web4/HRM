# Component Inventory for Modular IRP System

**Date**: October 6, 2025  
**Purpose**: Catalog existing resources and identify gaps

## What We Have

### 1. From Our Past Work

#### GPU Infrastructure ✅
- **GPU Mailboxes**: Already implemented and tested
  - Location: `/implementation/tiling_mailbox_torch_extension_v2/`
  - Status: Working on RTX 4090, RTX 2060, Jetson
  - Features: Zero-copy transfer, PBM/FTM variants

#### TinyVAE ✅ 
- **Compression Module**: Knowledge distillation proven
  - Location: `/sage/irp/plugins/tinyvae_irp_plugin.py`
  - Size: 3.4MB model (294K params)
  - Performance: MSE = 0.023 after distillation
  - Perfect for translation shims

#### IRP Framework ✅
- **Base Architecture**: Protocol defined
  - Location: `/sage/irp/base.py`
  - Interfaces: IRPPlugin, IRPState defined
  - Orchestrator: Basic version exists

#### Memory System ✅
- **SNARC Memory**: Salience-based selection
  - Location: `/memory_integration/snarc_bridge.py`
  - Features: Trust adjustment, consolidation
  - SQLite backend for persistence

#### TTS Output ✅
- **NeuTTS Air**: Already integrated
  - Location: `/sage/irp/plugins/neutts_air_impl.py`  
  - Status: Working IRP plugin
  - Model: 748M params, CPU-optimized

### 2. From GR00T (Available as Teacher)

#### Vision Understanding 🎓
- Can extract visual features (but 10B params too large)
- Can provide training signals for our Vision IRP
- Attention maps available for distillation

#### Action Generation 🎓
- Can generate robot actions 
- Can provide trajectory demonstrations
- Control policies available for imitation

#### Cross-modal Fusion 🎓
- Vision-language grounding examples
- Instruction following demonstrations

### 3. Pretrained Models (Can Acquire)

#### Vision Models (Fit 8GB)
- **MobileViT**: ~5M params, efficient vision transformer
- **EfficientNet-B0**: 5.3M params, proven backbone
- **ConvNeXt-Tiny**: 28M params, modern architecture
- **CLIP ViT-B/32**: 88M params, vision-language aligned

#### Audio Models
- **Wav2Vec2-base**: 95M params (might be tight)
- **Whisper-tiny**: 39M params, good for 8GB
- **EnCodec**: Facebook's audio codec, very efficient

#### Quantization Models
- **VQ-VAE codebooks**: Can train small ones
- **FSQ (Finite Scalar Quantization)**: More efficient than VQ

### 4. Available Infrastructure

#### Hardware ✅
- RTX 4090 (24GB) - Development
- RTX 2060 Super (8GB) - Testing constraint
- Jetson Orin Nano (8GB) - Deployment target

#### Software Stack ✅
- PyTorch 2.5.1 with CUDA 12.1
- Flash Attention compiled
- Mixed precision (FP16) support
- TensorRT for optimization

## What We Need to Acquire

### 1. Camera Input System 🔴
- **Need**: Consistent camera interface
- **Options**:
  - OpenCV for webcam
  - Synthetic data generator
  - ROS2 camera node
- **Action**: Build simple wrapper

### 2. Motor Control Interface 🔴
- **Need**: Robot control output
- **Options**:
  - Simulated robot (MuJoCo/Isaac Gym)
  - ROS2 control interface
  - Direct servo control
- **Action**: Start with simulation

### 3. Token Vocabulary 🟡
- **Need**: Unified token space for SAGE
- **Options**:
  - Learn from data (BPE-style)
  - Predefine based on modalities
  - Hierarchical (coarse-to-fine)
- **Action**: Design based on tasks

## What We Need to Build

### 1. Vision IRP Implementation 🔴
```python
class VisionIRP(SensorIRP):
    """Actual implementation beyond placeholder"""
    - Real camera input processing
    - Feature extraction (use MobileViT?)
    - TinyVAE shim for compression
    - Token quantization
```

### 2. Translation Shim System 🔴
```python
class ShimRegistry:
    """Manage shims between modules"""
    - Auto-select appropriate shim type
    - Handle shape/dtype conversions
    - Maintain metadata through pipeline
```

### 3. Trust-Attention Controller 🟡
```python
class TrustAttentionLoop:
    """Core learning mechanism"""
    - Track per-module trust scores
    - Compute surprise from predictions
    - Update trust based on surprise
    - Allocate attention by trust
```

### 4. Surprise Computer 🔴
```python
class SurpriseMetric:
    """Measure prediction error"""
    - Per-modality surprise functions
    - Normalize across modalities
    - Exponential moving average
```

### 5. Token Bus Infrastructure 🟡
```python
class TokenBus:
    """High-level token communication"""
    - Publish/subscribe pattern
    - Token sequence alignment
    - Context propagation
```

## Build Priority Order

### Phase 1: Foundation (Week 1)
1. ✅ GPU Mailboxes - Have it
2. ✅ IRP Base - Have it  
3. 🔴 Camera wrapper - Build simple version
4. 🔴 Vision IRP - Implement with MobileViT
5. 🔴 TinyVAE shim - Adapt existing code

### Phase 2: Core Loop (Week 2)
1. 🔴 Trust-attention controller
2. 🔴 Surprise computation
3. 🟡 Token vocabulary design
4. 🔴 Token bus system

### Phase 3: Expansion (Week 3)
1. 🔴 Motor control IRP
2. 🟡 Audio IRP (if needed)
3. ✅ Memory integration - Have SNARC
4. ✅ TTS output - Have NeuTTS

### Phase 4: Refinement (Week 4)
1. Knowledge distillation from GR00T
2. Multi-modal fusion
3. End-to-end optimization
4. Edge deployment

## Size Analysis for 8GB GPU

### Current Footprint
| Component | Params | Memory (FP16) | Status |
|-----------|--------|---------------|--------|
| TinyVAE | 294K | ~3MB | ✅ Have |
| NeuTTS | 748M | ~1.5GB | ✅ Have |
| SNARC Memory | <1M | <10MB | ✅ Have |
| GPU Mailboxes | - | 500MB | ✅ Have |

### Proposed Additions  
| Component | Params | Memory (FP16) | Status |
|-----------|--------|---------------|--------|
| MobileViT | 5M | ~20MB | 🔴 Need |
| Token Codebook | - | ~10MB | 🔴 Build |
| SAGE Core | 10M | ~40MB | 🟡 Modify existing |
| Motor IRP | 5M | ~20MB | 🔴 Build |

### Total Projected
~2.5GB used, 5.5GB free (plenty of headroom!)

## Immediate Actions

1. **Set up camera input** - Most fundamental need
2. **Implement Vision IRP** - Core sensing capability  
3. **Design token vocabulary** - Critical for integration
4. **Build trust-attention loop** - Learning mechanism
5. **Create simple motor output** - Close the loop

## Key Insights

- We have more infrastructure than we realized
- Main gaps are in integration, not components
- 8GB constraint is very manageable
- GR00T better as teacher than component
- Evolution strategy allows starting simple

## Next Step

Start with Vision IRP using MobileViT + TinyVAE shim, feeding into basic SAGE orchestrator with trust-attention loop. This gives us end-to-end pipeline to iterate on.
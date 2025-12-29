# Building Optimization Libraries from Source for CUDA 13

**Date**: December 28, 2025
**Platform**: NVIDIA Jetson AGX Thor Developer Kit
**CUDA Version**: 13.0
**Goal**: Build FlashAttention, vLLM, and Triton for CUDA 13 to unlock 10-100x speedup

## Background

All pre-built wheels for optimization libraries are compiled for CUDA 12. AGX Thor has CUDA 13 (Blackwell architecture), causing binary incompatibility:

```
ImportError: libcudart.so.12: cannot open shared object file
```

**Solution**: Build from source to compile against CUDA 13.

## Build Order Strategy

Building in priority order based on complexity vs impact:

1. **Triton** - Enables torch.compile (1.5-2x speedup, easiest build)
2. **FlashAttention-2** - Optimized attention (3-4x speedup, moderate)
3. **vLLM** - Full production stack (10-20x speedup, most complex)

## 1. Triton Build

**Repository**: https://github.com/triton-lang/triton
**Purpose**: JIT kernel compilation for PyTorch
**Impact**: Enables torch.compile() for 1.5-2x speedup
**Expected Build Time**: 15-30 minutes

### Build Steps

```bash
# Clone repository
cd /home/dp/ai-workspace
git clone https://github.com/triton-lang/triton.git
cd triton

# Build and install
pip install -e . --break-system-packages

# Log output to track progress
pip install -e . --break-system-packages 2>&1 | tee /tmp/triton_build.log
```

### Build Progress

#### Attempt 1: Editable Install (FAILED)

**Started**: 2025-12-28 ~16:40 UTC
**Completed**: 2025-12-28 ~16:54 UTC
**Duration**: ~14 minutes
**Command**: `pip install -e . --break-system-packages`
**Result**: ❌ **Build succeeded but package incomplete**

**What Happened**:
```bash
Successfully built triton
Installing collected packages: triton
Successfully installed triton-3.6.0+git1d5a8273
```

**The Problem**:
- Editable install created wheel successfully
- Package installed but contained no actual modules
- `import triton` worked but `triton.language` missing
- `dir(triton)` showed only `['__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']`
- PyTorch torch.compile failed with: `AttributeError: module 'triton' has no attribute 'language'`

**Root Cause Analysis**:
Editable installs (`-e` flag) for complex packages with C++/CUDA extensions don't always trigger full compilation. The install creates a development link to the source directory but doesn't build/install the actual compiled extensions.

#### Attempt 2: Full Install (IN PROGRESS)

**Started**: 2025-12-28 ~17:00 UTC
**Command**: `pip install . --break-system-packages` (no `-e` flag)
**Expected**: Full compilation and installation of all modules

**Key Observations**:
- Uninstalled previous editable version first
- Now doing complete build with all C++ extensions
- Should install triton.language and other required modules

### What's Being Compiled

Triton requires compiling:
- LLVM IR generation backend
- CUDA kernel launcher
- Python bindings
- Compiler driver

This is CPU-intensive work - the build process is compiling the compiler itself!

### Next Steps After Build

Once Triton completes:

1. **Test import**: `python3 -c "import triton; print(triton.__version__)"`
2. **Test torch.compile**: Simple model compilation test
3. **Benchmark**: Compare compiled vs vanilla model performance

### Lessons Learned

**Repository Structure**:
- Initially tried building from `triton/python/` subdirectory (WRONG)
- Correct location: Root `triton/` directory with setup.py
- Always check for setup.py/pyproject.toml at root first

**Build Time Estimation**:
- Pre-built wheels install in seconds
- Source builds take 15-30 minutes
- Trade-off: One-time cost for ongoing compatibility

## 2. FlashAttention-2 Build (Pending)

**Repository**: https://github.com/Dao-AILab/flash-attention
**Purpose**: Memory-efficient attention with O(n) memory vs O(n²)
**Impact**: 3-4x speedup on attention operations
**Expected Build Time**: 20-30 minutes

### Planned Build Steps

```bash
cd /home/dp/ai-workspace
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Build with CUDA 13
pip install . --no-build-isolation --break-system-packages
```

### Why This Matters

FlashAttention is the single most important optimization for transformer inference:
- Reduces memory bandwidth requirements (our bottleneck!)
- Enables longer context windows
- Fused kernel reduces GPU launches

**Without FlashAttention**: 30B model uses O(n²) memory for attention, limiting batch size and context
**With FlashAttention**: O(n) memory, 3-4x faster, longer contexts possible

## 3. vLLM Build (Future)

**Repository**: https://github.com/vllm-project/vllm
**Purpose**: Production-grade LLM serving with PagedAttention
**Impact**: 10-20x speedup overall
**Expected Build Time**: 45-60 minutes

### Planned Build Steps

```bash
cd /home/dp/ai-workspace
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Build for CUDA 13
VLLM_TARGET_DEVICE=cuda pip install -e . --break-system-packages
```

### Why vLLM is Last

vLLM is most complex because it includes:
- Custom CUDA kernels (like FlashAttention)
- PagedAttention memory management
- Continuous batching scheduler
- Multi-model serving infrastructure

Dependencies:
- ✅ Python + PyTorch (already installed)
- ⏳ Triton (building now)
- ⏳ FlashAttention (pending)

**Strategy**: Get Triton + FlashAttention working first, then tackle vLLM.

## Expected Performance Improvements

### Current (Vanilla PyTorch)

| Model | Tokens/sec | Time per Response |
|-------|------------|-------------------|
| Qwen 0.5B | ~60 tok/s | 1.67s |
| Nemotron 4B | ~4.6 tok/s | 21.7s |
| Qwen 14B | ~2.8 tok/s | 35.7s |
| Q3-Omni 30B | ~0.65 tok/s | 154s |

### Target (With Optimizations)

| Model | Expected | Speedup | Stack |
|-------|----------|---------|-------|
| Qwen 0.5B | ~90 tok/s | 1.5x | torch.compile |
| Nemotron 4B | ~15 tok/s | 3.3x | +FlashAttention |
| Qwen 14B | ~10 tok/s | 3.6x | +FlashAttention |
| Q3-Omni 30B | ~6-10 tok/s | 10-15x | +vLLM |

**Ultimate Goal** (with all optimizations + NVIDIA TensorRT-LLM):
- 70B models at 50-100+ tokens/sec
- Matching published AGX Thor benchmarks
- Utilizing full 500 TOPS FP16 capability

## Build Environment

```bash
# Platform info
uname -r
# 6.8.12-tegra

# CUDA version
nvcc --version
# CUDA 13.0

# PyTorch version
python3 -c "import torch; print(torch.__version__)"
# 2.9.0+cu130

# GPU info
nvidia-smi
# Blackwell architecture, 128GB unified memory
```

## CUDA 13 Specifics

**Why CUDA 13 is Different**:
- Blackwell GPU architecture requires CUDA 13+
- Binary incompatible with CUDA 12 libraries
- Bleeding edge - ecosystem catching up

**Verification**: Libraries must link against:
- `/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so.13`
- NOT `libcudart.so.12` (old version)

## Research Notes

### What We're Learning

**The Build Process IS The Learning**:
- Understanding library dependencies
- Seeing what's actually being compiled
- Discovering architecture-specific optimizations
- Learning CUDA build toolchain

**Why Source Builds Matter**:
1. Platform compatibility (CUDA versions, architectures)
2. Customization potential (enable specific features)
3. Debugging capability (have source code)
4. Latest features (not waiting for official wheels)

### Connection to SAGE

This work directly supports SAGE's resource management goals:
- **Metabolic budgeting**: Faster inference = more ATP available
- **Dynamic loading**: Can benchmark different stacks, choose best
- **Edge deployment**: Making 30B models practical on device
- **Trust calibration**: Measuring actual vs expected performance

## Timeline

- **16:39 UTC**: Started Triton clone
- **16:40 UTC**: Started Triton build
- **16:45+ UTC**: Build still compiling (expected)

**Total Expected**: 2-4 hours for all three libraries

## Next Steps

1. ⏳ **Complete Triton build** (in progress)
2. ✅ **Test torch.compile** with Triton
3. 📊 **Benchmark torch.compile speedup**
4. 🔨 **Build FlashAttention-2**
5. ✅ **Test FlashAttention import and basic usage**
6. 📊 **Benchmark FlashAttention speedup**
7. 🔨 **Build vLLM** (most complex)
8. ✅ **Test vLLM serving**
9. 📊 **Full stack benchmark**
10. 📝 **Document final results**

## Success Criteria

**Triton**: `import triton` works, torch.compile() produces speedup
**FlashAttention**: `import flash_attn` works, attention 3-4x faster
**vLLM**: `import vllm` works, full model serving 10-20x faster

**Overall**: Demonstrate path from 0.65 tok/s → 6-10 tok/s on 30B model

---

*"Building from source isn't just about getting it to work - it's about understanding why it works."*

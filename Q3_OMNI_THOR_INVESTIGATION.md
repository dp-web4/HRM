# Q3-Omni on Jetson AGX Thor - Full Investigation

## Platform
- **Hardware**: NVIDIA Jetson AGX Thor
- **GPU**: NVIDIA Thor (CUDA 13.0, Driver 580.00)
- **RAM**: 122GB
- **Swap**: 149GB configured
- **Architecture**: ARM64 (aarch64)

## Chronology of Attempts

### Phase 1: HuggingFace Transformers (Native)
**File**: `test_full_huggingface_model.py`
**Result**: ❌ FAILED - `AutoModelForCausalLM` doesn't recognize `Qwen3OmniMoeConfig`
**Error**: `ValueError: Unrecognized configuration class`
**Cause**: Q3-Omni is NOT a standard causal LM - it's a conditional generation model

### Phase 2: Direct Import (Native)
**File**: `test_full_model_direct_import.py`
**Result**: ❌ FAILED - Script error (model variable undefined)
**Status**: Found `Qwen3OmniMoeForConditionalGeneration` class but didn't instantiate properly
**Memory**: No loading occurred

### Phase 3: Official Pattern with Swap
**File**: `test_full_model_with_swap.py` (PID 546207)
**Result**: ⚠️ PARTIAL - Model started loading, then crashed
**Progress**: Loaded 113/2034 weights (5.6%) before process death
**Memory Before**: 25GB used, 97GB free, 625MB swap used
**What Happened**:
- Process loaded code2wav weights successfully
- Loading speed: ~2000-6000 weights/sec initially
- Process disappeared without error message
- Log file truncated at weight #115
- **No OOM killer message**
- **No memory exhaustion visible**

**Key Observation**: Model WAS loading and swap WAS available, but process died silently

### Phase 4: vLLM Native Build (12+ attempts)
**Blocking Issues**:
1. Missing `libnuma-dev` → FIXED
2. CPU-only PyTorch from standard PyPI → FIXED (used Jetson PyPI)
3. vLLM requires torch==2.9.1, Jetson only has 2.9.0 → PATCHED source
4. Torchvision version mismatch → PATCHED
5. Build isolation prevents ninja/CUDA detection → UNSOLVED
6. Ninja not found in isolated pip build environment → UNSOLVED

**Final Status**: Never achieved a working vLLM build natively

### Phase 5: vLLM Docker Container
**Attempts**:
1. **Container 25.09-py3 (stock)**: vLLM 0.10.1.1 - No Q3-Omni support
2. **Upgrade to 0.13.0**: CUDA 12 binaries incompatible with CUDA 13 container
3. **Mount vLLM source**: `std::bad_alloc` error during import (NOT during model loading)

**Docker Allocation Error Analysis**:
- Error: `terminate called after throwing an instance of 'std::bad_alloc'`
- When: During vLLM Python import, NOT model loading
- Memory available: 115GB free RAM + 149GB swap
- **NOT a memory availability issue**
- Likely: C++ extension incompatibility or static allocation issue

## Cross-Reference of Blocking Issues

| Issue | Method | Status | Blocker Type |
|-------|--------|--------|--------------|
| AutoModelForCausalLM wrong class | HF native | ❌ FATAL | Architecture mismatch |
| Model loading crash @ 5.6% | Native swap | ⚠️ PARTIAL | Unknown (silent death) |
| PyTorch 2.9.0 vs 2.9.1 | vLLM native | 🔧 PATCHED | Version ecosystem lag |
| Ninja in build isolation | vLLM native | ❌ UNSOLVED | Build system |
| CUDA 12 vs 13 binaries | vLLM docker | ❌ FATAL | Binary incompatibility |
| std::bad_alloc on import | vLLM docker+source | ❌ FATAL | C++ extension issue |

## Memory Behavior Analysis

**Expected**: Model loads → RAM fills → Swap activates → OOM or success
**Observed in Phase 3**: Model starts loading → Silent process death at 5.6%
**Observed in Phase 5**: Import fails with allocation error despite 115GB free

### Questions

1. **Why did Phase 3 crash silently?**
   - No OOM killer message
   - No Python traceback
   - Process just disappeared
   - Only 5.6% of weights loaded
   - Memory: 97GB free, swap barely touched

2. **Why std::bad_alloc with 115GB free RAM?**
   - Error during import (before model loading)
   - Not a total memory issue
   - Possibly: Large static allocation request
   - Possibly: Fragmentation or alignment issue
   - Possibly: 32-bit vs 64-bit pointer issue in C++ extensions

## What Actually Worked

✅ **Q3-Omni model architecture** is recognized by transformers (with trust_remote_code)
✅ **Weight loading** started successfully (113+ weights loaded)
✅ **Swap space** was configured and available
✅ **Docker + NVIDIA runtime** working with GPU access
✅ **CUDA operations** functional (nvidia-smi in container works)

## What Has NEVER Worked

❌ **Complete model loading** - Always crashes before finishing
❌ **vLLM on this platform** - Neither native nor containerized
❌ **Any successful inference** - Never got past loading stage

## Research Findings (December 21, 2025)

### 🎯 **NVIDIA Official Support Confirmed**
- vLLM container `nvcr.io/nvidia/vllm:25.09-py3` is **officially supported** on Jetson Thor
- 3.5X performance improvement over initial Thor launch (September 2025)
- Container comes with vLLM 0.9.2 pre-installed (we have 0.10.1.1)
- Optimizations: FlashInfer support, Xformers integration, Thor-specific kernels

### 🔍 **CUDA 12 vs 13 Issue Explained**
**Source**: vLLM GitHub Issue #28669

**Root Cause**:
- vLLM 0.11.0+ PyPI wheels compiled against CUDA 12.x
- Jetson Thor has CUDA 13.0
- Binary incompatibility: `libcudart.so.12` not found
- CUDA lacks backward compatibility

**Why Container Worked Initially**:
- NVIDIA container is built specifically for CUDA 13 on Jetson
- Stock container (25.09) has compatible binaries
- Upgrading vLLM inside container breaks compatibility (downloads CUDA 12 wheels)

**Solution Path**:
- Use stock NVIDIA container without upgrading
- OR build vLLM from source with CUDA 13 support
- OR wait for CUDA 13 aarch64 wheels (recently added to vLLM releases)

### 🧠 **std::bad_alloc Root Cause**
**Source**: Multiple Jetson vLLM issues (#5640, #7575, #8485, #26974)

**Common Causes**:
1. **GPU memory allocation failure** (NOT total system memory)
2. **`gpu_memory_utilization` set too high** (default 0.90)
3. **Static allocation request exceeds available GPU memory**

**Why It Happens During Import**:
- vLLM pre-allocates GPU memory pools during initialization
- C++ extensions request memory before Python model loading
- Default 0.90 utilization may exceed actual available GPU memory

**Proven Solutions**:
1. Reduce `--gpu-memory-utilization` to **0.70 or 0.75**
2. Use quantized models (INT4/INT8) to reduce memory footprint
3. Lower `--max-model-len` (context window size)
4. Use `dtype=float16` instead of bfloat16

**Quote from Aetherix blog**:
> "RuntimeError: Engine Core Initialization Failed... Solution: Reduce GPU memory utilization. Try `--gpu-memory-utilization 0.75` instead of 0.90."

### 📊 **Model Size Requirements**
- 7B models: Minimum 5.25GB GPU memory
- 30B models (Q3-Omni): Estimated ~22GB in FP16, ~11GB in INT4
- 70B models: Require 52.5GB+ (works on Thor 128GB)

### 🚨 **Phase 3 Silent Crash Mystery**
Still unexplained, but likely:
- Native transformers loading hit similar memory allocation issue
- Process killed by system (not OOM killer, but another limit?)
- Possible: cgroups memory limit, ulimit, or kernel parameter
- Recommendation: Focus on vLLM container approach (official method)

## Recommended Action Plan

### ✅ **Immediate Next Steps (High Probability of Success)**

1. **Use Stock NVIDIA Container** (no vLLM upgrade)
   ```bash
   docker run --runtime=nvidia --ipc=host \
     -v /path/to/model:/models/qwen3-omni-30b:ro \
     nvcr.io/nvidia/vllm:25.09-py3 \
     python3 -m vllm.entrypoints.openai.api_server \
       --model /models/qwen3-omni-30b \
       --gpu-memory-utilization 0.70 \
       --max-model-len 4096 \
       --dtype float16 \
       --trust-remote-code
   ```

2. **If Q3-Omni Not Supported in 0.10.1.1**:
   - Try with smaller Qwen model first (Llama-3.1-8B to verify setup)
   - Check if Q3-Omni requires vLLM 0.13.0+ features
   - Consider using dusty-nv/jetson-containers which may have newer builds

3. **Alternative: Native HuggingFace with Offloading**
   ```python
   model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
       model_path,
       device_map="auto",  # Automatic device mapping
       max_memory={0: "100GB"},  # Limit GPU memory
       offload_folder="/tmp/offload",  # CPU offload directory
       torch_dtype=torch.float16,
       trust_remote_code=True
   )
   ```

### 🔬 **Investigative Tasks**

1. **Check Q3-Omni vLLM Support**:
   - Determine minimum vLLM version for Q3-Omni
   - Check if conditional generation models work in vLLM
   - May need model-specific configuration

2. **Test dusty-nv Containers**:
   - Project: https://github.com/dusty-nv/jetson-containers
   - Community-maintained Jetson containers with vLLM
   - May have newer vLLM builds with Q3-Omni support

3. **Contact NVIDIA Developer Forums**:
   - Post specific Q3-Omni + vLLM 25.09 question
   - Link to model repository
   - Share error logs from attempts

## Next Steps to Investigate

1. ✅ **Search NVIDIA forums**: "Qwen3-Omni Jetson Thor" — COMPLETE
2. ✅ **Search NVIDIA forums**: "vLLM Jetson AGX Thor" — COMPLETE
3. ✅ **Check vLLM GitHub**: Issues mentioning Thor or aarch64 + CUDA 13 — COMPLETE
4. 🎯 **Try stock container with lower GPU memory** — HIGH PRIORITY
5. 🔍 **Verify Q3-Omni support in vLLM 0.10.1.1** — NEEDED
6. 🔄 **Test with Llama-3.1-8B baseline** — VALIDATION STEP

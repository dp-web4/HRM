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

## Next Steps to Investigate

1. **Search NVIDIA forums**: "Qwen3-Omni Jetson Thor"
2. **Search NVIDIA forums**: "vLLM Jetson AGX Thor"
3. **Check vLLM GitHub**: Issues mentioning Thor or aarch64 + CUDA 13
4. **Alternative**: Use smaller model for testing (Qwen2.5-Omni?)
5. **Alternative**: Test with different loading strategy (layer-by-layer?)
6. **Root cause**: Why silent death at 5.6% with abundant resources?

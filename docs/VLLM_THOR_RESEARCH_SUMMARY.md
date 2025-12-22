# vLLM on Jetson Thor - Research Summary

**Date**: December 21, 2025
**Purpose**: Determine why Q3-Omni fails to load in vLLM on Jetson Thor

## 🎯 Key Discoveries

### 1. vLLM IS Officially Supported on Jetson Thor
- **NVIDIA container**: `nvcr.io/nvidia/vllm:25.09-py3` ✅
- **Performance**: 3.5X improvement since Thor launch (Sept 2025)
- **Version**: vLLM 0.9.2 → 0.10.1.1 in stock container
- **Optimizations**: FlashInfer, Xformers, Thor-specific kernels

**Source**: NVIDIA Developer Blog + Forums

### 2. The CUDA 12 vs 13 Problem SOLVED
**Problem**: Our `std::bad_alloc` error

**Root Cause**:
- PyPI vLLM wheels compiled for CUDA 12.x
- Jetson Thor has CUDA 13.0
- No backward compatibility

**Why It Broke**:
- We tried to upgrade vLLM inside the container (`pip install --upgrade vllm`)
- Downloaded CUDA 12 wheels incompatible with CUDA 13 container
- Mounting vLLM source also used CUDA 12 compiled extensions

**Solution**:
✅ Use stock NVIDIA container WITHOUT upgrading vLLM
✅ Build vLLM from source with CUDA 13 (what we attempted)
⏳ Wait for CUDA 13 aarch64 wheels (recently added to vLLM)

**Source**: vLLM GitHub Issue #28669

### 3. std::bad_alloc Is GPU Memory, Not RAM
**Critical Insight**: The error happens during GPU memory pool initialization, NOT system RAM allocation.

**Why It Happens**:
- vLLM pre-allocates GPU memory during import
- Default `gpu_memory_utilization=0.90` (90% of GPU memory)
- Static allocation may exceed available GPU memory
- Happens BEFORE model loading starts

**Proven Solutions** (from Jetson vLLM users):
1. **Reduce `--gpu-memory-utilization` to 0.70 or 0.75**
2. Use quantized models (INT4/INT8)
3. Lower `--max-model-len` (context window)
4. Use `dtype=float16` instead of bfloat16

**Quote from Aetherix blog**:
> "RuntimeError: Engine Core Initialization Failed... Solution: Reduce GPU memory utilization. Try `--gpu-memory-utilization 0.75` instead of 0.90."

**Source**: Multiple vLLM GitHub issues (#5640, #7575, #8485, #26974)

## 📊 Model Memory Requirements

| Model Size | FP16 Memory | INT4 Memory | Thor Capacity |
|------------|-------------|-------------|---------------|
| 7B models  | ~5.25 GB    | ~2 GB       | ✅ Plenty     |
| 30B (Q3-Omni) | ~22 GB   | ~11 GB      | ✅ Works      |
| 70B models | ~52.5 GB    | ~26 GB      | ✅ Fits       |

Thor has **128GB unified memory**, so even 30B FP16 should fit comfortably.

## 🔍 What We Did Wrong

### Mistake #1: Upgrading vLLM in Container
```bash
# ❌ WRONG - Downloads CUDA 12 wheels
pip install --upgrade vllm

# ✅ CORRECT - Use stock container as-is
docker run nvcr.io/nvidia/vllm:25.09-py3
```

### Mistake #2: Using Default GPU Memory
```bash
# ❌ Likely too aggressive for initialization
--gpu-memory-utilization 0.90

# ✅ Conservative for first attempt
--gpu-memory-utilization 0.70
```

### Mistake #3: Not Checking Q3-Omni vLLM Support
- Q3-Omni is a **conditional generation model**, not standard LLM
- May need specific vLLM version or configuration
- Need to verify if vLLM 0.10.1.1 supports this architecture

## ✅ Recommended Next Steps (Priority Order)

### High Priority: Likely to Work
1. **Test stock container with conservative settings**
   ```bash
   docker run --runtime=nvidia --ipc=host \
     -v /path/to/qwen3-omni-30b:/models/qwen3-omni-30b:ro \
     nvcr.io/nvidia/vllm:25.09-py3 \
     python3 -m vllm.entrypoints.openai.api_server \
       --model /models/qwen3-omni-30b \
       --gpu-memory-utilization 0.70 \
       --max-model-len 4096 \
       --dtype float16 \
       --trust-remote-code
   ```

2. **If Q3-Omni not supported, verify with Llama-3.1-8B**
   - Confirm vLLM setup works
   - Then investigate Q3-Omni specific requirements

3. **Check Q3-Omni vLLM compatibility**
   - Search vLLM docs for conditional generation support
   - Check Q3-Omni repo for vLLM usage examples
   - May need special configuration flags

### Medium Priority: Alternative Approaches
4. **Try dusty-nv/jetson-containers**
   - Community-maintained Jetson containers
   - May have newer vLLM builds
   - Project: https://github.com/dusty-nv/jetson-containers

5. **Native HuggingFace with device_map="auto"**
   ```python
   model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
       model_path,
       device_map="auto",
       max_memory={0: "100GB"},
       offload_folder="/tmp/offload",
       torch_dtype=torch.float16,
       trust_remote_code=True
   )
   ```

### Low Priority: If All Else Fails
6. **Complete vLLM CUDA 13 source build**
   - We already patched dependencies
   - Need to solve ninja/build isolation issue
   - Long compile time but would work

7. **Contact NVIDIA Developer Forums**
   - Post Q3-Omni + vLLM 25.09 specific question
   - Share our investigation document
   - Ask about conditional generation model support

## 🧩 Mysteries Still Unexplained

### Phase 3: Silent Process Death at 5.6%
- Native transformers loading crashed silently
- No OOM killer, no Python traceback
- 97GB free RAM, 625MB swap barely used
- Process just disappeared

**Likely Cause** (new hypothesis):
- Hit similar GPU memory allocation limit
- System killed process for exceeding ulimit or cgroup
- Not relevant if we use vLLM container (official method)

## 📚 Reference Links

### Documentation
- [Aetherix: How to Run vLLM on Jetson Thor](https://blog.aetherix.com/how-to-run-vllm-on-jetson-agx-thor/)
- [NVIDIA: vLLM Container Announcement](https://forums.developer.nvidia.com/t/announcing-new-vllm-container-3-5x-increase-in-gen-ai-performance-in-just-5-weeks-of-jetson-agx-thor-launch/346634)

### GitHub Issues
- [vLLM #28669: CUDA 12/13 Mismatch on ARM64](https://github.com/vllm-project/vllm/issues/28669)
- [vLLM #26974: Docker Failures on Jetson Thor](https://github.com/vllm-project/vllm/issues/26974)
- [vLLM #5640: Installation on Jetson AGX Orin](https://github.com/vllm-project/vllm/issues/5640)

## 🎓 Lessons Learned

1. **Use Official Containers as-is**: Don't upgrade inside container unless you understand dependencies
2. **GPU Memory ≠ System Memory**: `std::bad_alloc` during import = GPU allocation, not RAM
3. **Conservative First**: Start with low `gpu_memory_utilization`, increase gradually
4. **Verify Model Support**: Not all model architectures work in all inference engines
5. **Read the Errors Carefully**: "During import" vs "during loading" are very different
6. **Community Resources**: Aetherix blog and GitHub issues had exact solutions

## 📝 Summary

**What Worked**:
- Docker + NVIDIA runtime ✅
- GPU access in container ✅
- Stock vLLM container (before we broke it) ✅

**What Broke**:
- Upgrading vLLM (CUDA 12/13 mismatch) ❌
- Default GPU memory utilization (too high) ❌
- Not verifying Q3-Omni vLLM support ❌

**What to Try Next**:
1. Stock container + conservative GPU settings
2. Test with known-working model (Llama-3.1-8B)
3. Investigate Q3-Omni specific vLLM requirements

**Confidence Level**: HIGH for getting vLLM working, MEDIUM for Q3-Omni specifically

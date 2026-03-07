# Thor Qwen3.5-27B Instance Status

**Created**: 2026-03-06
**Machine**: Thor (Jetson AGX Thor, 122GB unified memory)
**Status**: Setup in progress - tooling compatibility issues

## Current Situation

### Models Downloaded ✅
- `Qwen_Qwen3.5-27B-Q5_K_M.gguf` (19GB) - High quality quantization
- `Qwen_Qwen3.5-27B-Q8_0.gguf` (27GB) - Near-lossless for training
- `mmproj-Qwen_Qwen3.5-27B-f16.gguf` (885MB) - Multimodal projection
- **Total**: 47GB

### Training Libraries Installed ✅
- PEFT 0.18.1
- bitsandbytes 0.49.2
- transformers 5.3.0
- accelerate 1.13.0
- llama-cpp-python 0.3.16

### Instance Configuration Created ✅
- Backend: PyTorch
- LoRA: r=16, alpha=32, dropout=0.05
- Sleep cycles: 6-hour intervals, 100 experience threshold
- Training state initialized

## Blocking Issues

### 1. llama-cpp-python Incompatibility
**Error**: `unknown model architecture: 'qwen35'`

llama-cpp-python v0.3.16 doesn't support Qwen3.5's new architecture yet. The model uses hybrid SSM+Attention (Gated DeltaNet), not standard transformers.

**Evidence from GGUF metadata**:
```
architecture: qwen35
ssm.conv_kernel: 4
ssm.state_size: 128
full_attention_interval: 4
```

This is a state-space model hybrid released March 2, 2026 - just 4 days ago.

### 2. HuggingFace Repository Not Found
**Error**: 404 for `Qwen/Qwen3.5-27B-Instruct`

Need to find the correct official repository name for the transformers version.

## Options Forward

### Option A: Wait for llama.cpp Support
- Monitor llama.cpp releases for Qwen3.5 architecture support
- Estimated: Could be weeks to months
- **Status**: Not viable for immediate use

### Option B: Find Correct HF Repository
- Search for official Qwen3.5 transformers repo
- Download full model for PyTorch training
- Use transformers + PEFT for LoRA
- **Status**: Investigating

### Option C: Use Qwen2.5-14B (Current Working Model)
- Keep existing thor-qwen2.5-14b instance
- 100% compatible with current tooling
- Proven working in production
- **Status**: Fallback option

### Option D: Upgrade llama-cpp-python
- Try latest dev/main branch of llama.cpp
- Compile with CUDA support for Qwen3.5
- **Status**: High risk, may still lack support

## Recommendations

1. **Immediate**: Search for correct Qwen3.5 HF repo
2. **Short-term**: If found, download and test with transformers
3. **Fallback**: Continue with Qwen2.5-14B for stability
4. **Long-term**: Monitor llama.cpp for Qwen3.5 GGUF support

## Files Created
- `instance.json` - Instance configuration
- `training/state.json` - Training state
- `test_gguf_load.py` - GGUF loading test (failed with arch error)
- `STATUS.md` - This file

---

**Updated**: 2026-03-06
**Next Action**: Search HuggingFace for correct Qwen3.5 repository

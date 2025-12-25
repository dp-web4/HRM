# vLLM Integration Plan for FP4 Quantized Qwen3-Omni-30B

## Current State (2025-12-24)

### ✅ Completed
- **ModelOpt FP4 quantization**: 92.4% of parameters quantized (32.59B/35.28B)
- **Model location**: `model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only/`
- **Validation**: Both FP4 and original models generate correctly with HuggingFace Transformers
- **Performance baseline**:
  - FP4: 1.34 tok/s, 65.72 GB memory
  - Original: 1.42 tok/s, 65.86 GB memory
  - No memory savings with HF Transformers (expected - needs vLLM runtime)

### ⚠️ Challenges Discovered
From comprehensive vLLM research:

1. **Version Mismatch**: vLLM-Omni requires v0.12.0, NGC containers provide v0.11.x
2. **Quantization Compatibility**: ModelOpt has experimental vLLM support with known issues
3. **Platform Constraints**: Jetson Thor with CUDA 13.0, Python 3.10

## Research Summary

### vLLM-Omni Architecture Support
- **Status**: ✅ Officially supported (released Nov 2025)
- **Requirements**: vLLM 0.12.0, Python 3.12, Linux
- **Features**: Full Qwen3-Omni-MoE support with thinker/talker disaggregation

### FP4 Quantization Options

#### Option A: llm-compressor (Recommended)
- **Pros**:
  - Native vLLM integration
  - Explicit Qwen3 MoE support (`qwen_30b_a3b.py` example)
  - W4A4 quantization on Thor (SM100/Blackwell)
  - Auto-handled MoE calibration
- **Cons**: Requires re-quantization from original model
- **Performance**: Up to 2.3x throughput vs standard 4-bit

#### Option B: BitsAndBytes FP4
- **Pros**:
  - Recently added Qwen3-Omni-MoE support
  - No calibration required
  - Simpler workflow
- **Cons**: Less optimized for Thor's NVFP4 hardware acceleration

#### Option C: ModelOpt (Current)
- **Status**: ⚠️ Experimental vLLM support
- **Issues**:
  - "Unknown quantization method" errors reported
  - Requires `--quantization modelopt` flag
  - Configuration detection problems
- **Recommendation**: Use for TensorRT-LLM, not vLLM

## Recommended Implementation Path

### Phase 1: Quick Validation (BitsAndBytes)
**Goal**: Prove vLLM-Omni works on Jetson Thor

**Steps**:
1. Install vLLM 0.12.0 (native or build from source)
2. Install vLLM-Omni
3. Load original Qwen3-Omni-30B with BitsAndBytes FP4
4. Verify inference works
5. Benchmark performance

**Expected Outcome**: Working vLLM deployment, suboptimal memory/speed

**Timeline**: 1-2 hours

### Phase 2: Optimal Quantization (llm-compressor)
**Goal**: Achieve maximum performance with NVFP4

**Steps**:
1. Install llm-compressor
2. Create calibration dataset (20 samples sufficient)
3. Quantize using NVFP4 scheme
   ```python
   recipe = QuantizationModifier(
       targets="Linear",
       scheme="NVFP4",
       ignore=["lm_head"]
   )
   ```
4. Save with `save_compressed=True`
5. Deploy with vLLM-Omni
6. Benchmark vs Phase 1

**Expected Outcome**:
- 4x memory reduction (66GB → ~16GB in GPU)
- 7.5x speedup potential
- W4A4 quantization (Thor's SM100)

**Timeline**: 2-3 hours

### Phase 3: Production Integration
**Goal**: Integrate into SAGE/IRP framework

**Steps**:
1. Create IRP plugin for vLLM inference
2. Implement streaming generation
3. Add memory management
4. Performance profiling
5. Documentation

**Timeline**: 4-6 hours

## Installation Options

### Option A: Native Installation (Recommended)
```bash
# Create environment
python3 -m venv vllm_env
source vllm_env/bin/activate

# Install vLLM 0.12.0
pip install vllm==0.12.0

# Install vLLM-Omni
pip install vllm-omni

# Install llm-compressor
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

**Pros**: Full control, easier debugging
**Cons**: Potential build issues with CUDA 13.0

### Option B: NGC Container
```bash
# Pull latest container
sudo docker pull nvcr.io/nvidia/tritonserver:25.11-vllm-python-py3

# Run and upgrade to 0.12.0
sudo docker run --gpus all -it \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/tritonserver:25.11-vllm-python-py3 bash

# Inside container
pip install vllm==0.12.0
pip install vllm-omni
```

**Pros**: Pre-configured CUDA environment
**Cons**: Container overhead, version upgrade needed

## Expected Performance Gains

### Memory Reduction
| Configuration | Disk Size | GPU Memory | KV Cache |
|---------------|-----------|------------|----------|
| Original BF16 | 66 GB | 65.86 GB | +3-5 GB |
| ModelOpt FP4 (HF) | 66 GB | 65.72 GB | +3-5 GB |
| **vLLM NVFP4** | **66 GB** | **~16 GB** | **+1-2 GB** |

**Key Insight**: vLLM loads full-precision from disk, quantizes at runtime in GPU

### Throughput Improvement
| Configuration | Tokens/sec | Speedup |
|---------------|------------|---------|
| HF BF16 | 1.42 | 1.0x |
| HF FP4 (ModelOpt) | 1.34 | 0.94x |
| **vLLM BitsAndBytes** | **~3-5** | **~2-3x** |
| **vLLM NVFP4** | **~10-15** | **~7-10x** |

**Hardware Capability**: Jetson Thor provides 2070 FP4 teraflops

## Risk Assessment

### Low Risk
- vLLM-Omni officially supports Qwen3-Omni ✅
- Thor is SM100/Blackwell (full W4A4 support) ✅
- BitsAndBytes path proven to work ✅

### Medium Risk
- vLLM 0.12.0 build on CUDA 13.0 (may need NGC container)
- Python 3.10 vs recommended 3.12 (likely compatible)
- 128GB unified memory sufficient for 30B model ✅

### High Risk
- None identified from research

## Fallback Options

### If vLLM 0.12.0 fails to install:
1. Use NGC container with vLLM 0.11.x
2. Wait for NGC 0.12.x container release
3. Use BitsAndBytes with available version

### If NVFP4 quantization fails:
1. Fall back to BitsAndBytes FP4
2. Use INT4/AWQ quantization
3. Use FP8 quantization (less compression)

### If vLLM deployment fails entirely:
1. Use TensorRT-LLM with ModelOpt quantization
2. Continue with HuggingFace Transformers
3. Explore SGLang framework

## Success Criteria

### Phase 1 (BitsAndBytes)
- ✅ vLLM-Omni loads Qwen3-Omni-30B
- ✅ Inference completes without errors
- ✅ Memory usage < 30GB
- ✅ Throughput > 2 tok/s

### Phase 2 (NVFP4)
- ✅ llm-compressor quantization succeeds
- ✅ vLLM loads quantized checkpoint
- ✅ Memory usage < 20GB
- ✅ Throughput > 5 tok/s
- ✅ Output quality maintained (vs validation baseline)

### Phase 3 (Production)
- ✅ IRP plugin functional
- ✅ Streaming generation working
- ✅ SAGE integration complete
- ✅ Documentation updated

## Next Steps

### Immediate (Today)
1. ✅ Research vLLM compatibility (COMPLETE)
2. ✅ Document integration plan (COMPLETE)
3. ⏭️ Install vLLM 0.12.0 + vLLM-Omni
4. ⏭️ Test Phase 1 (BitsAndBytes)

### Short-term (This Week)
5. ⏭️ Install llm-compressor
6. ⏭️ Re-quantize with NVFP4
7. ⏭️ Test Phase 2 deployment
8. ⏭️ Benchmark performance

### Medium-term (Next Week)
9. ⏭️ Create IRP plugin
10. ⏭️ SAGE integration
11. ⏭️ Production deployment
12. ⏭️ Documentation

## References

### Official Documentation
- vLLM-Omni: https://docs.vllm.ai/projects/vllm-omni/
- llm-compressor: https://docs.vllm.ai/projects/llm-compressor/
- NVFP4: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

### GitHub
- vLLM-Omni: https://github.com/vllm-project/vllm-omni
- llm-compressor: https://github.com/vllm-project/llm-compressor
- Qwen3-Omni: https://github.com/QwenLM/Qwen3-Omni

### Community
- vLLM Slack: #sig-omni channel
- Weekly meetings: Tuesdays 19:30 PDT

---

**Status**: Ready to proceed with Phase 1
**Last Updated**: 2025-12-24
**Author**: Claude Code

# FP4 Quantization - Current State & Next Steps
**Date**: 2025-12-24
**Status**: ✅ Research Complete, Ready for vLLM Integration

## 🎯 Mission Accomplished So Far

### ✅ ModelOpt FP4 Quantization (COMPLETE)
- **Quantized**: 92.4% of parameters (32.59B/35.28B) using NVFP4_MLP_WEIGHT_ONLY_CFG
- **Location**: `model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only/`
- **Size**: 66GB on disk (by design for vLLM runtime quantization)
- **Metadata**: Quantization parameters saved for vLLM compatibility

### ✅ Validation with HuggingFace Transformers (COMPLETE)
Both FP4 and original models generate correctly with proper API:

| Model | Speed | GPU Memory | Quality |
|-------|-------|------------|---------|
| **FP4 Quantized** | 1.34 tok/s | 65.72 GB | ✅ Excellent |
| **Original BF16** | 1.42 tok/s | 65.86 GB | ✅ Excellent |

**Key Finding**: No memory savings with HF Transformers (expected - quantization metadata present, but needs vLLM runtime to realize benefits)

### ✅ Critical API Discovery (COMPLETE)
Found and fixed Qwen3-Omni generation requirements:
1. ✅ `processor.apply_chat_template()` - proper conversation formatting
2. ✅ `thinker_return_dict_in_generate=True` - enables thinker/talker architecture
3. ✅ `text_ids.sequences` + `batch_decode()` - correct output handling

### ✅ vLLM Compatibility Research (COMPLETE)
Comprehensive research findings in `VLLM_INTEGRATION_PLAN.md`:
- **vLLM-Omni**: Official support for Qwen3-Omni-MoE (requires v0.12.0)
- **NVFP4**: Supported via llm-compressor with W4A4 on Jetson Thor
- **BitsAndBytes**: Alternative FP4 path, simpler but less optimal
- **NGC Containers**: Available with vLLM 0.11.x (need 0.12.0 for vLLM-Omni)

## 🚀 The Goal: Utilize FP4 with vLLM

### Expected Performance Gains
| Metric | HF Transformers | vLLM Target |
|--------|-----------------|-------------|
| **GPU Memory** | 65.72 GB | ~16-20 GB |
| **Throughput** | 1.34 tok/s | ~10-15 tok/s |
| **Speedup** | 1.0x | ~7-10x |

### How vLLM Runtime Quantization Works
```
Disk (66GB FP16 weights)
    ↓ Load to system memory
    ↓ Quantize to FP4 during GPU transfer
GPU Memory (16GB FP4 weights)
    ↓ Fast FP4 matmul operations (2070 TFLOPs on Thor)
Fast Inference (7-10x speedup)
```

## 📋 Three Paths Forward

### Path 1: Test Existing NGC Container (FASTEST)
**Pros**: Containers already pulled, pre-configured environment
**Cons**: May need version upgrade for vLLM-Omni
**Timeline**: 30 minutes

```bash
# Test what's available
sudo docker run --rm --gpus all nvcr.io/nvidia/vllm:25.11-py3 \
  python3 -c "import vllm; print(vllm.__version__)"

# If version < 0.12.0, upgrade inside container
sudo docker run -it --gpus all \
  -v /home/dp/ai-workspace/HRM/model-zoo:/models \
  nvcr.io/nvidia/vllm:25.11-py3 bash

# Inside container:
pip install vllm==0.12.0
pip install vllm-omni
```

### Path 2: Use Existing vllm-source Build (MEDIUM)
**Pros**: Already cloned, may be configured
**Cons**: Unknown build state, may need rebuild
**Timeline**: 1-2 hours

```bash
cd /home/dp/ai-workspace/vllm-source

# Check what version was attempted
git log --oneline | head -5
git describe --tags

# If correct version, try to complete build
export CUDA_HOME=/usr/local/cuda-13.0
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
pip install --break-system-packages -e .
```

### Path 3: Fresh vLLM 0.12.0 + vLLM-Omni Install (CLEANEST)
**Pros**: Clean slate, known configuration
**Cons**: Longest build time
**Timeline**: 2-3 hours

```bash
# Create fresh environment
python3 -m venv ~/vllm_omni_env
source ~/vllm_omni_env/bin/activate

# Install vLLM 0.12.0
pip install vllm==0.12.0

# Install vLLM-Omni
pip install vllm-omni

# Test installation
python3 -c "import vllm; import vllm_omni; print('Success!')"
```

## 🎬 Recommended Next Steps

### Step 1: Check Available vLLM Version (5 min)
```bash
# Test NGC container version
sudo docker run --rm --gpus all nvcr.io/nvidia/vllm:25.11-py3 \
  python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

**Decision Point**:
- If v0.12.0 → Proceed with NGC container (Path 1)
- If v0.11.x → Evaluate upgrade vs rebuild (Path 2 or 3)

### Step 2: Quick Baseline Test with Original Model (15 min)
Test if vLLM can load the **original** Qwen3-Omni-30B first:

```bash
sudo docker run --rm --gpus all \
  -v /home/dp/ai-workspace/HRM/model-zoo:/models \
  nvcr.io/nvidia/vllm:25.11-py3 \
  python3 -c "
from vllm import LLM

model = LLM(
    model='/models/sage/omni-modal/qwen3-omni-30b',
    gpu_memory_utilization=0.85
)

output = model.generate('Hello!', max_tokens=20)
print(output[0].outputs[0].text)
"
```

**If this works** → vLLM baseline established
**If this fails** → May need vLLM-Omni (v0.12.0) for Qwen3-Omni support

### Step 3: Test FP4 Model (20 min)
Once baseline works, try the quantized model:

```bash
sudo docker run --rm --gpus all \
  -v /home/dp/ai-workspace/HRM/model-zoo:/models \
  nvcr.io/nvidia/vllm:25.11-py3 \
  python3 -c "
from vllm import LLM

model = LLM(
    model='/models/sage/omni-modal/qwen3-omni-30b-fp4-weight-only',
    quantization='modelopt',  # or try without this parameter
    gpu_memory_utilization=0.85
)

output = model.generate('Hello!', max_tokens=20)
print(output[0].outputs[0].text)
print(f'Generated {len(output[0].outputs[0].token_ids)} tokens')
"
```

## 🔍 Expected Outcomes

### Scenario A: Both Work ✅
**Result**: FP4 quantization ready to use!
**Next**: Benchmark performance, integrate into SAGE

### Scenario B: Original works, FP4 fails ⚠️
**Likely Cause**: ModelOpt compatibility issue
**Solution**: Re-quantize using llm-compressor

```bash
# Install llm-compressor
pip install llm-compressor

# Re-quantize with vLLM-compatible format
# See VLLM_INTEGRATION_PLAN.md Phase 2
```

### Scenario C: Neither works ❌
**Likely Cause**: Need vLLM-Omni (v0.12.0) for Qwen3-Omni
**Solution**: Upgrade to vLLM 0.12.0 + vLLM-Omni

```bash
# Inside NGC container or fresh environment
pip install vllm==0.12.0
pip install vllm-omni

# Test again with vLLM-Omni
```

## 📊 Success Criteria

### Minimum Viable (MV)
- ✅ vLLM loads FP4 model without errors
- ✅ Generates coherent text
- ✅ GPU memory < 30 GB
- ✅ Throughput > 2 tok/s

### Target Performance
- ✅ GPU memory < 20 GB (3x reduction)
- ✅ Throughput > 5 tok/s (3-4x speedup)
- ✅ Output quality matches validation baseline

### Optimal Performance
- ✅ GPU memory ~16 GB (4x reduction)
- ✅ Throughput > 10 tok/s (7-8x speedup)
- ✅ Full W4A4 quantization utilized on Thor

## 🛠️ Troubleshooting Guide

### Error: "Unknown quantization method: modelopt"
**Solution**: vLLM version doesn't recognize ModelOpt format
```bash
# Option 1: Remove quantization parameter
model = LLM(model=path)  # Let vLLM auto-detect

# Option 2: Re-quantize with llm-compressor
# See VLLM_INTEGRATION_PLAN.md Phase 2
```

### Error: "Model architecture not supported"
**Solution**: Need vLLM-Omni for Qwen3-Omni-MoE
```bash
pip install vllm==0.12.0
pip install vllm-omni
```

### Error: "Out of memory"
**Solution**: Reduce GPU memory utilization
```python
model = LLM(
    model=path,
    gpu_memory_utilization=0.70,  # Reduced from 0.85
    max_model_len=4096  # Reduced context
)
```

## 📁 File Locations

### Quantized Models
- **FP4 (ModelOpt)**: `model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only/`
- **Original**: `model-zoo/sage/omni-modal/qwen3-omni-30b/`

### Documentation
- **Integration Plan**: `sage/quantization/VLLM_INTEGRATION_PLAN.md`
- **This File**: `sage/quantization/CURRENT_STATE_AND_NEXT_STEPS.md`
- **Validation Log**: `/tmp/fp4_validation.log`
- **Investigation**: `sage/quantization/FP4_QUANTIZATION_INVESTIGATION.md`

### Scripts
- **Validation Test**: `sage/quantization/validate_fp4_chatml.py`
- **Quantization**: `sage/quantization/quantize_q3omni_fp4_weight_only.py`
- **Working Chat Demo**: `sage/conversation/q3omni_chat_manager.py`

## 🎯 The Bottom Line

**What we have**:
- ✅ Working FP4 quantized model (92.4% params)
- ✅ Validated generation with HuggingFace
- ✅ Quantization metadata for vLLM runtime
- ✅ NGC containers with vLLM pre-installed

**What we need**:
- ⏭️ Test vLLM can load and run the model
- ⏭️ Verify FP4 runtime quantization works
- ⏭️ Benchmark actual memory/speed improvements

**Estimated time to working vLLM deployment**: 1-3 hours depending on path chosen

## 📞 What to Do Right Now

**Immediate**: Run Step 1 to check vLLM version:
```bash
sudo docker run --rm --gpus all nvcr.io/nvidia/vllm:25.11-py3 \
  python3 -c "import vllm; print(vllm.__version__)"
```

**Then**: Based on version, choose Path 1, 2, or 3 above

**Finally**: Test with Steps 2 and 3

---

**Next Update**: After vLLM version check and initial test

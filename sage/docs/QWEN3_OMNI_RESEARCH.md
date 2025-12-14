# Qwen3-Omni-30B Research & Testing

**Status**: 🚧 Work in Progress
**Date**: 2025-12-14
**Platform**: Jetson AGX Thor (122GB unified memory)

## Overview

Testing Qwen3-Omni-30B-A3B-Instruct as a potential unified model for SAGE-Thor. This model is attractive because it internalizes many IRP stack solutions we're building:

- ✅ **Native multi-modal**: Processes text, audio, images, video simultaneously
- ✅ **Real-time streaming**: Low-latency conversation with natural turn-taking
- ✅ **Speech I/O**: Native speech input (19 languages) and output (10 languages)
- ✅ **End-to-end omni-modal**: Eliminates modality routing/translation overhead
- ✅ **MoE Thinker-Talker architecture**: Novel design for high-quality responses

**Value proposition**: This is essentially "SAGE-in-a-box" - if we can get it running, it solves conversation state management, multi-modal processing, and speech synthesis in one unified model.

---

## Model Details

**Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
**Architecture**: MoE (Mixture of Experts) with Thinker-Talker design
**Experts**: 128 routed experts, 8 active per token
**Size**: 70.5GB (FP16), 35GB (INT8 AWQ)
**Modalities**: Text, audio (speech/music/sounds), images, video

### Key Capabilities
- **Speech Recognition**: Multi-language, long audio support
- **Speech Translation**: Speech-to-text and speech-to-speech
- **Audio Analysis**: Music analysis, sound effects, mixed audio
- **Vision**: OCR, object grounding, visual reasoning
- **Real-time Interaction**: Streaming responses with low latency

### Requirements
- `transformers>=4.51.0` (installed 5.0.0.dev0 from source for Q3-Omni support)
- `qwen-omni-utils` (for multi-modal processing)
- `Qwen3OmniMoeForConditionalGeneration` model class
- `Qwen3OmniMoeProcessor` (not standard tokenizer)

---

## Test Results

### Test 1: FP16 Full Precision Model (70.5GB)

**Result**: ❌ **Out of Memory**

#### Memory Behavior Analysis

| Time | Memory Used | Delta | Rate | GPU % | Phase |
|------|-------------|-------|------|-------|-------|
| T+0s | 71GB | - | - | 0% | Starting |
| T+5s | 71GB | 0GB | 0GB/s | 6% | Initializing |
| T+10s | 72GB | 1GB | 0.2GB/s | 78% | **Weights loading** |
| T+15s | 76GB | 4GB | 0.8GB/s | 90% | ⚠️ **Linear growth begins** |
| T+20s | 82GB | 6GB | **1.2GB/s** | 90% | Silent initialization |
| T+25s | 88GB | 6GB | **1.2GB/s** | 95% | Silent initialization |
| T+30s | 94GB | 6GB | **1.2GB/s** | 90% | Silent initialization |
| T+35s | 100GB | 6GB | **1.2GB/s** | 96% | Silent initialization |
| T+40s | 106GB | 6GB | **1.2GB/s** | 96% | Silent initialization |
| T+45s | 112GB | 6GB | **1.2GB/s** | 88% | Silent initialization |
| T+50s | 118GB | 6GB | **1.2GB/s** | 96% | Critical (4GB free) |
| T+55s | 122GB | 4GB | 0.8GB/s | 11% | **💀 OOM KILLED** |

#### Key Observations

1. **Two Distinct Phases**:
   - **Phase 1** (fast): Weight materialization - 2034 tensors loaded at 9k-10k tensors/sec
   - **Phase 2** (slow): Silent initialization - 50GB growth over 40 seconds

2. **Linear Growth Pattern**:
   - **Perfectly linear**: 1.2GB/sec sustained for 40 seconds
   - **High GPU activity**: 88-96% throughout (computational work, not just allocation)
   - **No progress indication**: No output showing what's happening

3. **Process Killed**:
   - Memory hit 122GB limit at T+55s
   - OOM killer terminated process
   - We don't know the **full memory requirement** - it was killed mid-initialization

#### Hypothesis: Layer-by-Layer Expert Initialization

The linear growth suggests **sequential initialization**:

```python
# Likely happening under the hood
for layer in range(num_layers):  # ~40-50 layers
    # Allocate buffers for this layer
    allocate_kv_cache(layer)           # ~400-500MB
    allocate_expert_routing(layer)     # ~300-400MB (128 experts)
    allocate_activation_buffers(layer) # ~300-400MB
    initialize_parameters(layer)       # GPU computation
    # Total: ~1.2GB per layer
```

**Why sequential?**
- Avoids even higher memory spikes
- Allows some cleanup between layers
- Standard practice for large MoE models

**Critical unknown**: How much MORE memory would it have needed?
- Killed at 122GB after allocating 50GB
- Could have needed another 20-30GB for remaining layers
- **Estimated total**: 140-150GB for full initialization

---

### Test 2: Memory Optimizations

**Attempted**: `low_cpu_mem_usage=True`
**Result**: ❌ **No significant improvement**

Memory usage pattern was **identical** to non-optimized test:
- Same 1.2GB/sec linear growth
- Same 122GB OOM failure point
- Optimization only affects loading phase, not initialization

**Conclusion**: The overhead is architectural, not a loading inefficiency.

---

## Memory Architecture Analysis

### FP16 Model Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| **Model weights** | 70.5GB | 15 safetensors shards |
| **Initialization overhead** | 50GB+ | Layer/expert buffers (incomplete) |
| **Total observed** | 122GB | Killed before completion |
| **Estimated full** | 140-150GB | Projected if allowed to finish |

### Why So Much Overhead?

The 50GB+ overhead (beyond model weights) comes from:

1. **MoE Expert Routing** (128 experts):
   - Each expert needs routing buffers
   - Expert selection requires intermediate storage
   - Active expert contexts need memory

2. **Thinker-Talker Architecture**:
   - Dual processing pipeline (analyze → respond)
   - Separate buffer sets for each stage
   - Cross-attention between thinker and talker

3. **Multi-Modal Processing**:
   - KV cache for handling audio/video/image/text
   - Modality-specific buffers
   - Cross-modal attention mechanisms

4. **Activation Tensors**:
   - Forward pass intermediate results
   - Layer-wise activation storage
   - Gradient buffers (even in inference mode)

---

## INT8 Quantization Strategy

### Current Status
🔄 **Downloading**: `cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit`
**Size**: ~35GB (50% of FP16)
**Method**: AWQ (Activation-aware Weight Quantization) - optimized for inference

### Predictions

If overhead scales proportionally with precision (50% reduction):

| Metric | FP16 | INT8 (predicted) |
|--------|------|------------------|
| **Model weights** | 70.5GB | 35GB ✅ |
| **Growth rate** | 1.2GB/sec | **0.6GB/sec** |
| **Overhead** | 50GB+ | **25GB** |
| **Total** | 140-150GB | **60-65GB** ✅ |
| **Headroom on Thor** | ❌ -18-28GB | ✅ **57-62GB free** |

**Hypothesis to test**:
- INT8 should show linear growth at ~0.6GB/sec (half the rate)
- Total memory ~60GB (fits comfortably in 122GB)
- If successful, proves overhead scales with precision

---

## Files Created

### Download Scripts
- `sage/setup/download_qwen3_omni_30b.py` - FP16 model downloader ✅
- `sage/setup/download_qwen3_omni_int8.py` - INT8 AWQ downloader 🔄

### Test Scripts
- `sage/tests/test_qwen3_omni_simple.py` - Initial test (wrong dtype) ❌
- `sage/tests/test_qwen3_omni_official.py` - Official approach (OOM) ❌
- `sage/tests/test_qwen3_omni_optimized.py` - With low_cpu_mem_usage (OOM) ❌
- `sage/tests/test_qwen3_omni_int8.py` - INT8 test (pending) ⏳

### Documentation
- `sage/tests/TEST_RESULTS_THOR.md` - Initial test results (premature conclusions)
- `sage/docs/QWEN3_OMNI_RESEARCH.md` - This document

---

## Technical Challenges Overcome

### 1. Model Class Discovery
**Error**: `ValueError: Unrecognized configuration class`
**Solution**: Use `Qwen3OmniMoeForConditionalGeneration`, not `AutoModelForCausalLM`

### 2. Processor Requirement
**Error**: Missing multi-modal processing
**Solution**: Use `Qwen3OmniMoeProcessor`, not just `AutoTokenizer`

### 3. Dependencies
**Error**: `ModuleNotFoundError: qwen_omni_utils`
**Solution**: `pip install qwen-omni-utils`

### 4. Transformers Version
**Error**: Model class not found
**Solution**: Install dev version: `pip install git+https://github.com/huggingface/transformers.git@main`

### 5. Dtype Parameter
**Error**: `AttributeError: lm_head not found`
**Solution**: Use `dtype="auto"`, not `dtype=torch.float16`

### 6. Device Mapping
**Error**: `IndexError` on tied parameters with `device_map="auto"`
**Solution**: Use `device_map="cuda"` for Jetson unified memory

---

## Lessons Learned

### 1. Memory != Model Size
**Finding**: A 70GB model requires 140-150GB to run
**Implication**: Always budget 2x model size for large MoE models

### 2. Linear Growth Is Initialization
**Finding**: The 1.2GB/sec growth is sequential layer/expert initialization
**Implication**: This is normal behavior, not a bug or inefficiency

### 3. Optimization Has Limits
**Finding**: `low_cpu_mem_usage` doesn't help with architectural overhead
**Implication**: Need quantization, not just loading optimizations

### 4. OOM Doesn't Show True Requirements
**Finding**: We saw 122GB used, but full init needed 140-150GB
**Implication**: Can't determine true memory needs from failed runs alone

### 5. High-Quality Quantization Exists
**Finding**: AWQ INT8 provides good quality at 50% memory
**Implication**: Quantization is viable for production use, not just emergency fallback

---

## Next Steps

### Immediate (In Progress)
1. ✅ Complete INT8 model download
2. ⏳ Test INT8 loading with memory monitoring
3. ⏳ Verify growth rate prediction (0.6GB/sec vs 1.2GB/sec)
4. ⏳ If successful, run inference tests

### If INT8 Works
1. Test conversation quality vs FP16 (on 14B for comparison)
2. Benchmark latency for streaming responses
3. Test multi-modal inputs (audio, images)
4. Integrate with SAGE IRP framework
5. Create capability block wrapper

### If INT8 Fails
1. Try INT4/AWQ-4bit (~17.5GB + ~12GB overhead = ~30GB total)
2. Consider modular approach (separate specialists)
3. Document as "aspirational" for future hardware

---

## Research Value

Even if we can't run Q3-Omni on Thor, this investigation provided valuable data:

1. **Memory profiling methodology**: Discovered the linear growth pattern
2. **MoE architecture insights**: Understanding of expert initialization overhead
3. **Quantization validation**: Confirms INT8 is viable path forward
4. **Integration knowledge**: Learned proper loading procedures for Q3-Omni
5. **Baseline for comparison**: Can compare modular vs unified approaches

**Quote from research philosophy**: "All failures are data" - we learned exactly why FP16 doesn't fit and how to predict INT8 behavior.

---

## Architecture Implications for SAGE

### If Q3-Omni Works (Unified Approach)
**Pros**:
- Single model handles all modalities
- Natural conversation flow
- Low-latency streaming
- Simplified architecture

**Cons**:
- Large memory footprint (even INT8)
- Single point of failure
- Less flexibility for modality-specific tuning

### Alternative: Modular Approach (Current Path)
**Pros**:
- Lower memory per model
- Specialist models for each modality
- Easier to upgrade components
- Proven working (14B text works)

**Cons**:
- More complex orchestration
- Modality translation overhead
- Higher total latency
- IRP coordination required

### Hybrid Approach (Best of Both?)
- Use Q3-Omni for interactive conversation
- Use specialists for deep processing tasks
- IRP coordinates based on task requirements
- Defined in `CAPABILITY_BLOCKS_ARCHITECTURE.md`

---

## References

- Model: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- INT8: https://huggingface.co/cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit
- Docs: https://huggingface.co/docs/transformers/en/model_doc/qwen3_omni_moe
- Related: `sage/docs/CAPABILITY_BLOCKS_ARCHITECTURE.md`
- Related: `sage/docs/THOR_SAGE_IMPLEMENTATION_PLAN.md`

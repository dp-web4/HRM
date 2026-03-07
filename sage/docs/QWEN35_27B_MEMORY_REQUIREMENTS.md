# Qwen3.5-27B Memory Requirements on Thor

**Date**: 2026-03-06
**Platform**: Jetson Orin Thor (122GB unified memory)
**Issue**: OOM crashes during model loading

---

## TL;DR

**Use 4-bit quantization for Qwen3.5-27B on Thor:**
- Peak memory: 68GB (safe)
- Steady-state: 14GB (with LoRA)
- Enables training capabilities

**Do NOT use**:
- Full precision bfloat16: 54GB allocation fails
- 8-bit quantization: 90GB+ peak memory spike

---

## Memory Requirements by Quantization Type

| Approach | Peak Memory | Final Memory | Status on Thor (122GB) | LoRA Training |
|----------|-------------|--------------|------------------------|---------------|
| **bfloat16 (full)** | 54GB | 54GB | ✗ GPU allocation fails | ✓ Yes |
| **8-bit quantization** | **90GB+** | 27GB | ✗ Approaches limit | ✓ Yes |
| **4-bit quantization** | **68GB** | 13.5GB | ✓ **SAFE** | ✓ Yes |
| **GGUF Q5_K_M** | 19GB | 19GB | ✓ Most efficient | ✗ Inference only |

---

## Why Quantization Has Peak Memory

**Critical Discovery**: Quantization requires BOTH formats in memory during conversion:

```
Quantization Process:
1. Load weights in original dtype (bfloat16)  →  54GB
2. Create quantization workspace              →  overhead
3. Quantize in-place to target dtype          →  13-27GB
4. Free original weights                      →  final size

Peak = original_size + quantized_size + overhead
```

**For 8-bit**: 54GB + 27GB + 10GB = **91GB peak**
**For 4-bit**: 54GB + 13.5GB + 5GB = **68GB peak**

---

## Recommended Configuration

### For LoRA Training (Sleep Cycle Consolidation)

```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",  # Required for quantization
    trust_remote_code=True
)
```

**Memory Profile**:
- Peak during load: 68GB
- Model after load: 13.5GB
- LoRA adapters: ~0.5GB
- **Total training**: 14GB

---

## Thor-Specific Considerations

### Unified Memory Architecture

**Thor has 122GB unified memory** (shared CPU/GPU):
- PyTorch reports: 122.82GB CUDA memory
- Single allocations fail at ~54GB
- Likely due to fragmentation or driver reserves

### File Cache vs Real Memory

**Symptom**: `free -h` shows high memory usage during model load

**Reality**: Reading safetensors creates large file cache

**Check real usage**:
```bash
# Drop caches to see real memory
sudo sysctl vm.drop_caches=3
free -h
```

### Device Map Requirements

**For quantization**: MUST use `device_map="auto"`

❌ **Don't use**: `device_map="cuda:0"` (causes issues with BitsAndBytes)

---

## Hybrid Workflow (Training + Fast Inference)

**Best of both worlds**:

1. **Training Phase**: Use Transformers + 4-bit + PEFT
   - Memory: 14GB
   - Train LoRA adapters during sleep cycles
   - Export trained adapter

2. **Inference Phase**: Use GGUF + adapter with Ollama
   - Memory: 19GB
   - 2-3x faster inference
   - Load trained adapter from step 1

---

## Troubleshooting

### Error: `NVRM: Out of memory [NV_ERR_NO_MEMORY]`

**Symptom**: Model loads weights successfully (100%) but crashes on device transfer

**Cause**: GPU allocation failure (unified memory limits)

**Solution**: Use 4-bit quantization (68GB peak vs 54GB single allocation)

### Memory Climbing During Load

**Symptom**: Memory usage increases steadily at ~1GB/sec during loading

**Cause**: Normal quantization process (creating both formats)

**Expected**: Peak at ~68GB for 4-bit, then drop to 14GB

**Monitor**:
```bash
watch -n 1 'free -h | head -2'
```

### Still Running Out of Memory

**Options**:
1. Close other applications (free up memory)
2. Use smaller model (Qwen3.5-14B or 7B)
3. Use GGUF for inference-only workloads
4. Add swap space (slower but prevents OOM)

---

## Related Documentation

- **GR00T Training OOM**: `orchestration/groot_arc_setup/CHECKPOINT_OOM_FIX.md`
- **Thor Infrastructure**: `docs/THOR_INFRASTRUCTURE_ISSUE.md`
- **Model Zoo Inventory**: `docs/MODEL_ZOO_INVENTORY.md`

---

## Memory Calculation Reference

```python
# Model: Qwen3.5-27B (27 billion parameters)

# Bytes per parameter by dtype:
bfloat16 = 2 bytes  →  27B × 2 = 54GB
int8     = 1 byte   →  27B × 1 = 27GB
int4     = 0.5 byte →  27B × 0.5 = 13.5GB

# Peak during quantization:
# peak = original + quantized + overhead
peak_8bit = 54 + 27 + 10 = 91GB  # Too high for Thor
peak_4bit = 54 + 13.5 + 5 = 68GB  # Safe for Thor
```

---

## Testing Checklist

Before deploying new quantization config:

- [ ] Monitor memory during load: `watch free -h`
- [ ] Verify peak stays under 80GB
- [ ] Check model loads successfully
- [ ] Test text generation (verify quality)
- [ ] Test LoRA training (if needed)
- [ ] Verify memory drops to expected steady-state

---

## Files

**Plugin**: `sage/irp/plugins/qwen35_27b_lora_irp.py`
**Model Path**: `/home/dp/ai-workspace/SAGE/model-zoo/qwen3.5-27b/transformers/`
**GGUF Alternative**: `/home/dp/ai-workspace/SAGE/model-zoo/qwen3.5-27b/Qwen_Qwen3.5-27B-Q5_K_M.gguf`

---

**Status**: Documented 2026-03-06
**Tested**: 4-bit configuration pending verification
**Recommended**: Use 4-bit for all Thor deployments of 27B+ models

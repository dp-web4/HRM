# Qwen3.5-27B Performance Investigation - Thor

## Hardware
- **Machine**: NVIDIA Thor (Jetson AGX)
- **CUDA Memory**: 122.82 GB (unified architecture)
- **Compute**: 2,000 TOPS, Compute Capability 11.0
- **Total RAM**: 122GB unified (shared CPU/GPU)

## Issues Fixed

### 1. CPU Offload (RESOLVED)
**Problem**: Model was being offloaded to CPU despite having 122GB VRAM available.
```
Some parameters are on the meta device because they were offloaded to the cpu.
```

**Root Cause**: Using `device_map="auto"` caused HuggingFace Accelerate to make poor decisions about memory placement.

**Solution**: Changed to `device_map="cuda:0"` with `low_cpu_mem_usage=True` to load directly to GPU.

**Commit**: `8e882f24 - Fix Qwen3.5-27B loading: use device_map=cuda:0 + low_cpu_mem_usage`

**Result**: Model now loads directly to GPU in ~46 seconds (18.12 it/s average). No more CPU offload warnings.

### 2. KeyError in Finalize (RESOLVED)
**Problem**: Generation crashed with `KeyError: 'final_energy'` during finalization.

**Root Cause**: Code tried to access `state['final_energy']` but this key wasn't in the state dict. The actual key was `state['energy']`, which gets copied to `result['final_energy']`.

**Solution**: Changed line 391 from `state['final_energy']` to `state['energy']`.

**Commit**: `81ab02cc - Fix KeyError in Qwen3.5-27B finalize: use state['energy'] not state['final_energy']`

**Result**: Generation completes successfully without crashes.

##  CRITICAL REMAINING ISSUE: Catastrophic Inference Speed

### Current Performance
- **Measured**: ~1 tok/sec (100 tokens in 1m 40s)
- **Expected**: 20-30+ tok/sec for this hardware
- **Gap**: 20-30x SLOWER than expected

### Evidence
```bash
# Test: "What is 2+2?" with max_iterations=1
real    1m40.469s   # Total time
user    0m0.042s
sys     0m0.034s

# From daemon log:
Input IDs shape: torch.Size([1, 14]), max ID: 3437, vocab size: 248077
Generation successful, output shape: torch.Size([1, 114])
# Generated ~100 tokens in 100 seconds = 1 tok/sec
```

### What We Know
1. ✅ Model loads correctly to GPU (no CPU offload)
2. ✅ Model is in bfloat16 (full precision, ~54GB)
3. ✅ Generation completes successfully (no errors)
4. ❌ Generation is catastrophically slow (1 tok/sec vs expected 20-30)

### Potential Bottlenecks (NOT YET INVESTIGATED)

1. **Generation Parameters**: Check the actual generation call for inefficient settings:
   - Beam search instead of greedy?
   - Excessive max_new_tokens?
   - Inefficient sampling parameters?

2. **Gated DeltaNet Implementation**: Qwen3.5 uses hybrid SSM+Attention (Gated DeltaNet). The warning says:
   ```
   The fast path is not available because one of the required library is not installed.
   Falling back to torch implementation.
   ```
   This could be causing 20-30x slowdown!

3. **Flash Attention Missing**: Log shows:
   ```
   To install follow https://github.com/fla-org/flash-linear-attention#installation
   ```
   Missing optimized kernels could explain the slowdown.

4. **LoRA Overhead**: LoRA is enabled (`Training: True, LoRA: True`). Even in inference mode, LoRA adapter overhead might be significant.

5. **Synchronization**: On unified memory architectures, excessive CPU-GPU sync could slow things down.

### Next Steps to Investigate

1. **Install Flash Linear Attention**:
   ```bash
   # Install FLA for optimized Gated DeltaNet
   pip install flash-linear-attention
   pip install causal-conv1d
   ```

2. **Profile Generation**: Add timing to the generation code to identify bottleneck:
   - Time for tokenization
   - Time for each forward pass
   - Time for sampling/decoding

3. **Check Generation Settings**: Review the actual `model.generate()` call parameters in the IRP plugin.

4. **Test Without LoRA**: Try disabling LoRA to see if it's adding overhead.

5. **Compare**: Test a simpler model (e.g., Qwen2-7B) to establish baseline performance on this hardware.

## Files Modified
- `sage/irp/plugins/qwen35_27b_lora_irp.py`:
  - Changed device_map from "auto" to "cuda:0"
  - Added low_cpu_mem_usage=True
  - Fixed final_energy KeyError

## Status
- ✅ Model loads correctly
- ✅ Generation works without crashes
- ❌ **Performance still ~30x slower than expected**
- ⏳ Root cause of slow inference not yet identified

## User Expectation
> "we have 122G vram, so that's not a resource under constraint. we have massive compute. we should be seeing massive performance, but we're not. let's get to the bottom of this."

**Current state**: We fixed the loading issues, but the massive performance is still not achieved. Further investigation needed into the actual inference bottleneck.

---

*Last updated: 2026-03-06 23:36 UTC*
*Session: Thor autonomous work*

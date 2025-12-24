# Training Optimization Notes

## GPU Utilization Issue Identified (2025-10-26)

### Observed Behavior
- Periodic bursts of 100% GPU utilization
- Long stretches of 0% GPU activity
- ~280ms per batch on RTX 4090 (should be <50ms)

### Root Cause: Data Loading Bottleneck

**Current configuration:**
```python
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

**Problem:**
- No parallel data loading (`num_workers=0`)
- No pinned memory for faster CPU→GPU transfer
- No prefetching
- Tokenization happens on main thread
- GPU waits ~200ms per batch for CPU to prepare data

**Timeline per batch:**
1. GPU: Forward + Backward pass (~50ms) ✅
2. CPU: Tokenize next example (~200ms) ❌ GPU idle
3. CPU→GPU: Transfer data (~30ms) ❌ GPU idle

**Result: GPU idle ~83% of the time**

### Solution: Optimized DataLoader

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,           # 4 parallel workers tokenize ahead
    pin_memory=True,         # Use pinned memory for fast transfer
    persistent_workers=True, # Keep workers alive between epochs
    prefetch_factor=2        # Each worker prefetches 2 batches
)
```

**Expected improvement:**
- 4 workers tokenize in parallel
- 2 batches prefetched per worker = 8 batches ready
- Pinned memory reduces transfer overhead
- GPU utilization: 0-20% → 80-95%
- Training speed: 3.6 it/s → 15-20 it/s (4-5x faster)
- Total time: 23 minutes → 5-6 minutes

### Why Current Training Will Still Complete

With only 25 training examples:
- Current: ~7s/epoch × 200 epochs = 23 minutes total
- Optimized: ~1.5s/epoch × 200 epochs = 5 minutes total

**Decision: Let current training complete**
- It's already running and stable
- Will finish overnight regardless
- Optimization is minor time saving for this small dataset
- Can use optimized version for future larger-scale training

### When Optimization Matters

**Small datasets (25-100 examples):**
- Bottleneck is minimal (seconds difference)
- Current approach is fine

**Large datasets (1000+ examples):**
- Bottleneck becomes hours wasted
- Optimized DataLoader is essential
- Also consider:
  - Gradient accumulation for larger effective batch size
  - Mixed precision training (fp16/bf16)
  - Compiled models (`torch.compile()`)

### Additional Optimizations for Future

#### 1. Gradient Accumulation
```python
accumulation_steps = 4
for batch in dataloader:
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit:** Simulate batch_size=4 with batch_size=1 memory footprint

#### 2. Mixed Precision (BF16)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    loss = model(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefit:** 2x faster, 2x less memory (RTX 4090 has great bf16 performance)

#### 3. Torch Compile (PyTorch 2.0+)
```python
model = torch.compile(model, mode="reduce-overhead")
```

**Benefit:** 1.5-2x speedup from kernel fusion

#### 4. Batch Size Tuning
```python
# Current: batch_size=1
# Optimal: batch_size=4-8 (still fits in memory)
# Use gradient accumulation if memory constrained
```

**Benefit:** Better GPU utilization, more stable gradients

### Measured Performance (RTX 4090)

**Current (unoptimized):**
- Speed: 3.6 it/s
- GPU utilization: 15-20%
- Time per epoch: 7s
- Total (200 epochs): 23 min

**Projected (optimized):**
- Speed: 15-20 it/s
- GPU utilization: 80-95%
- Time per epoch: 1.5s
- Total (200 epochs): 5 min

**Projected (optimized + bf16 + compile):**
- Speed: 30-40 it/s
- GPU utilization: 95-100%
- Time per epoch: 0.7s
- Total (200 epochs): 2.5 min

### Recommendations

**For this run:**
- ✅ Let it complete (already stable, will finish overnight)

**For next iteration:**
- ✅ Use optimized DataLoader (updated in code)
- ✅ Consider bf16 mixed precision
- ✅ Try batch_size=4 with gradient accumulation
- ✅ Profile with `torch.profiler` to find remaining bottlenecks

**For production/large-scale:**
- ✅ All above optimizations
- ✅ Distributed training if multiple GPUs
- ✅ FSDP for models >3B params
- ✅ DeepSpeed for maximum efficiency

# HRM Training Optimization Notes

*Extracted from Nova's detailed training analysis*

## Current Training Status (20+ hours on Legion RTX 4090)

### What's Working
- Batch size 20 (reduced from 24) keeps VRAM stable at ~12/16 GB
- Validation every 1000 steps (was every 50 - saving 40 minutes!)
- Checkpoint resume + saves every 500 steps
- Training progressing with claimed 80%+ accuracy (on training set)

### Critical Issues to Address

## 1. Driver/Kernel Issues

### Problem: `nv_queue` Blocked Tasks
Indicates latent driver issues under heavy PCIe & copy pressure.

### Solutions:
```bash
# Lock known-good driver/CUDA pair
# For RTX 4090 laptop with 80W limit (vs 150W desktop)

# Disable PCIe ASPM
sudo grub-update # Add: pcie_aspm=off

# Enable persistence mode
sudo nvidia-smi -pm 1

# Set consistent clocks
sudo nvidia-smi -lgc 1980  # Lock GPU clock
```

## 2. DataLoader Bottleneck

### Current Settings (Too Conservative)
- `num_workers=2` (was 4) - caused data wait bottleneck
- Missing prefetch optimization

### Optimized Settings:
```python
DataLoader(
    dataset,
    batch_size=20,
    num_workers=4,  # Or 6 if NVMe is fast
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    shuffle=True
)
```

## 3. Validation Strategy

### Current Problem
Full validation on entire dataset is expensive.

### Improved Strategy:
```python
# Fast validation (every 1000 steps)
fast_val_size = int(0.1 * len(val_loader))  # 10% subset
for i, batch in enumerate(val_loader):
    if i >= fast_val_size:
        break
    # Quick validation

# Full validation (every 10000 steps or on improvement)
if fast_val_improved or steps % 10000 == 0:
    # Run complete validation
```

## 4. Resume Logic Bug

### Current Issue
```python
# This assumes deterministic shuffling - WRONG with shuffle=True!
if epoch == start_epoch and global_step > 0:
    if batch_idx < (global_step % len(train_loader)):
        continue
```

### Correct Approach:
```python
# Option 1: Save/restore RNG states
checkpoint = {
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'rng_states': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
    },
    'epoch': epoch,
    'global_step': global_step
}

# Option 2: Accept non-determinism, start fresh each resume
# Just load model/optimizer, let DataLoader start naturally
```

## 5. Throughput Measurement

### Suspicious Metrics
Claimed "~2000 iterations/second" doesn't match observed 8 it/s validation.

### Proper Timing:
```python
import time

# Around training loop
data_start = time.perf_counter()
batch = next(dataloader_iter)
data_time = time.perf_counter() - data_start

compute_start = time.perf_counter()
loss = model(batch)
loss.backward()
optimizer.step()
compute_time = time.perf_counter() - compute_start

# Log both
wandb.log({
    'data_time_ms': data_time * 1000,
    'compute_time_ms': compute_time * 1000,
    'samples_per_sec': batch_size / (data_time + compute_time)
})
```

## 6. Learning Rate Scaling

### With Batch Size Change
```python
# Original: batch_size=24, lr=3e-4
# New: batch_size=20

# Linear scaling
new_lr = original_lr * (new_batch_size / original_batch_size)
new_lr = 3e-4 * (20 / 24)  # = 2.5e-4

# Or use scheduler to be robust
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

## 7. Checkpoint Best Practices

### Track Metadata
```python
# Save alongside checkpoint
metadata = {
    'best_val_loss': best_val_loss,
    'best_val_step': global_step,
    'best_epoch': epoch,
    'config_hash': hashlib.md5(str(config).encode()).hexdigest(),
    'timestamp': datetime.now().isoformat()
}

with open('checkpoint_meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## 8. Performance Optimizations

### Enable CuDNN Autotuner
```python
torch.backends.cudnn.benchmark = True  # Good for consistent input sizes
```

### Channels-Last Memory Format
```python
model = model.to(memory_format=torch.channels_last)
input = input.to(memory_format=torch.channels_last)
```

## 9. Jetson-Specific Preparations

### Key Differences to Expect:
1. **Memory**: 8GB unified (not 16GB VRAM) - need INT8/FP16
2. **Different behavior**: Unified memory, different num_workers optimal
3. **TensorRT**: Export ONNX → TensorRT for optimization

### Preparation Steps:
```python
# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "hrm_model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Then use TensorRT on Jetson
```

## Immediate Action Items

1. **Fix DataLoader**: Increase workers to 4, add prefetch
2. **Fix resume logic**: Either save RNG or accept non-determinism  
3. **Add proper timing**: Measure real throughput
4. **Implement fast validation**: 10% subset for frequent checks
5. **Lock driver config**: Prevent nv_queue issues

## Monitoring During Training

Track these metrics:
- Samples/second (not iterations)
- Data wait time vs compute time
- VRAM usage over time
- Validation loss on consistent subset
- Temperature and power draw

This should significantly improve training stability and provide accurate progress tracking.
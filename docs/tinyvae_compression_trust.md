# TinyVAE and Compression Trust

## Quick Reference

TinyVAE implements **compression trust** - the principle that meaningful communication requires both compression AND trust in shared decompression artifacts.

## Core Concept

```
Input (64x64 RGB) → Encoder → Latent (64D) → Decoder → Output
                        ↓
                  Trust Metadata
                   (LCT Wrapper)
```

## Why This Matters

### Without Compression Trust
- Raw pixel transmission: 64×64×3 = 12,288 values
- No semantic understanding
- No cross-entity sharing
- No memory efficiency

### With Compression Trust  
- Compressed latent: 64 values (192× compression!)
- Semantic representation
- Shareable between entities
- Efficient storage/recall

## Key Implementation Details

### 1. Latent Dimension = 64
```python
# Not 16, not 128, but 64 - a sweet spot
latent_dim = 64  # Rich enough for complexity, compact enough for speed
```

### 2. GroupNorm for Stability
```python
# Works with batch_size=1 (critical for Jetson)
nn.GroupNorm(num_groups=8, num_channels=channels)
```

### 3. Trust Through VAE Loss
```python
# Reconstruction + KL divergence = compression trust
loss = recon_loss + beta_kl * kl_loss
# beta_kl=0.1 balances fidelity vs regularization
```

## Testing on Jetson

### Basic Compression Test
```python
# Does compression preserve meaning?
image = get_attention_crop()  # 64x64 from motion peak
latent = tinyvae.encode(image)
reconstructed = tinyvae.decode(latent)

# Not pixel-perfect, but semantically similar
assert perceptual_similarity(image, reconstructed) > 0.85
```

### Cross-Camera Trust Test
```python
# Can both cameras share understanding?
crop_cam0 = get_crop_from_camera(0)
crop_cam1 = get_crop_from_camera(1)

latent_cam0 = tinyvae.encode(crop_cam0)
latent_cam1 = tinyvae.encode(crop_cam1)

# If viewing same object, latents should align
if same_object_visible():
    assert cosine_similarity(latent_cam0, latent_cam1) > 0.7
```

### Memory Efficiency Test
```python
# Compression enables efficient memory
raw_storage = 64 * 64 * 3 * 4  # 49,152 bytes (float32)
compressed_storage = 64 * 4      # 256 bytes (float32)
compression_ratio = raw_storage / compressed_storage  # 192x!

# Can store 192x more memories in same space
```

## Trust Calibration Scenarios

### Scenario 1: Single Entity
- Entity encodes its own perceptions
- Trust = 1.0 (perfect self-trust)
- No calibration needed

### Scenario 2: Dual Cameras  
- Two cameras, same TinyVAE model
- Trust ≈ 0.9 (high but not perfect)
- Minor calibration for perspective differences

### Scenario 3: Multiple Jetsons
- Different devices, same model checkpoint
- Trust ≈ 0.8 (good alignment)
- Calibration for hardware variations

### Scenario 4: Different Training
- Same architecture, different training data
- Trust ≈ 0.5 (moderate alignment)
- Significant calibration required

## Integration with Web4 Concepts

### LCT Wrapping
```python
def create_lct(latent, entity_id="jetson_01"):
    return {
        "latent": latent.tolist(),
        "entity": entity_id,
        "model": "tinyvae_v1",
        "timestamp": time.time(),
        "trust_score": 1.0  # Self-trust
    }
```

### Dictionary Entity
```python
# Future: Quantize latents to discrete tokens
class TinyVAEDictionary:
    def __init__(self, num_codes=512):
        self.codebook = nn.Embedding(num_codes, 64)
    
    def quantize(self, latent):
        # Find nearest code
        distances = torch.cdist(latent, self.codebook.weight)
        indices = distances.argmin(dim=1)
        return indices  # Discrete tokens!
```

## Tomorrow's Testing Priorities

1. **Basic Functionality**
   - Load model checkpoint
   - Encode/decode test images
   - Verify FP16 performance

2. **Visual Pipeline Integration**
   - Motion detection → crops
   - Crops → TinyVAE latents
   - Latents → memory storage

3. **Dual Camera Alignment**
   - Compare latents from both cameras
   - Measure trust scores
   - Test calibration needs

4. **Performance Benchmarks**
   - Encoding speed (target: <10ms)
   - Memory usage (target: <100MB)
   - Power efficiency

## Quick Commands for Testing

```bash
# Run basic TinyVAE test
cd /mnt/c/exe/projects/ai-agents/HRM
python visual_monitor/test_tinyvae_pipeline.py

# Run with dual cameras
python visual_monitor/test_dual_camera_compression.py

# Benchmark performance
python forum/nova/jetson_test_kit/benchmark_tinyvae.py
```

## Key Metrics to Track

- **Compression Ratio**: 192x (12,288 → 64)
- **Reconstruction Quality**: >0.85 perceptual similarity
- **Encoding Speed**: <10ms per crop
- **Trust Score**: >0.7 for aligned entities
- **Memory Footprint**: <100MB for model + buffers

## Links to Theory

- [Web4 Compression Trust Framework](https://github.com/dp-web4/web4/blob/main/compression_trust_unification.md)
- [Why Compression Trust Matters](https://github.com/dp-web4/web4/blob/main/compression_trust_triads.md)
- [Full Integration Guide](compression_trust_integration.md)

---

*Remember: We're not just compressing pixels. We're creating a shared language of perception.*

# TinyVAE IRP Plugin + Visual Attention Integration

This document outlines the integration of a compact VAE-based Iterative Refinement Primitive (IRP) into the existing motion-based visual attention monitor on Jetson.

---

## 🔧 Overview

- **Goal:** Extract latent encodings from motion-focused image crops using a compact VAE.
- **Use case:** Efficient, hierarchical perception and compression on Jetson-class devices.
- **Output:** Latents will be passed to downstream IRP stages (e.g., SNARCs, similarity search, tagging).

---

## 📦 Components

### 1. `TinyVAEIRP` (plugin module)
- Lightweight convolutional VAE with configurable latent dim
- Tuned for 64×64 grayscale crops (single-channel or RGB configurable)
- Uses only depthwise separable convolutions for memory efficiency

### 2. `tinyvae_irp_plugin.py`
- Contains:
  - `TinyVAEIRP` class
  - Autoencoder forward pass (`encode`, `decode`, `reconstruct`)
  - Latent vector access (`get_latents`)
  - Optional `reconstruction_loss` for training/debug

### 3. `test_tinyvae_pipeline.py`
- Hooks into existing `visual_monitor_impl.py`
- Runs `get_latest_crop()`, feeds into `TinyVAEIRP`
- Prints latent vector + optionally shows reconstruction overlay

---

## 🔄 Integration Tasks (Claude)

1. **Plugin Mounting:**
   - Add `TinyVAEIRP` as an IRP-compatible plugin
   - Enable dynamic loading via `irp_plugin_registry`

2. **Data Feed:**
   - Mount `get_latest_crop()` from `visual_monitor_impl.py` as default source
   - Optionally accept crops from other attention layers

3. **Optional SNARC Emission:**
   - Add latent output to SNARC metadata structure if desired
   - Enable toggle for publishing raw latent vs reconstruction delta

---

## 🔬 Technical Details

### TinyVAEIRP Config
```python
TinyVAEIRP(
    input_channels=1,
    latent_dim=16,
    img_size=64,
)
```

### Crop Source
```python
from visual_monitor_impl import get_latest_crop
crop = get_latest_crop()
```

---

## 🧪 Testing

Run:
```bash
python test_tinyvae_pipeline.py
```

Output:
- Live crop view
- Latent vector (printed)
- Optional reconstructed image display (toggle on/off)

---

## 📁 Files

- `tinyvae_irp_plugin.py`: VAE model + wrapper
- `test_tinyvae_pipeline.py`: Integration test script

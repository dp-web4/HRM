# PyTorch GPU Support on Jetson Orin Nano

## Current Issue (August 16, 2025)
- **Problem**: cuDNN version mismatch
- **JetPack 6.x** ships with **cuDNN 9.3.0**
- **NVIDIA PyTorch wheels** require **cuDNN 8.x**
- This causes: `libcudnn.so.8: version 'libcudnn.so.8' not found`

## Solutions

### Option 1: Use Docker Container (Recommended)
```bash
# Pull the L4T PyTorch container
sudo docker pull nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

# Run with GPU support
sudo docker run --runtime nvidia -it --rm \
    -v /home/sprout/ai-workspace/HRM:/workspace \
    nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3
```

### Option 2: Downgrade cuDNN to 8.x
1. Remove cuDNN 9: `sudo apt remove libcudnn9-cuda-12`
2. Download cuDNN 8.9 for CUDA 12 from NVIDIA Developer
3. Install manually

### Option 3: Build PyTorch from Source
Build PyTorch from source against cuDNN 9 (time-consuming)

### Option 4: Use CPU-only for Development
Currently installed PyTorch 2.4.0 works on CPU. HRM is small enough (27M params) to run on CPU for testing.

## Current Status
- PyTorch 2.4.0a0 installed (NVIDIA wheel)
- Works on CPU
- GPU blocked by cuDNN version mismatch
- HRM can run on CPU for development

## Next Steps
1. For production: Use Docker container
2. For development: Continue with CPU
3. Long-term: Wait for NVIDIA to release PyTorch wheels compatible with cuDNN 9

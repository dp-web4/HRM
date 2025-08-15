# PyTorch GPU Setup for Jetson Orin Nano - Complete Solution

## Current Status
- **Hardware**: Jetson Orin Nano (8GB, Compute Capability 8.7)
- **JetPack**: 6.2.1 (L4T R36.4.4)
- **CUDA**: 12.6 installed
- **PyTorch**: 2.5.0a0 installed but missing libcusparseLt.so.0

## The Problem
PyTorch 2.5.0 requires `libcusparseLt` (CUDA Sparse Linear Algebra Library) which isn't available in our current CUDA installation. This library is part of newer CUDA toolkit versions.

## Solution Options

### Option 1: Use NVIDIA Container (Recommended)
The most reliable way is using NVIDIA's L4T containers which have all dependencies:

```bash
# Pull the container
docker pull nvcr.io/nvidia/l4t-pytorch:r36.3.0-pth2.2-py3

# Run with GPU support
docker run -it --runtime nvidia --network host \
    -v /home/sprout/ai-workspace:/workspace \
    nvcr.io/nvidia/l4t-pytorch:r36.3.0-pth2.2-py3

# Inside container, test GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Option 2: Install Older PyTorch Version
Use PyTorch 2.0 or 2.1 which don't require libcusparseLt:

```bash
# Uninstall current version
pip3 uninstall torch -y

# Install PyTorch 2.0.0 + CUDA 11.8 (compatible with CUDA 12.6)
pip3 install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

### Option 3: Build PyTorch from Source
Build PyTorch specifically for our Jetson configuration:

```bash
# This takes 4-6 hours but gives best performance
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.1.0  # Use stable version

# Set environment
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.7"  # Orin compute capability
export CUDA_HOME=/usr/local/cuda

# Build
python3 setup.py install
```

### Option 4: Manual Library Installation
Download and install the missing library:

```bash
# Download CUDA toolkit 12.2 (has libcusparseLt)
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux_aarch64.deb

# Extract just the library we need
dpkg-deb -x cuda_12.2.0_*.deb cuda_extract
sudo cp cuda_extract/usr/local/cuda-12.2/targets/aarch64-linux/lib/libcusparseLt.so.0* /usr/local/cuda/lib64/
sudo ldconfig
```

## Quick Workaround for Testing

For immediate testing without GPU:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA

import torch
print(f"PyTorch {torch.__version__} - CPU mode")

# Your HRM code here
device = torch.device('cpu')
```

## Recommended Path Forward

1. **For immediate testing**: Use CPU mode (HRM is small enough)
2. **For production**: Use Docker container approach
3. **For best performance**: Build from source (overnight)

## Testing GPU Support

Once you have a working installation:

```python
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Performance test
    x = torch.randn(1000, 1000).cuda()
    %timeit y = torch.matmul(x, x)
```

## Notes
- The Jetson Orin Nano has excellent AI performance even in CPU mode
- HRM (27M params) fits easily in 8GB RAM
- GPU acceleration provides ~5-10x speedup for training
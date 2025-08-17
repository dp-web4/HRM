#!/bin/bash
# PyTorch installation for Jetson Orin Nano
# Using pre-built wheels from NVIDIA for JetPack 6

set -e

echo "=== PyTorch GPU Installation for Jetson Orin Nano ==="
echo "JetPack 6.x (L4T R36.4.4)"
echo ""

# Check current status
echo "Current status:"
python3 -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"
echo ""

# Uninstall existing PyTorch if needed
echo "Removing any existing PyTorch installation..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install dependencies
echo ""
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev

# Upgrade pip
pip3 install --upgrade pip setuptools wheel

# For JetPack 6.0, we use the NVIDIA index
echo ""
echo "Installing PyTorch 2.2.0 for JetPack 6..."
echo "This will download ~2GB, please be patient..."

# Method 1: Try the direct wheel first (fastest if it works)
echo "Attempting direct wheel installation..."
pip3 install --no-cache-dir \
    'torch @ https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl' \
    || {
    echo "Direct wheel failed, trying alternative method..."
    
    # Method 2: Use NVIDIA's PyTorch index
    pip3 install --no-cache-dir \
        --index-url https://developer.download.nvidia.com/compute/redist/jp/v60 \
        torch torchvision torchaudio
}

# Test the installation
echo ""
echo "=== Testing PyTorch installation ==="
python3 << 'PYTHON_EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Simple GPU test
    try:
        x = torch.rand(3, 3).cuda()
        y = torch.rand(3, 3).cuda()
        z = x + y
        print(f"GPU compute test passed: {z.shape}")
        print("\n✅ SUCCESS! PyTorch with CUDA is working!")
    except Exception as e:
        print(f"GPU test failed: {e}")
        sys.exit(1)
else:
    print("\n❌ CUDA not available!")
    print("Possible issues:")
    print("1. Try: export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:\$LD_LIBRARY_PATH")
    print("2. Reboot the system")
    print("3. Check nvidia-smi output")
    sys.exit(1)
PYTHON_EOF

echo ""
echo "Installation complete!"

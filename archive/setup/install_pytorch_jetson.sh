#!/bin/bash
# Install PyTorch with CUDA support for Jetson Orin Nano (JetPack 6.x)

set -e

echo "=== PyTorch GPU Installation for Jetson Orin Nano ==="
echo "JetPack 6.x (L4T R36.x) with Python 3.10"
echo ""

# Check current installation
echo "Current PyTorch installation:"
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"

echo ""
echo "This script will install PyTorch 2.1.0 with CUDA support for Jetson"
echo "WARNING: This will uninstall the current CPU-only PyTorch"
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Uninstall existing PyTorch
echo ""
echo "=== Step 1: Removing existing PyTorch ==="
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install dependencies
echo ""
echo "=== Step 2: Installing dependencies ==="
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev

# Export CUDA paths
echo ""
echo "=== Step 3: Setting up CUDA paths ==="
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install PyTorch for JetPack 6
echo ""
echo "=== Step 4: Installing PyTorch 2.1.0 for JetPack 6 ==="
echo "Downloading from NVIDIA's servers (this may take a while)..."

# For JetPack 6.x, we need the wheel from NVIDIA's repository
pip3 install --no-cache-dir --verbose \
    https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# Install torchvision (compatible version)
echo ""
echo "=== Step 5: Installing torchvision ==="
pip3 install torchvision==0.16.0

# Verify installation
echo ""
echo "=== Step 6: Verifying installation ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
else:
    print('ERROR: CUDA not available!')
"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "If CUDA is still not available, try:"
echo "1. Reboot the system"
echo "2. Check nvidia-smi output"
echo "3. Verify CUDA installation with: ls /usr/local/cuda/"
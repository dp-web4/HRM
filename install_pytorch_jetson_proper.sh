#!/bin/bash
# Proper PyTorch installation for Jetson Orin with JetPack 6.x

echo "🚀 Installing PyTorch for Jetson Orin (JetPack 6.x)"
echo "=================================================="
echo ""

# Check current environment
echo "📊 System Information:"
echo "JetPack version: $(cat /etc/nv_tegra_release | head -1)"
echo "Python version: $(python3 --version)"
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"
echo ""

# For JetPack 6.0+ with Python 3.10, we need PyTorch 2.1.0
# This is the official NVIDIA wheel for Jetson
TORCH_INSTALL="torch==2.1.0a0+41361538.nv23.06"

echo "📦 Installing PyTorch 2.1.0 for Jetson..."
echo "This version is specifically built for:"
echo "- JetPack 6.0+"
echo "- CUDA 12.2"
echo "- Python 3.10"
echo ""

# Install directly from NVIDIA's index
pip3 install --no-cache-dir "${TORCH_INSTALL}" --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/

# Also install torchvision if needed
echo ""
echo "📦 Installing torchvision (optional but recommended)..."
pip3 install --no-cache-dir torchvision==0.16.0a0+6e9de53 --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/

echo ""
echo "✅ Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    
    # Test basic CUDA operation
    x = torch.randn(2, 3).cuda()
    y = torch.randn(2, 3).cuda()
    z = x + y
    print(f'CUDA tensor test: Success! Shape={z.shape}')
"

echo ""
echo "🎉 PyTorch installation complete!"
#!/bin/bash
# Optimized PyTorch installation for Jetson
# Uses NVIDIA's pre-built wheels

echo "🚀 Installing PyTorch for Jetson (Optimized)"
echo "==========================================="
echo ""

# Check if already installed
if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch is already installed!"
    python3 -c "
import torch
print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
    exit 0
fi

echo "📦 Installing from NVIDIA Jetson wheels..."
echo "This uses pre-built wheels optimized for Jetson"
echo ""

# For JetPack 6.x with Python 3.10
# Using a specific wheel that should work
WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"

echo "Downloading PyTorch wheel (this may take a few minutes)..."
wget -q --show-progress "$WHEEL_URL" -O torch_jetson.whl

if [ -f torch_jetson.whl ]; then
    echo ""
    echo "Installing PyTorch..."
    pip3 install torch_jetson.whl
    
    echo ""
    echo "Cleaning up..."
    rm torch_jetson.whl
    
    echo ""
    echo "✅ Testing installation..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
else
    echo "❌ Download failed. Try the full install script instead."
    exit 1
fi

echo ""
echo "🎉 PyTorch installed successfully!"
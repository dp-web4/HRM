#!/bin/bash
# Install PyTorch from NVIDIA Jetson wheel

echo "🚀 Installing PyTorch for Jetson from NVIDIA wheel"
echo "================================================"
echo ""

# First uninstall CPU-only version
echo "🧹 Removing CPU-only PyTorch..."
pip3 uninstall torch torchvision -y

echo ""
echo "📦 Downloading PyTorch wheel for Jetson..."
echo "This is a 2.1GB file, it will take time..."
echo ""

# PyTorch 2.1.0 for JetPack 6.0 (should work with 6.2.1)
WHEEL_URL="https://developer.download.nvidia.cn/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl"
WHEEL_FILE="torch_jetson.whl"

# Download with progress
wget --show-progress -O "$WHEEL_FILE" "$WHEEL_URL" || {
    echo "❌ Download failed from primary mirror"
    echo "Trying alternative download..."
    
    # Alternative: Use direct NVIDIA developer download
    WHEEL_URL2="https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v2.0/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"
    wget --show-progress -O "$WHEEL_FILE" "$WHEEL_URL2"
}

if [ -f "$WHEEL_FILE" ]; then
    echo ""
    echo "📦 Installing PyTorch wheel..."
    pip3 install "$WHEEL_FILE"
    
    echo ""
    echo "🧹 Cleaning up..."
    rm -f "$WHEEL_FILE"
    
    echo ""
    echo "✅ Testing installation..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    x = torch.randn(2, 3).cuda()
    print(f'CUDA test: Success!')
else:
    print('Warning: CUDA not available - check installation')
"
else
    echo "❌ Failed to download wheel"
    echo ""
    echo "Manual installation required:"
    echo "1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "2. Find PyTorch 2.x for JetPack 6.x"
    echo "3. Download and install manually"
fi

echo ""
echo "Done!"
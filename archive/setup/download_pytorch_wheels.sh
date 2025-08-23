#!/bin/bash
# Download PyTorch wheels for Jetson Orin Nano

echo "📥 Downloading PyTorch wheels for Jetson Orin Nano"
echo "================================================"
echo ""

# Create download directory
mkdir -p pytorch_wheels
cd pytorch_wheels

# For JetPack 6.0+ (R36.x), we need specific wheels
# These URLs are from NVIDIA's official Jetson zoo

echo "🔍 Attempting multiple download sources..."
echo ""

# Option 1: PyTorch 2.1.0 for JetPack 6.0
echo "Option 1: PyTorch 2.1.0 (JP6.0)..."
TORCH_URL1="https://nvidia.box.com/shared/static/0w1zmsrzjizc0ynsspgp6gq4atvyd0u7.whl"
wget -O torch-2.1.0-jp6.whl "$TORCH_URL1" --progress=bar:force 2>&1 | tail -f -n +6 || echo "Failed"

# Option 2: From developer.download.nvidia.com
echo ""
echo "Option 2: PyTorch from developer portal..."
TORCH_URL2="https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp310-cp310-linux_aarch64.whl"
wget -O torch-2.0.0-jp5.whl "$TORCH_URL2" --timeout=30 2>/dev/null || echo "Failed"

# Option 3: Direct GitHub releases (community builds)
echo ""
echo "Option 3: Checking community builds..."
TORCH_URL3="https://github.com/Qengineering/PyTorch-Jetson-Nano/releases/download/v1.13.0/torch-1.13.0a0+git7c98e70-cp310-cp310-linux_aarch64.whl"
wget -O torch-1.13.0-community.whl "$TORCH_URL3" --timeout=30 2>/dev/null || echo "Failed"

echo ""
echo "📊 Downloaded wheels:"
ls -lah *.whl 2>/dev/null || echo "No wheels downloaded"

echo ""
echo "💡 Manual download instructions:"
echo "1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "2. Find section 'JetPack 6.0 / 6.0 DP'"
echo "3. Download PyTorch 2.1.0 or 2.3.0 wheel"
echo "4. Install with: pip3 install <downloaded.whl>"
echo ""
echo "Alternative: Use container approach"
echo "docker run -it --runtime nvidia nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3"

cd ..
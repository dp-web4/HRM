#!/bin/bash
# PyTorch installation for JetPack 6.2.1 (L4T 36.4.4)

echo "🚀 Installing PyTorch for JetPack 6.2.1"
echo "======================================="
echo ""

# Check environment
echo "📊 System Information:"
cat /etc/nv_tegra_release | head -1
echo "Python: $(python3 --version)"
echo ""

# For JetPack 6.x, we need to use the correct wheel
# Based on NVIDIA forums, for JetPack 6.2.1 we should use:
echo "📦 Downloading PyTorch 2.5.0 for JetPack 6.x..."
echo ""

# Direct download approach
WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v611/pytorch/torch-2.5.0a0+872d972e41.nv24.10.17622225-cp310-cp310-linux_aarch64.whl"
VISION_URL="https://developer.download.nvidia.com/compute/redist/jp/v611/pytorch/torchvision-0.20.0a0+eb8a8e0-cp310-cp310-linux_aarch64.whl"

echo "Downloading PyTorch wheel..."
wget -q --show-progress "$WHEEL_URL" -O torch_jetson.whl || {
    echo "Failed to download PyTorch wheel"
    echo "Trying alternative version..."
    
    # Alternative for JetPack 6.0
    ALT_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl"
    wget -q --show-progress "$ALT_WHEEL" -O torch_jetson.whl
}

if [ -f torch_jetson.whl ]; then
    echo ""
    echo "📦 Installing PyTorch..."
    pip3 install torch_jetson.whl
    
    echo ""
    echo "📦 Installing torchvision..."
    wget -q "$VISION_URL" -O torchvision_jetson.whl 2>/dev/null || echo "Torchvision download optional"
    [ -f torchvision_jetson.whl ] && pip3 install torchvision_jetson.whl
    
    echo ""
    echo "🧹 Cleaning up..."
    rm -f torch_jetson.whl torchvision_jetson.whl
    
    echo ""
    echo "✅ Testing installation..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    # Simple CUDA test
    try:
        x = torch.randn(2, 3).cuda()
        y = x * 2
        print(f'CUDA compute test: Success!')
    except Exception as e:
        print(f'CUDA compute test failed: {e}')
" || echo "Installation verification failed"
else
    echo "❌ Failed to download PyTorch wheel"
    echo ""
    echo "Alternative: Use Docker with pre-installed PyTorch:"
    echo "docker pull nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3"
fi

echo ""
echo "Done!"
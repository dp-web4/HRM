#!/bin/bash
# PyTorch installation for Jetson Orin Nano with JetPack 6.2.1 (L4T 36.4.4)

echo "🚀 Installing PyTorch for Jetson Orin Nano"
echo "=========================================="
echo ""
echo "📊 System Details:"
echo "- Model: Jetson Orin Nano"
echo "- JetPack: 6.2.1 (L4T R36.4.4)"
echo "- Python: 3.10.12"
echo "- Architecture: aarch64"
echo ""

# First uninstall CPU-only version
echo "🧹 Removing existing PyTorch..."
pip3 uninstall torch torchvision torchaudio -y 2>/dev/null

echo ""
echo "📦 Installing PyTorch for JetPack 6.x..."
echo ""

# For JetPack 6.x (L4T R36.x) with Python 3.10
# Based on NVIDIA forums, we need PyTorch built for JP6
JETPACK_VERSION="60"  # JetPack 6.0+

# Method 1: Try the official pip index for Jetson
echo "Method 1: Using NVIDIA pip index..."
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v${JETPACK_VERSION}/pytorch/ || {
    echo ""
    echo "Method 1 failed. Trying direct wheel download..."
    
    # Method 2: Direct wheel download
    # PyTorch 2.3.0 for JetPack 6.0 (latest stable for Orin)
    TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl"
    
    echo "Downloading PyTorch 2.3.0 for JetPack 6.x..."
    wget -q --show-progress "$TORCH_WHEEL" -O torch_orin.whl
    
    if [ -f torch_orin.whl ]; then
        pip3 install torch_orin.whl
        rm torch_orin.whl
        
        # Also try to get torchvision
        VISION_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl"
        wget -q "$VISION_WHEEL" -O vision_orin.whl 2>/dev/null && {
            pip3 install vision_orin.whl
            rm vision_orin.whl
        }
    else
        echo "❌ Download failed"
    fi
}

echo ""
echo "✅ Verifying installation..."
python3 << 'EOF'
import sys
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} installed")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        
        # Test CUDA computation
        try:
            x = torch.randn(2, 3).cuda()
            y = x * 2
            print(f"✅ CUDA computation test passed!")
        except Exception as e:
            print(f"⚠️  CUDA computation failed: {e}")
    else:
        print("⚠️  CUDA not available - this might be wrong wheel")
        print("   Check if CUDA is properly installed")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)
EOF

echo ""
echo "💡 If CUDA is not available:"
echo "1. Check CUDA installation: ls /usr/local/cuda*"
echo "2. Check LD_LIBRARY_PATH includes CUDA libs"
echo "3. Try: export PATH=/usr/local/cuda/bin:\$PATH"
echo ""
echo "Done!"
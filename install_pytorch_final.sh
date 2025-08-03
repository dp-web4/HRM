#!/bin/bash
# Final PyTorch installation script for Jetson Orin

echo "🚀 Installing PyTorch for Jetson (Final Attempt)"
echo "=============================================="
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

echo "📊 System Info:"
cat /etc/nv_tegra_release | head -1
echo ""

echo "📦 Method 1: Installing from PyPI index..."
echo "This will download the standard PyTorch (might take time)"
echo ""

# Install with timeout and progress
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir || {
    echo ""
    echo "❌ Standard installation failed or timed out"
    echo ""
    echo "🐳 Alternative: Use Docker (RECOMMENDED)"
    echo "Run: ./setup_hrm_docker.sh"
    echo ""
    echo "Or manually install PyTorch for Jetson:"
    echo "1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "2. Download the wheel for JetPack 6.x"
    echo "3. Install with: pip3 install <downloaded_wheel.whl>"
    exit 1
}

echo ""
echo "✅ Testing installation..."
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        # Simple CUDA test
        x = torch.randn(2, 3).cuda()
        y = x * 2
        print(f"CUDA compute test: Success! Shape={y.shape}")
    except Exception as e:
        print(f"CUDA test failed: {e}")
        print("Note: This PyTorch might not be optimized for Jetson")
        print("Consider using Docker or Jetson-specific wheels")
EOF

echo ""
echo "🎉 Installation complete!"
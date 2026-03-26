#!/bin/bash
# Install PyTorch with CUDA 12.1 support (should work with CUDA 12.6)

echo "🚀 Installing PyTorch with CUDA 12.1 support"
echo "==========================================="
echo ""
echo "📊 System:"
echo "- Jetson Orin Nano"  
echo "- CUDA 12.6 installed"
echo "- Python 3.10.12"
echo ""

# Set CUDA environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "📦 Installing PyTorch 2.1.0 with CUDA 12.1..."
echo "This should be compatible with CUDA 12.6"
echo ""

# Install PyTorch with CUDA 12.1 support
pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "✅ Testing installation..."
python3 << 'EOF'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} installed")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        # Try to diagnose why
        print("\n🔍 Debugging CUDA availability:")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
        
        # Check if this is architecture mismatch
        print(f"\nPyTorch CUDA architectures: {torch.cuda.get_arch_list()}")
        print(f"PyTorch built with CUDA: {torch.version.cuda}")
    else:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
        # Simple test
        x = torch.randn(2, 3).cuda()
        print(f"✅ CUDA test passed! Tensor on: {x.device}")
        
except Exception as e:
    print(f"❌ Error: {e}")
EOF

echo ""
echo "💡 Notes:"
echo "- This is standard PyTorch, not Jetson-optimized"
echo "- May have reduced performance vs Jetson-specific builds"
echo "- But should work for testing HRM"
echo ""
echo "If CUDA still not available:"
echo "1. The wheel might not support Orin's compute capability (8.7)"
echo "2. Consider using CPU mode for initial testing"
echo "3. Or use Docker with NVIDIA runtime"
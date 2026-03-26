#!/bin/bash
# PyTorch installation for Jetson Orin Nano with JetPack 6
# Using the latest available wheel from NVIDIA

set -e

echo "=== PyTorch GPU Installation for Jetson Orin Nano ==="
echo "JetPack 6.x (L4T R36.4.4)"
echo "Installing PyTorch 2.4.0 with CUDA support"
echo ""

# Check current status
echo "Current status:"
python3 -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"
echo ""

# Set LD_LIBRARY_PATH for CUDA
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

# Uninstall existing PyTorch
echo "Removing any existing PyTorch installation..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch 2.4.0 for JetPack 6
echo ""
echo "Installing PyTorch 2.4.0 (this will download ~2GB)..."
pip3 install --no-cache-dir --verbose \
    'https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl'

# Install compatible torchvision
echo ""
echo "Installing torchvision..."
pip3 install --no-cache-dir torchvision==0.19.0

# Test the installation
echo ""
echo "=== Testing PyTorch installation ==="
python3 << 'PYTHON_EOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/aarch64-linux-gnu/nvidia:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Simple GPU test
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"GPU compute test passed: computed {z.shape} matrix multiplication")
        print("\n✅ SUCCESS! PyTorch with CUDA is working on Orin Nano!")
    except Exception as e:
        print(f"GPU test failed: {e}")
        sys.exit(1)
else:
    print("\n❌ CUDA not available!")
    print("Trying to diagnose...")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
    
    # Check if CUDA libraries exist
    import os.path
    cuda_lib = "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so"
    if os.path.exists(cuda_lib):
        print(f"✓ CUDA library found at {cuda_lib}")
    else:
        print(f"✗ CUDA library not found at {cuda_lib}")
    
    print("\nTry:")
    print("1. Add to ~/.bashrc:")
    print("   export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH")
    print("2. Then: source ~/.bashrc")
    print("3. Or reboot the system")
    sys.exit(1)
PYTHON_EOF

# Add LD_LIBRARY_PATH to bashrc if not already there
if ! grep -q "LD_LIBRARY_PATH.*nvidia" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA library path for PyTorch" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "Added LD_LIBRARY_PATH to ~/.bashrc"
fi

echo ""
echo "=== Installation complete! ==="
echo ""
echo "If CUDA is still not available:"
echo "1. Run: source ~/.bashrc"
echo "2. Or open a new terminal"
echo "3. Or reboot the system"

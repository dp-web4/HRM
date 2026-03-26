#!/bin/bash
# Fix CUDA library paths for PyTorch on Jetson

echo "🔧 Fixing CUDA library paths for PyTorch"
echo "========================================"
echo ""

# Set up environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Create symbolic links for missing libraries
echo "📎 Creating symbolic links for CUDA libraries..."

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Need sudo to create symbolic links in /usr/lib"
    SUDO="sudo"
else
    SUDO=""
fi

# Link CUDA libraries to system library path
$SUDO ln -sf /usr/local/cuda/targets/aarch64-linux/lib/libcublas.so.12 /usr/lib/aarch64-linux-gnu/libcublas.so.11 2>/dev/null
$SUDO ln -sf /usr/local/cuda/targets/aarch64-linux/lib/libcudnn.so* /usr/lib/aarch64-linux-gnu/ 2>/dev/null

# Check for nvToolsExt
if [ ! -f /usr/lib/aarch64-linux-gnu/libnvToolsExt.so.1 ]; then
    echo "⚠️  libnvToolsExt.so.1 not found - installing CUDA toolkit..."
    $SUDO apt-get update && $SUDO apt-get install -y cuda-nvtx-12-6 || echo "Failed to install nvtx"
fi

echo ""
echo "📝 Setting up environment file..."
cat > ~/cuda_env.sh << 'EOF'
# CUDA environment for PyTorch
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
EOF

echo ""
echo "✅ Testing PyTorch with CUDA..."
source ~/cuda_env.sh
python3 -c "
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/targets/aarch64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} loaded!')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        x = torch.randn(2, 2).cuda()
        print(f'✅ GPU test passed!')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "💡 To use PyTorch with CUDA, run:"
echo "   source ~/cuda_env.sh"
echo "   python3 your_script.py"
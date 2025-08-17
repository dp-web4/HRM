#!/bin/bash
# Build Flash Attention for Jetson Orin Nano
# Flash Attention provides optimized GPU kernels for transformer attention

set -e

echo "=== Building Flash Attention for Jetson Orin Nano ==="
echo "Architecture: SM 8.7 (Ampere)"
echo "CUDA: 12.6"
echo "cuDNN: 9.3.0"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found. Installing CUDA toolkit..."
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-6 cuda-nvcc-12-6
fi

# Set environment
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

echo "CUDA version:"
nvcc --version || echo "nvcc still not found"

# Clone Flash Attention
FLASH_DIR="$HOME/flash-attention"
if [ ! -d "$FLASH_DIR" ]; then
    echo ""
    echo "Cloning Flash Attention v2..."
    git clone https://github.com/Dao-AILab/flash-attention.git $FLASH_DIR
    cd $FLASH_DIR
    # Use v2.5.8 which has better Ampere support
    git checkout v2.5.8
else
    echo "Flash Attention already cloned"
    cd $FLASH_DIR
    git fetch
    git checkout v2.5.8
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --user packaging ninja

# Build Flash Attention
echo ""
echo "Building Flash Attention (this will take 30-60 minutes)..."
echo "Using architecture: SM 8.7 for Orin Nano"

# Set architecture for Orin Nano (SM 8.7)
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=4  # Limit parallel jobs to avoid OOM

# Build the package
python3 setup.py build

# Install the package
echo ""
echo "Installing Flash Attention..."
pip3 install --user -e .

# Test installation
echo ""
echo "Testing Flash Attention installation..."
python3 -c "
try:
    import flash_attn
    print(f'✅ Flash Attention version: {flash_attn.__version__}')
    
    # Test CUDA availability
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA is available')
        print(f'   Device: {torch.cuda.get_device_name(0)}')
        
        # Test flash attention import
        from flash_attn import flash_attn_func
        print('✅ Flash Attention kernel imported successfully')
    else:
        print('⚠️  CUDA not available - Flash Attention requires GPU')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "=== Build complete! ==="
echo ""
echo "Flash Attention is optimized for:"
echo "- Ampere architecture (SM 8.0+)"
echo "- Mixed precision training (FP16/BF16)"
echo "- Long sequence lengths"
echo ""
echo "To use in HRM, Flash Attention will automatically be detected and used."
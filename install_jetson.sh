#!/bin/bash
# HRM Dependencies Installation for Jetson Orin Nano
# Compatible with JetPack 6.x and CUDA 12.6

echo "🧠 HRM (Hierarchical Reasoning Model) Setup for Jetson Orin Nano"
echo "================================================================"
echo ""

# Check if we're on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: This doesn't appear to be a Jetson device"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Update package list
echo "📦 Updating package list..."
sudo apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    libopenjp2-7 \
    libpng-dev \
    libjpeg-dev \
    ninja-build \
    cmake

# Check CUDA
echo ""
echo "🔍 Checking CUDA installation..."
if command_exists nvcc; then
    echo "✅ CUDA compiler found:"
    nvcc --version
else
    echo "⚠️  CUDA compiler (nvcc) not found in PATH"
    echo "   Installing CUDA toolkit..."
    
    # Check if CUDA 12.6 is already installed
    if [ -d /usr/local/cuda-12.6 ]; then
        echo "CUDA 12.6 directory found, adding to PATH"
        export CUDA_HOME=/usr/local/cuda-12.6
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    else
        echo "⚠️  CUDA 12.6 not found. HRM requires CUDA for optimal performance."
        echo "   The model can run on CPU but will be much slower."
    fi
fi

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch for Jetson
echo ""
echo "🔥 Installing PyTorch for Jetson..."
# For JetPack 6.x, we need the right PyTorch wheel
# First try the official PyTorch index
echo "Trying official PyTorch wheels..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# If that fails, we'll need NVIDIA's Jetson-specific wheels
if ! python3 -c "import torch" 2>/dev/null; then
    echo "⚠️  Standard PyTorch installation failed, trying NVIDIA Jetson wheels..."
    
    # Download and install Jetson-specific PyTorch
    TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl"
    wget -q --show-progress $TORCH_URL -O torch_jetson.whl
    python3 -m pip install torch_jetson.whl
    rm torch_jetson.whl
fi

# Install FlashAttention dependencies
echo ""
echo "📦 Installing build dependencies for FlashAttention..."
python3 -m pip install packaging ninja

# Note about FlashAttention
echo ""
echo "ℹ️  Note: FlashAttention 3 (for Hopper/Ada GPUs) may not be compatible with Orin's Ampere GPU."
echo "   HRM should work without it, though performance may be reduced."

# Install HRM requirements
echo ""
echo "📦 Installing HRM requirements..."
cd /home/sprout/ai-workspace/HRM

# Install requirements one by one for better error handling
echo "Installing core dependencies..."
python3 -m pip install einops tqdm coolname pydantic wandb omegaconf hydra-core huggingface_hub

# Install adam-atan2 (custom optimizer)
echo "Installing adam-atan2 optimizer..."
python3 -m pip install adam-atan2

# Install argdantic
echo "Installing argdantic..."
python3 -m pip install argdantic

# Test PyTorch installation
echo ""
echo "✅ Testing PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'Device compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA not available - will run on CPU (slower)')
print(f'Number of CPU threads: {torch.get_num_threads()}')
"

# Create a simple test script
echo ""
echo "📝 Creating test script..."
cat > test_hrm_jetson.py << 'EOF'
#!/usr/bin/env python3
"""Test HRM setup on Jetson"""
import sys
import torch

print("HRM Jetson Test")
print("=" * 50)

# Check imports
try:
    import einops
    import adam_atan2
    import wandb
    import hydra
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Check GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test GPU computation
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("✅ GPU computation test passed")
else:
    device = torch.device("cpu")
    print("\n⚠️  GPU not available, using CPU")

# Estimate feasibility
print("\n📊 Feasibility Analysis:")
print(f"   HRM has 27M parameters")
print(f"   Estimated memory for inference: ~200-500 MB")
print(f"   Estimated memory for training: ~2-4 GB")

if torch.cuda.is_available():
    free_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if free_memory >= 4:
        print(f"   ✅ Sufficient GPU memory for training small batches")
    else:
        print(f"   ⚠️  Limited GPU memory - may need small batch sizes")

print("\n🎯 Recommended experiments on Jetson:")
print("   1. Sudoku solver (1K examples) - should work well")
print("   2. Small maze solving - feasible")
print("   3. ARC tasks - may need reduced batch size")
EOF

chmod +x test_hrm_jetson.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Run ./test_hrm_jetson.py to verify setup"
echo "2. Try the Sudoku demo (lightweight, perfect for Jetson):"
echo "   python dataset/build_sudoku_dataset.py --output-dir data/sudoku-jetson --subsample-size 100 --num-aug 100"
echo "3. Start training with small batch size:"
echo "   python pretrain.py data_path=data/sudoku-jetson epochs=1000 global_batch_size=32"
echo ""
echo "⚠️  Note: Jetson has limited memory compared to desktop GPUs."
echo "   Adjust batch sizes and dataset sizes accordingly."
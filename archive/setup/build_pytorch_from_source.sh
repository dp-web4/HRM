#!/bin/bash
# Build PyTorch from source for Jetson Orin Nano
# This will build against the system's cuDNN 9 and CUDA 12.6

set -e

echo "=== Building PyTorch from Source for Jetson Orin Nano ==="
echo "This will take 2-4 hours and requires ~15GB of disk space"
echo "System: JetPack 6.x, cuDNN 9.3.0, CUDA 12.6"
echo ""

# Configuration
PYTORCH_VERSION="v2.4.0"  # Stable release
BUILD_DIR="/home/sprout/pytorch-build"
INSTALL_PREFIX="/home/sprout/.local"

echo "Build directory: $BUILD_DIR"
echo "Install prefix: $INSTALL_PREFIX"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Step 1: Install build dependencies
echo ""
echo "=== Step 1: Installing build dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-yaml \
    python3-typing-extensions \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    gfortran \
    libomp-dev \
    cuda-toolkit-12-6 \
    cuda-nvcc-12-6

# Install Python dependencies
pip3 install --user \
    numpy \
    pyyaml \
    typing-extensions \
    dataclasses \
    ninja \
    setuptools \
    cmake \
    cffi

# Step 2: Set up environment variables
echo ""
echo "=== Step 2: Setting up environment variables ==="
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0  # Disable NCCL for single GPU
export USE_DISTRIBUTED=0  # Disable distributed for single GPU
export USE_QNNPACK=0  # Disable QNNPACK (x86 only)
export USE_PYTORCH_QNNPACK=0
export TORCH_CUDA_ARCH_LIST="8.7"  # Orin Nano is SM 8.7
export CUDA_HOME=/usr/local/cuda-12.6
export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu
export CUDNN_INCLUDE_DIR=/usr/include
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which python3))/../"}
export PYTHON_BIN_PATH=$(which python3)
export USE_MKLDNN=0  # Disable MKL-DNN (x86 only)
export MAX_JOBS=4  # Limit parallel jobs to avoid OOM

# Add CUDA to PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "CUDNN_LIB_DIR: $CUDNN_LIB_DIR"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "MAX_JOBS: $MAX_JOBS"

# Step 3: Clone PyTorch repository
echo ""
echo "=== Step 3: Cloning PyTorch repository ==="
if [ -d "$BUILD_DIR" ]; then
    echo "Build directory exists. Pulling latest changes..."
    cd $BUILD_DIR
    git fetch
    git checkout $PYTORCH_VERSION
else
    echo "Cloning PyTorch $PYTORCH_VERSION..."
    git clone --recursive --branch $PYTORCH_VERSION https://github.com/pytorch/pytorch.git $BUILD_DIR
    cd $BUILD_DIR
fi

# Update submodules
git submodule sync
git submodule update --init --recursive

# Step 4: Apply patches for Jetson if needed
echo ""
echo "=== Step 4: Applying Jetson-specific configurations ==="

# Create a custom build configuration
cat > .pytorch_build_config.txt << 'CONFIG_EOF'
USE_CUDA=1
USE_CUDNN=1
CUDA_HOME=/usr/local/cuda-12.6
CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu
CUDNN_INCLUDE_DIR=/usr/include
TORCH_CUDA_ARCH_LIST=8.7
MAX_JOBS=4
USE_NCCL=0
USE_DISTRIBUTED=0
USE_MKLDNN=0
USE_QNNPACK=0
BUILD_TEST=0
CONFIG_EOF

# Step 5: Build PyTorch
echo ""
echo "=== Step 5: Building PyTorch (this will take 2-4 hours) ==="
echo "Starting build at $(date)"
echo "You can monitor system resources with: tegrastats"
echo ""

# Clean any previous builds
python3 setup.py clean || true

# Build and install
python3 setup.py develop --user

# Step 6: Build torchvision (optional but recommended)
echo ""
echo "=== Step 6: Building torchvision ==="
read -p "Build torchvision too? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    VISION_DIR="$HOME/torchvision-build"
    if [ ! -d "$VISION_DIR" ]; then
        git clone --recursive --branch v0.19.0 https://github.com/pytorch/vision.git $VISION_DIR
    fi
    cd $VISION_DIR
    python3 setup.py develop --user
fi

# Step 7: Test the installation
echo ""
echo "=== Step 7: Testing PyTorch installation ==="
cd $HOME
python3 << 'PYTHON_EOF'
import torch
import sys

print("=" * 50)
print("PyTorch Build Test")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Memory info
    print(f"\nMemory Info:")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Performance test
    print(f"\nPerformance Test:")
    size = 1024
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    
    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    flops = 2 * size**3 * 100 / elapsed / 1e9  # GFLOPS
    print(f"Matrix multiplication ({size}x{size}): {flops:.2f} GFLOPS")
    
    print("\n✅ SUCCESS! PyTorch with CUDA is working perfectly!")
else:
    print("\n❌ CUDA not available")
    print("Check LD_LIBRARY_PATH and CUDA installation")
    sys.exit(1)
PYTHON_EOF

echo ""
echo "=== Build Complete! ==="
echo "Built at: $(date)"
echo ""
echo "PyTorch is now installed from source with cuDNN 9 support!"
echo "Location: $BUILD_DIR"
echo ""
echo "To use in other projects, make sure to:"
echo "export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:\$LD_LIBRARY_PATH"
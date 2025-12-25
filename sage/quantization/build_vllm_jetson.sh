#!/bin/bash
# vLLM Build Script for Jetson AGX Thor
# Based on lessons learned from FP4 quantization journey

set -e  # Exit on error

echo "========================================================================="
echo "vLLM Build for Jetson AGX Thor - Optimized for Qwen3-Omni FP4"
echo "========================================================================="
echo ""

# Configuration
VLLM_VERSION="v0.12.0"  # Required for vLLM-Omni
VLLM_SOURCE_DIR="/home/dp/ai-workspace/vllm-source"
BUILD_LOG="/tmp/vllm_build_$(date +%Y%m%d_%H%M%S).log"

echo "📋 Build Configuration:"
echo "  vLLM Version: $VLLM_VERSION"
echo "  Source Dir: $VLLM_SOURCE_DIR"
echo "  Build Log: $BUILD_LOG"
echo ""

# Step 1: Environment Setup
echo "🔧 Step 1: Setting up Jetson Thor environment..."
export CUDA_HOME=/usr/local/cuda-13.0
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# vLLM specific environment variables
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=4  # Limit parallel jobs to avoid OOM on Jetson
export TORCH_CUDA_ARCH_LIST="9.0"  # SM100/Blackwell architecture for Thor

echo "  ✅ CUDA_HOME: $CUDA_HOME"
echo "  ✅ CUDACXX: $CUDACXX"
echo "  ✅ TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo ""

# Verify CUDA installation
if [ ! -f "$CUDACXX" ]; then
    echo "❌ Error: nvcc not found at $CUDACXX"
    exit 1
fi

CUDA_VERSION=$($CUDACXX --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
echo "  CUDA Version: $CUDA_VERSION"
echo ""

# Step 2: Navigate to vLLM source
echo "📂 Step 2: Preparing vLLM source..."
cd "$VLLM_SOURCE_DIR"

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo "❌ Error: $VLLM_SOURCE_DIR is not a git repository"
    exit 1
fi

echo "  Current branch: $(git branch --show-current)"
echo "  Current commit: $(git log --oneline -1)"
echo ""

# Step 3: Checkout v0.12.0
echo "🔀 Step 3: Checking out vLLM $VLLM_VERSION..."

# Stash any changes
if ! git diff --quiet; then
    echo "  Stashing local changes..."
    git stash
fi

# Fetch latest tags
echo "  Fetching tags..."
git fetch --tags 2>&1 | grep -v "From\|remote:" || true

# Checkout v0.12.0
echo "  Checking out $VLLM_VERSION..."
if git checkout tags/$VLLM_VERSION -b build-$VLLM_VERSION-jetson 2>/dev/null; then
    echo "  ✅ Created new branch build-$VLLM_VERSION-jetson"
elif git checkout $VLLM_VERSION 2>/dev/null; then
    echo "  ✅ Checked out $VLLM_VERSION"
else
    echo "  ⚠️  Tag $VLLM_VERSION not found, checking available tags..."
    git tag | grep "v0.12" | head -10
    echo ""
    echo "  Trying v0.12.0..."
    git checkout v0.12.0 || {
        echo "❌ Failed to checkout v0.12.0. Please choose from available tags above."
        exit 1
    }
fi

echo "  Current version: $(git describe --tags)"
echo ""

# Step 4: Clean previous builds
echo "🧹 Step 4: Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info
find . -type d -name  "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.so" -delete 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✅ Cleaned"
echo ""

# Step 5: Install build dependencies
echo "📦 Step 5: Installing build dependencies..."
pip install --break-system-packages --upgrade pip setuptools wheel 2>&1 | tail -3
pip install --break-system-packages ninja cmake 2>&1 | tail -3
echo "  ✅ Build tools installed"
echo ""

# Step 6: Build vLLM
echo "🔨 Step 6: Building vLLM (this will take 30-60 minutes)..."
echo "  Output logged to: $BUILD_LOG"
echo "  Monitor: tail -f $BUILD_LOG"
echo ""

START_TIME=$(date +%s)

# Build with pip install -e . (editable mode)
echo "  Starting build at $(date)"
pip install --break-system-packages -v -e . > "$BUILD_LOG" 2>&1

BUILD_EXIT_CODE=$?
END_TIME=$(date +%s)
BUILD_DURATION=$((END_TIME - START_TIME))

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "  ✅ vLLM built successfully in ${BUILD_DURATION}s"
else
    echo "  ❌ Build failed with exit code $BUILD_EXIT_CODE"
    echo "  Last 50 lines of build log:"
    tail -50 "$BUILD_LOG"
    exit $BUILD_EXIT_CODE
fi
echo ""

# Step 7: Verify vLLM installation
echo "✅ Step 7: Verifying vLLM installation..."
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
    echo "❌ vLLM import failed"
    exit 1
}
python3 -c "import vllm; print(f'vLLM CUDA available: {vllm.envs.VLLM_USE_CUDA}')" || true
echo ""

# Step 8: Install vLLM-Omni
echo "🎯 Step 8: Installing vLLM-Omni..."
pip install --break-system-packages vllm-omni 2>&1 | tail -10

if python3 -c "import vllm_omni" 2>/dev/null; then
    echo "  ✅ vLLM-Omni installed successfully"
else
    echo "  ⚠️  vLLM-Omni installation may have issues"
fi
echo ""

# Step 9: Final verification
echo "🎊 Step 9: Final verification..."
echo ""
echo "vLLM Installation:"
python3 -c "
import vllm
import sys
print(f'  Version: {vllm.__version__}')
print(f'  Location: {vllm.__file__}')
try:
    import vllm_omni
    print(f'  vLLM-Omni: ✅ Available')
except ImportError:
    print(f'  vLLM-Omni: ❌ Not found')
"

echo ""
echo "========================================================================="
echo "✅ BUILD COMPLETE"
echo "========================================================================="
echo ""
echo "📊 Summary:"
echo "  vLLM Version: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "  Build Time: ${BUILD_DURATION}s"
echo "  Build Log: $BUILD_LOG"
echo ""
echo "🚀 Next Steps:"
echo "  1. Test with original Qwen3-Omni model"
echo "  2. Test with FP4 quantized model"
echo "  3. Benchmark performance"
echo ""
echo "📝 Test commands saved to: sage/quantization/test_vllm_deployment.sh"
echo "========================================================================="

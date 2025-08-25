#!/bin/bash
# SAGE Portable Environment Setup Script
# Works across WSL2, Jetson Orin Nano, and Legion RTX 4090

set -e  # Exit on error

echo "====================================="
echo "SAGE Portable Environment Setup"
echo "====================================="

# Detect platform
PLATFORM="unknown"
GPU_TYPE="none"
CUDA_VERSION="none"

# Check if we're on Jetson
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    echo "✓ Detected: Jetson platform"
elif [ -f /proc/version ] && grep -q Microsoft /proc/version; then
    PLATFORM="wsl2"
    echo "✓ Detected: WSL2 platform"
else
    PLATFORM="linux"
    echo "✓ Detected: Standard Linux platform"
fi

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    GPU_TYPE=$(nvidia-smi -q | grep "Product Name" | head -1 | cut -d: -f2 | xargs)
    echo "✓ GPU: $GPU_TYPE"
    echo "✓ CUDA: $CUDA_VERSION"
else
    echo "⚠ No NVIDIA GPU detected"
fi

# Create virtual environment
ENV_NAME="sage_env"
if [ -d "$ENV_NAME" ]; then
    echo "⚠ Virtual environment $ENV_NAME already exists, recreating..."
    rm -rf $ENV_NAME
fi

python3 -m venv $ENV_NAME
echo "✓ Created virtual environment: $ENV_NAME"

# Activate environment
source $ENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install base requirements
echo "Installing base requirements..."
pip install -r sage_requirements_minimal.txt

# Platform-specific PyTorch installation
echo "Installing PyTorch for platform..."
case $PLATFORM in
    jetson)
        # Jetson-specific PyTorch
        echo "Installing PyTorch for Jetson..."
        # Use pre-built wheels from NVIDIA
        pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    wsl2|linux)
        # Desktop GPU PyTorch
        if [[ $CUDA_VERSION == "12."* ]]; then
            echo "Installing PyTorch for CUDA 12.x..."
            pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ $CUDA_VERSION == "11."* ]]; then
            echo "Installing PyTorch for CUDA 11.x..."
            pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            echo "Installing CPU-only PyTorch..."
            pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
        ;;
esac

# Test PyTorch installation
echo "Testing PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA compute capability: {torch.cuda.get_device_capability(0)}')
"

# Create activation script
cat > activate_sage.sh << 'EOF'
#!/bin/bash
# Quick activation script for SAGE environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/sage_env/bin/activate"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "SAGE environment activated!"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
EOF
chmod +x activate_sage.sh

echo ""
echo "====================================="
echo "✅ SAGE environment setup complete!"
echo "====================================="
echo ""
echo "To activate the environment:"
echo "  source sage_env/bin/activate"
echo "Or use the quick activation script:"
echo "  source ./activate_sage.sh"
echo ""
echo "Platform: $PLATFORM"
echo "GPU: $GPU_TYPE"
echo "CUDA: $CUDA_VERSION"
echo ""
#!/bin/bash
# Setup CUDA and PyTorch for Jetson Orin Nano
# JetPack 6.x (L4T R36.4.4)

set -e

echo "=== Jetson Orin Nano CUDA & PyTorch Setup ==="
echo "Model: Jetson Orin Nano"
echo "JetPack: 6.x (L4T R36.4.4)"
echo "Python: 3.10"
echo ""

# Step 1: Install CUDA development kit
echo "=== Step 1: Installing CUDA development kit ==="
echo "This will install CUDA binaries and nvcc compiler"
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6 nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc

# Step 2: Set up CUDA environment
echo ""
echo "=== Step 2: Setting up CUDA environment ==="
echo "Adding CUDA to PATH and LD_LIBRARY_PATH"

# Add to current session
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12

# Add to bashrc for persistence
if ! grep -q "CUDA_HOME=/usr/local/cuda" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA environment" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda-12/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-12" >> ~/.bashrc
fi

# Verify CUDA installation
echo ""
echo "=== Step 3: Verifying CUDA installation ==="
if [ -f /usr/local/cuda-12/bin/nvcc ]; then
    /usr/local/cuda-12/bin/nvcc --version
else
    echo "Warning: nvcc not found, trying to create symlink..."
    sudo ln -sf /usr/local/cuda-12 /usr/local/cuda
fi

# Step 4: Install PyTorch for Orin Nano
echo ""
echo "=== Step 4: Installing PyTorch for Orin Nano ==="
echo "Reinstalling torch if needed..."

# First check if we need to reinstall
python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && {
    echo "PyTorch with CUDA already working!"
} || {
    echo "Installing PyTorch with CUDA support..."
    
    # Uninstall existing PyTorch if present
    pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
    
    # For Orin Nano with JetPack 6, we need the specific wheel
    # Using the JP6.0 wheel which is compatible with R36.4.4
    echo "Downloading PyTorch 2.3.0 for JetPack 6..."
    
    # This is the correct URL for JetPack 6.0 (compatible with 6.x)
    wget -q --show-progress -O torch-2.3.0-jp6.whl \
        "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+40ec155e58.nv24.05.14710581-cp310-cp310-linux_aarch64.whl"
    
    if [ -f torch-2.3.0-jp6.whl ]; then
        echo "Installing PyTorch wheel..."
        pip3 install torch-2.3.0-jp6.whl
        rm torch-2.3.0-jp6.whl
        
        # Install compatible torchvision
        pip3 install torchvision==0.18.0
    else
        echo "Error: Failed to download PyTorch wheel"
        echo "Trying alternative: Installing from NVIDIA's index"
        pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 torch torchvision
    fi
}

# Step 5: Verify everything works
echo ""
echo "=== Step 5: Final verification ==="
python3 << EOF
import sys
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test a simple operation
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        c = a + b
        print(f"✓ GPU compute test: {c.cpu().numpy()}")
        print("\n🎉 SUCCESS! PyTorch with CUDA is working on Orin Nano!")
    else:
        print("\n⚠️  CUDA not available. You may need to:")
        print("1. Reboot the system")
        print("2. Source ~/.bashrc")
        print("3. Check nvidia-smi")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "If CUDA is still not available, try:"
echo "1. source ~/.bashrc"
echo "2. sudo reboot"
echo "3. Check: nvidia-smi"
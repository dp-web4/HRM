#!/usr/bin/env python3
"""
Diagnose CUDA initialization issues
"""

import os
import sys
import subprocess

print("🔍 CUDA Diagnostics")
print("=" * 60)

# Check environment
print("\n1. Environment Variables:")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Check Python and PyTorch
print("\n2. Python Environment:")
print(f"   Python: {sys.version}")
print(f"   Executable: {sys.executable}")

try:
    import torch
    print(f"\n3. PyTorch Info:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA Built: {torch.version.cuda}")
    print(f"   cuDNN: {torch.backends.cudnn.version()}")
    
    # Try different CUDA operations
    print("\n4. CUDA Operations:")
    
    # Basic check
    try:
        cuda_available = torch.cuda.is_available()
        print(f"   torch.cuda.is_available(): {cuda_available}")
    except Exception as e:
        print(f"   ❌ cuda.is_available() failed: {e}")
    
    # Device count
    try:
        device_count = torch.cuda.device_count()
        print(f"   torch.cuda.device_count(): {device_count}")
    except Exception as e:
        print(f"   ❌ cuda.device_count() failed: {e}")
    
    # Current device
    try:
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f"   torch.cuda.current_device(): {current_device}")
    except Exception as e:
        print(f"   ❌ cuda.current_device() failed: {e}")
    
    # Try to create a tensor
    print("\n5. Tensor Creation Test:")
    try:
        # CPU tensor first
        cpu_tensor = torch.randn(2, 2)
        print(f"   ✓ CPU tensor created: shape {cpu_tensor.shape}")
        
        # GPU tensor
        if torch.cuda.is_available():
            gpu_tensor = torch.randn(2, 2, device='cuda')
            print(f"   ✓ GPU tensor created: shape {gpu_tensor.shape}")
        else:
            print("   ⚠️  Skipping GPU tensor (CUDA not available)")
    except Exception as e:
        print(f"   ❌ Tensor creation failed: {e}")
    
except ImportError as e:
    print(f"\n❌ PyTorch import failed: {e}")

# Check system CUDA
print("\n6. System CUDA Check:")
try:
    # nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,cuda_version', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   nvidia-smi output: {result.stdout.strip()}")
    else:
        print(f"   ❌ nvidia-smi failed: {result.stderr}")
except Exception as e:
    print(f"   ❌ nvidia-smi error: {e}")

# Check CUDA libraries
print("\n7. CUDA Library Paths:")
cuda_paths = [
    "/usr/local/cuda",
    "/usr/local/cuda-12.6",
    "/usr/local/cuda-12.8",
    "/usr/lib/x86_64-linux-gnu",
]

for path in cuda_paths:
    if os.path.exists(path):
        print(f"   ✓ Found: {path}")
        # Check for libcudart
        libcudart = os.path.join(path, "lib64", "libcudart.so")
        if os.path.exists(libcudart):
            print(f"     - libcudart.so found")
    else:
        print(f"   ✗ Not found: {path}")

print("\n" + "=" * 60)
print("💡 Diagnosis Summary:")
print("   If CUDA is not available, try:")
print("   1. export CUDA_HOME=/usr/local/cuda-12.6")
print("   2. export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")
print("   3. Restart Python/Jupyter kernel")
print("   4. Check driver compatibility with nvidia-smi")
print("=" * 60)
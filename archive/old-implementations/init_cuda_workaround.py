#!/usr/bin/env python3
"""
CUDA initialization workaround
"""

import os
import sys

# Force CUDA initialization through environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

# Try alternate initialization
print("🔧 Attempting CUDA workaround...")

# Method 1: Try with numba if available
try:
    from numba import cuda
    cuda.detect()
    print("✓ Numba CUDA detection successful")
except Exception as e:
    print(f"✗ Numba not available: {e}")

# Method 2: Reset CUDA context
try:
    import cupy
    cupy.cuda.runtime.deviceReset()
    print("✓ CuPy device reset successful")
except Exception as e:
    print(f"✗ CuPy not available: {e}")

# Method 3: Direct cudart call
try:
    import torch
    # Try to force CUDA initialization
    torch.cuda.init()
    print(f"✓ PyTorch CUDA init: {torch.cuda.is_initialized()}")
    
    # Try device synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("✓ CUDA synchronize successful")
        device_props = torch.cuda.get_device_properties(0)
        print(f"✓ GPU: {device_props.name}")
except Exception as e:
    print(f"✗ PyTorch CUDA error: {e}")

# Method 4: Check for common issues
print("\n🔍 Checking common issues:")

# Check if running in container
if os.path.exists('/.dockerenv'):
    print("⚠️  Running in Docker container - may need --gpus all")

# Check if running over SSH
if 'SSH_CONNECTION' in os.environ:
    print("⚠️  Running over SSH - may need X11 forwarding")

# Check persistence mode
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '-pm'], capture_output=True, text=True)
    if 'Persistence Mode' in result.stdout:
        print("✓ NVIDIA persistence mode available")
except:
    pass

print("\n💡 Suggestions:")
print("1. Try: sudo nvidia-smi -pm 1  (enable persistence mode)")
print("2. Try: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm")
print("3. Try: sudo systemctl restart nvidia-persistenced")
print("4. Check dmesg for NVIDIA errors: dmesg | grep -i nvidia")
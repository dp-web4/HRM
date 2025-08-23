#!/usr/bin/env python3
"""
PyTorch CUDA workaround for Jetson Orin Nano
Handles missing library issues
"""

import os
import sys
import warnings

# Set up CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/targets/aarch64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')

# Suppress the specific library warning
os.environ['LD_PRELOAD'] = ''  # Clear any preloads that might conflict

try:
    # Try importing torch with error handling
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        import torch
        
    print(f"✅ PyTorch {torch.__version__} loaded successfully!")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        
        # Test GPU computation
        try:
            x = torch.randn(3, 3).cuda()
            y = x @ x.T
            print(f"✅ GPU computation test passed!")
            print(f"Result shape: {y.shape}")
            print(f"Result device: {y.device}")
        except Exception as e:
            print(f"⚠️ GPU computation failed: {e}")
    else:
        # If CUDA not available, try to diagnose
        print("\n🔍 Debugging CUDA availability:")
        print(f"PyTorch built with CUDA: {torch.version.cuda}")
        print(f"CUDA runtime version: {torch._C._cuda_getCompiledVersion()}")
        
        # Check if it's a library issue
        if hasattr(torch._C, '_cuda_isDriverSufficient'):
            print(f"Driver sufficient: {torch._C._cuda_isDriverSufficient()}")
            
        # For now, we can still use CPU
        print("\n💡 Using CPU mode for now")
        device = torch.device('cpu')
        x = torch.randn(3, 3)
        y = x @ x.T
        print(f"✅ CPU computation works: {y.shape}")

except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")
    print("\n💡 Troubleshooting steps:")
    print("1. Check if PyTorch is installed: pip3 list | grep torch")
    print("2. Install missing CUDA libraries:")
    print("   sudo apt-get install cuda-libraries-12-6")
    print("3. Set LD_LIBRARY_PATH:")
    print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    sys.exit(1)

print("\n✅ PyTorch is ready for use!")
#!/usr/bin/env python3
"""
Test SAGE Portable Environment
Validates core functionality across platforms
"""

import sys
import platform
import subprocess

def test_basic_imports():
    """Test basic Python imports"""
    results = {}
    
    # Standard library
    try:
        import math
        import random
        import time
        results['stdlib'] = 'OK'
    except ImportError as e:
        results['stdlib'] = f'FAIL: {e}'
    
    # NumPy
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        results['numpy'] = f'OK (v{np.__version__})'
    except ImportError:
        results['numpy'] = 'Not installed'
    except Exception as e:
        results['numpy'] = f'Error: {e}'
    
    # OpenCV
    try:
        import cv2
        results['opencv'] = f'OK (v{cv2.__version__})'
    except ImportError:
        results['opencv'] = 'Not installed'
    
    # PyTorch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            results['pytorch'] = f'OK (v{torch.__version__}, GPU: {device_name})'
        else:
            results['pytorch'] = f'OK (v{torch.__version__}, CPU only)'
    except ImportError:
        results['pytorch'] = 'Not installed'
    except Exception as e:
        results['pytorch'] = f'Error: {e}'
    
    return results

def test_irp_modules():
    """Test IRP module imports"""
    results = {}
    
    # Base IRP
    try:
        from sage.irp.base import IRPPlugin, IRPState
        results['irp_base'] = 'OK'
    except ImportError as e:
        results['irp_base'] = f'Import error: {e}'
    except Exception as e:
        results['irp_base'] = f'Error: {e}'
    
    # Vision plugin
    try:
        from sage.irp.plugins.vision_impl import create_vision_irp
        results['vision_irp'] = 'OK'
    except ImportError:
        results['vision_irp'] = 'Not available'
    except Exception as e:
        results['vision_irp'] = f'Error: {e}'
    
    return results

def test_gpu_info():
    """Get GPU information"""
    info = {}
    
    # NVIDIA SMI
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', 
                               '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            info['gpu'] = parts[0].strip()
            info['memory'] = parts[1].strip()
            info['driver'] = parts[2].strip()
            info['compute'] = parts[3].strip()
        else:
            info['gpu'] = 'No NVIDIA GPU detected'
    except Exception:
        info['gpu'] = 'nvidia-smi not available'
    
    return info

def main():
    """Run all tests"""
    print("=" * 60)
    print("SAGE Portable Environment Test")
    print("=" * 60)
    print()
    
    # System info
    print("System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Machine: {platform.machine()}")
    print()
    
    # GPU info
    print("GPU Information:")
    gpu_info = test_gpu_info()
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test imports
    print("Core Libraries:")
    import_results = test_basic_imports()
    for lib, status in import_results.items():
        symbol = "✓" if status.startswith("OK") else "✗"
        print(f"  {symbol} {lib}: {status}")
    print()
    
    # Test IRP modules
    print("IRP Modules:")
    irp_results = test_irp_modules()
    for module, status in irp_results.items():
        symbol = "✓" if status == "OK" else "✗"
        print(f"  {symbol} {module}: {status}")
    print()
    
    # Summary
    all_ok = all(v.startswith("OK") or v == "Not installed" 
                 for v in {**import_results, **irp_results}.values())
    
    if all_ok:
        print("✅ Environment is ready for SAGE!")
    else:
        print("⚠️  Some components need attention")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
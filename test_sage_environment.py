#!/usr/bin/env python3
"""
Comprehensive SAGE Environment Test
Validates environment is ready across all platforms
"""

import sys
import os

def test_torch_cuda():
    """Test PyTorch with CUDA"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test basic CUDA operation
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print(f"✓ CUDA computation test passed")
            
            return True
        else:
            print("⚠ CUDA not available, CPU mode only")
            return False
            
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def test_opencv():
    """Test OpenCV"""
    try:
        import cv2
        import numpy as np
        print(f"✓ OpenCV {cv2.__version__} installed")
        
        # Test basic operation
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"✓ OpenCV operations working")
        return True
        
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False

def test_irp_imports():
    """Test IRP module imports"""
    success = True
    
    # Add HRM to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from sage.irp.base import IRPPlugin, IRPState
        print("✓ IRP base modules imported")
    except ImportError as e:
        print(f"✗ IRP base import failed: {e}")
        success = False
    
    try:
        from sage.irp.plugins.vision_impl import create_vision_irp
        print("✓ Vision IRP plugin available")
    except ImportError:
        print("⚠ Vision IRP plugin not available (optional)")
    except Exception as e:
        print(f"⚠ Vision IRP error: {e}")
    
    return success

def test_attention_demo():
    """Test the attention demo can run"""
    try:
        import visual_monitor.simple_attention_demo as demo
        print("✓ Attention demo module loads")
        return True
    except ImportError as e:
        print(f"⚠ Attention demo not available: {e}")
        return False

def main():
    print("=" * 70)
    print("SAGE ENVIRONMENT VALIDATION")
    print("=" * 70)
    print()
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print()
    
    print("Core Libraries:")
    print("-" * 40)
    
    # Test numpy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} installed")
    except ImportError:
        print("✗ NumPy not installed")
    
    # Test scipy
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__} installed")
    except ImportError:
        print("⚠ SciPy not installed (optional)")
    
    print()
    print("GPU Support:")
    print("-" * 40)
    cuda_ok = test_torch_cuda()
    
    print()
    print("Computer Vision:")
    print("-" * 40)
    cv_ok = test_opencv()
    
    print()
    print("IRP Framework:")
    print("-" * 40)
    irp_ok = test_irp_imports()
    
    print()
    print("Demos:")
    print("-" * 40)
    demo_ok = test_attention_demo()
    
    print()
    print("=" * 70)
    if cuda_ok and cv_ok and irp_ok:
        print("✅ ENVIRONMENT READY FOR SAGE!")
        print("   Full GPU acceleration available")
    elif cv_ok and irp_ok:
        print("✅ ENVIRONMENT READY FOR SAGE!")
        print("   (CPU mode - install CUDA toolkit for GPU)")
    else:
        print("⚠️  Some components need attention")
        print("   Run: pip install -r sage_requirements.txt")
    print("=" * 70)

if __name__ == "__main__":
    main()
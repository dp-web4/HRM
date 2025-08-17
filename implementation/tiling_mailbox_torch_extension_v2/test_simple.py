#!/usr/bin/env python3
"""
Simple test to verify mailbox extension loads and works.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mailbox_ext
    print("✓ Extension loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load extension: {e}")
    sys.exit(1)

def test_initialization():
    """Test mailbox initialization."""
    print("\n=== Testing Initialization ===")
    
    # Test peripheral mailbox init
    try:
        record_stride = 64
        capacity = 1024
        pbm_ptrs = mailbox_ext.pbm_init(record_stride, capacity)
        pbm_hdr = int(pbm_ptrs[0].item())
        pbm_payload = int(pbm_ptrs[1].item())
        print(f"✓ PBM initialized: header={hex(pbm_hdr)}, payload={hex(pbm_payload)}")
    except Exception as e:
        print(f"✗ PBM init failed: {e}")
        return False
    
    # Test focus tensor mailbox init
    try:
        ftm_capacity = 256
        ftm_ptrs = mailbox_ext.ftm_init(ftm_capacity)
        ftm_hdr = int(ftm_ptrs[0].item())
        ftm_ring = int(ftm_ptrs[1].item())
        print(f"✓ FTM initialized: header={hex(ftm_hdr)}, ring={hex(ftm_ring)}")
    except Exception as e:
        print(f"✗ FTM init failed: {e}")
        return False
    
    return True, (pbm_hdr, pbm_payload), (ftm_hdr, ftm_ring)

def test_peripheral_ops(pbm_ptrs):
    """Test peripheral mailbox push/pop."""
    print("\n=== Testing Peripheral Mailbox ===")
    pbm_hdr, pbm_payload = pbm_ptrs
    
    # Create test data
    test_data = torch.arange(64, dtype=torch.uint8, device='cuda')
    print(f"Created test tensor: shape={test_data.shape}, device={test_data.device}")
    
    # Test push
    try:
        success = mailbox_ext.pbm_push_bytes_cuda(pbm_hdr, pbm_payload, test_data)
        print(f"✓ Push operation: {success}")
    except Exception as e:
        print(f"✗ Push failed: {e}")
        return False
    
    # Test pop
    try:
        max_records = 10
        record_stride = 64
        result = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, max_records, record_stride)
        print(f"✓ Pop operation: got {result.numel()} bytes")
    except Exception as e:
        print(f"✗ Pop failed: {e}")
        return False
    
    return True

def test_focus_ops(ftm_ptrs):
    """Test focus tensor mailbox operations."""
    print("\n=== Testing Focus Tensor Mailbox ===")
    ftm_hdr, ftm_ring = ftm_ptrs
    
    # Create a test tensor
    test_tensor = torch.randn(32, 32, device='cuda')
    dev_ptr = test_tensor.data_ptr()
    
    # Test push
    try:
        shape = [32, 32, 1, 1]
        stride = [32, 1, 1, 1]
        ndim = 2
        dtype = 2  # FTM_DTYPE_F32
        tag = 42
        ttl = 100
        
        success = mailbox_ext.ftm_push_ptr(
            ftm_hdr, ftm_ring, dev_ptr,
            shape, stride, ndim, dtype, tag, ttl
        )
        print(f"✓ Push tensor pointer: {success}")
    except Exception as e:
        print(f"✗ Push failed: {e}")
        return False
    
    # Test pop
    try:
        result = mailbox_ext.ftm_pop(ftm_hdr, ftm_ring)
        print(f"✓ Pop tensor metadata: got {len(result)} fields")
        for key in result:
            print(f"  - {key}: {result[key]}")
    except Exception as e:
        print(f"✗ Pop failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Mailbox Extension Test")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False
    
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test initialization
    result = test_initialization()
    if not result:
        return False
    
    success, pbm_ptrs, ftm_ptrs = result
    
    # Test peripheral mailbox
    if not test_peripheral_ops(pbm_ptrs):
        return False
    
    # Test focus tensor mailbox
    if not test_focus_ops(ftm_ptrs):
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
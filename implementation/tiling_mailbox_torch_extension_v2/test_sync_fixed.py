#!/usr/bin/env python3
"""
Test with GPT's recommended synchronization patterns.
Validates count-based pop and proper sync handling.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mailbox_ext

def test_pbm_count_based_pop():
    """Test GPT's recommended PBM pattern with count-based pop."""
    print("\n=== Testing Count-Based PBM Pop ===")
    
    # Initialize
    record_stride = 64
    capacity = 1024
    pbm_ptrs = mailbox_ext.pbm_init(record_stride, capacity)
    pbm_hdr = int(pbm_ptrs[0].item())
    pbm_payload = int(pbm_ptrs[1].item())
    
    # Push 16 records as GPT suggests
    print("Pushing 16 records...")
    for i in range(16):
        rec = torch.full((64,), i, dtype=torch.uint8, device='cuda')
        success = mailbox_ext.pbm_push_bytes_cuda(pbm_hdr, pbm_payload, rec)
        assert success, f"Push {i} failed"
    
    # Pop in two batches of 8
    print("Popping first 8 records...")
    out1 = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 8, 64)
    
    print("Popping next 8 records...")
    out2 = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 8, 64)
    
    # Sync as GPT recommends
    torch.cuda.synchronize()
    
    # Validate
    assert out1.numel() == 8*64, f"Expected 512 bytes, got {out1.numel()}"
    assert out2.numel() == 8*64, f"Expected 512 bytes, got {out2.numel()}"
    
    # Check data integrity
    out1_cpu = out1.cpu().reshape(8, 64)
    out2_cpu = out2.cpu().reshape(8, 64)
    
    for i in range(8):
        expected = i
        actual = out1_cpu[i, 0].item()
        assert actual == expected, f"Record {i}: expected {expected}, got {actual}"
    
    for i in range(8):
        expected = i + 8
        actual = out2_cpu[i, 0].item()
        assert actual == expected, f"Record {i+8}: expected {expected}, got {actual}"
    
    print("✓ Count-based pop working correctly")
    print(f"✓ Data integrity verified (0-15 in order)")
    
    # Test empty pop
    print("\nTesting empty pop...")
    out3 = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 8, 64)
    torch.cuda.synchronize()
    assert out3.numel() == 0, f"Expected 0 bytes from empty mailbox, got {out3.numel()}"
    print("✓ Empty pop returns zero-size tensor")
    
    return True

def test_ftm_with_sync():
    """Test FTM with proper synchronization."""
    print("\n=== Testing FTM with Synchronization ===")
    
    # Initialize
    ftm_capacity = 256
    ftm_ptrs = mailbox_ext.ftm_init(ftm_capacity)
    ftm_hdr = int(ftm_ptrs[0].item())
    ftm_ring = int(ftm_ptrs[1].item())
    
    # Create multiple test tensors
    tensors = []
    for i in range(5):
        t = torch.full((16, 16), float(i), device='cuda')
        tensors.append(t)
        
        # Push tensor metadata
        success = mailbox_ext.ftm_push_ptr(
            ftm_hdr, ftm_ring, t.data_ptr(),
            [16, 16, 1, 1], [16, 1, 1, 1],
            2, 2, i, 100-i  # tag=i, ttl=100-i
        )
        assert success, f"Push {i} failed"
    
    # Pop all tensors back
    print("Popping 5 tensor records...")
    for i in range(5):
        result = mailbox_ext.ftm_pop(ftm_hdr, ftm_ring)
        
        # Verify metadata
        tag = result["meta"][2].item()
        ttl = result["ttl"][0].item()
        shape0 = result["shape"][0].item()
        shape1 = result["shape"][1].item()
        
        assert tag == i, f"Expected tag {i}, got {tag}"
        assert ttl == 100-i, f"Expected ttl {100-i}, got {ttl}"
        assert shape0 == 16 and shape1 == 16, f"Shape mismatch"
        
        print(f"  Record {i}: tag={tag}, ttl={ttl}, shape=[{shape0},{shape1}]")
    
    # Test empty pop
    print("\nTesting empty FTM pop...")
    result = mailbox_ext.ftm_pop(ftm_hdr, ftm_ring)
    dev_ptr = result["dev_ptr"][0].item()
    assert dev_ptr == 0, "Expected null pointer from empty mailbox"
    print("✓ Empty pop returns zero dev_ptr")
    
    return True

def test_concurrent_patterns():
    """Test concurrent push/pop patterns."""
    print("\n=== Testing Concurrent Patterns ===")
    
    # Initialize
    pbm_ptrs = mailbox_ext.pbm_init(256, 1024)
    pbm_hdr = int(pbm_ptrs[0].item())
    pbm_payload = int(pbm_ptrs[1].item())
    
    # Interleaved push/pop
    print("Testing interleaved operations...")
    for round in range(3):
        # Push batch
        for i in range(10):
            rec = torch.full((256,), round*10 + i, dtype=torch.uint8, device='cuda')
            mailbox_ext.pbm_push_bytes_cuda(pbm_hdr, pbm_payload, rec)
        
        # Pop half
        out = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 5, 256)
        assert out.numel() == 5*256, f"Round {round}: Expected 1280 bytes"
        
        # Verify first value
        first_val = out[0].item()
        expected = round * 10
        assert first_val == expected, f"Round {round}: Expected {expected}, got {first_val}"
    
    # Final cleanup pop
    remaining = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 100, 256)
    torch.cuda.synchronize()
    
    expected_remaining = 15  # 3 rounds * 5 remaining each
    actual_remaining = remaining.numel() // 256
    assert actual_remaining == expected_remaining, f"Expected {expected_remaining} records, got {actual_remaining}"
    
    print(f"✓ Interleaved pattern working")
    print(f"✓ Cleanup retrieved {actual_remaining} remaining records")
    
    return True

def main():
    """Run all synchronization tests."""
    print("=" * 60)
    print("Synchronization Fix Validation")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False
    
    tests = [
        ("Count-Based PBM Pop", test_pbm_count_based_pop),
        ("FTM with Sync", test_ftm_with_sync),
        ("Concurrent Patterns", test_concurrent_patterns),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All synchronization fixes validated!")
        print("GPT's debug notes were spot-on!")
    
    return all(s for _, s in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
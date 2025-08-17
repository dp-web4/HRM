
# CUDA Mailbox Extension Debug Notes

## Context

During initial runs of the tiling mailbox Torch extension, `pbm_pop` was returning zeros. Compilation and linking were successful, but this revealed a subtle synchronization issue. This document captures the fixes and reasoning so future developers understand what was happening.

## Root Cause

- CUDA kernels launch asynchronously.  
- Our `pbm_pop` kernel wrote results on a stream, but Python attempted to read the returned tensor immediately, before the kernel finished.  
- The binding also did not return the actual **count of popped records**, so consumers could not distinguish “no records” from “buffer not filled yet.”

## Fixes

### 1. Return Pop Count

Modify the kernel to return how many records were popped:

```cpp
__global__ void k_pbm_pop(PBM_Header* h, uint8_t* payload,
                          uint8_t* dst, int max_records, int record_stride,
                          int* d_count) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint32_t out_bytes = 0;
    int got = pbm_try_pop_bulk(h, payload, dst, record_stride, max_records, &out_bytes);
    *d_count = got;
  }
}
```

In the binding:

```cpp
auto d_count = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
pbm_pop_kernel_launch(..., d_count.data_ptr<int>(), stream.stream());

auto h_count = d_count.cpu().item<int>();  // syncs stream for scalar
auto trimmed = out.narrow(0, 0, h_count * record_stride);
return trimmed;
```

### 2. Use CUDA Stream Guard

Ensure kernels run on the same stream PyTorch expects:

```cpp
auto stream = at::cuda::getCurrentCUDAStream();
at::cuda::CUDAStreamGuard guard(stream);
```

### 3. Explicit Synchronization

- In **tests**, call `torch.cuda.synchronize()` after push/pop cycles.  
- In bindings, only synchronize when fetching scalars back to CPU (like `d_count`). Returning CUDA tensors remains async-safe.

## Recommended Test

```python
for i in range(16):
    rec = torch.full((64,), i, dtype=torch.uint8, device='cuda')
    mailbox_ext.pbm_push_bytes_cuda(pbm_hdr, pbm_payload, rec)

out1 = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 8, 64)
out2 = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 8, 64)
torch.cuda.synchronize()

assert out1.numel() == 8*64
assert out2.numel() == 8*64
```

## Next Small Wins

- Use a pinned host scalar for `count` (avoids implicit sync on `.cpu()`).  
- Introduce CUDA events for producer/consumer handoff instead of manual synchronizes.  
- Add stream priorities (focus vs peripheral).  
- Keep profiler check (`test_profile.py`) active and flip memcpy warning into a hard assert.

## Status

- Infrastructure compiles and links ✅  
- Push/Pop kernels run on GPU ✅  
- Remaining work: **sync handling, telemetry, and event wiring**.

---

**Takeaway:** The zeros were not a logic bug, just missing synchronization and record count reporting. Fixes above make the mailbox extension robust for async GPU execution.

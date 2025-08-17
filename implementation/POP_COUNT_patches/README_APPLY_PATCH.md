
# Patch: Pop Count + Stream Guard for CUDA Mailboxes

This patch fixes the "zeros on pop" behavior by:
1) Returning the **count** of records popped from the CUDA kernel, and
2) Guarding launches with PyTorch's **current CUDA stream** so everything runs on the expected stream.
The binding then **narrows** the returned buffer to the actual size and syncs only for the tiny scalar `count`.

## Files Patched
- `tiling_mailbox_torch_extension_v2/src/mailbox_kernels.cu`
- `tiling_mailbox_torch_extension_v2/src/mailbox_ext.cpp`

## How to Apply
From the directory **containing** `tiling_mailbox_torch_extension_v2/`:

### Using `git apply` (preferred if repo is a git checkout)
```bash
git apply POP_COUNT_SYNC_FIX.patch
```

### Using `patch`
```bash
patch -p0 < POP_COUNT_SYNC_FIX.patch
```

> If you moved files or changed paths, adjust the `-p` level accordingly.

## Rebuild
```bash
cd tiling_mailbox_torch_extension_v2
export TORCH_CUDA_ARCH_LIST="8.7"     # or your Jetson CC
python setup.py build_ext --inplace
```

## Verify
Run both tests:
```bash
python test_push_pop.py
python test_profile.py
```
Expected:
- `pbm_pop_bulk_cuda` returns a **non-empty** CUDA tensor sized to `count * record_stride`.
- `test_profile.py` prints **"No memcpy H<->D"** during the push/pop cycle.
- Focus mailbox pointer round-trips exactly (pointer equality).

## Rationale
- CUDA kernels are asynchronous; returning a **count** allows the binding to know how many records are valid.
- Moving a 4-byte count scalar to CPU is a tiny, acceptable sync point while keeping bulk data **async on device**.
- `CUDAStreamGuard` ensures launches happen on PyTorch's current stream, avoiding cross-stream hazards.

## Next Steps (Optional)
- Use a **pinned host scalar** and `cudaMemcpyAsync` for the count to avoid implicit sync on `.cpu()`.
- Switch producer/consumer handoff to **CUDA events**.
- Add **stream priorities** (focus > peripheral) to match SAGE policy.
- Promote profiler memcpy warnings to **hard asserts** in CI.

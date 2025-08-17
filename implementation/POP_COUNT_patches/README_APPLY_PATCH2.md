
# Patch 2: Pinned Host Scalar + cudaMemcpyAsync for Pop Count

This patch removes the implicit `.cpu()` synchronization when reading the pop `count`.  
Instead, it uses **pinned host memory** and `cudaMemcpyAsync` on the **current CUDA stream** to transfer the 4-byte scalar.

## What Changes
- Adds pinned host tensor allocation for the count (`pinned_memory=True`)
- Calls `cudaMemcpyAsync(..., stream)` to copy the scalar without blocking other streams
- Performs a **stream-local synchronize** right before slicing the return buffer (keeps correctness; still lighter than a global sync)

> Advanced: You can keep the binding fully async by *returning the pinned count tensor* alongside the data buffer and letting Python narrow after awaiting; see `src/README.md` note in the patch.

## Apply
From the directory containing `tiling_mailbox_torch_extension_v2/`:
```bash
git apply POP_COUNT_ASYNC_PINNED.patch
# or: patch -p0 < POP_COUNT_ASYNC_PINNED.patch
```

## Rebuild
```bash
cd tiling_mailbox_torch_extension_v2
export TORCH_CUDA_ARCH_LIST="8.7"
python setup.py build_ext --inplace
```

## Verify
```bash
python test_push_pop.py
python test_profile.py
```
Expect:
- No implicit sync via `.cpu()` in the binding path.
- Functionality identical (correct data, correct sizes).
- Profiler still shows **no host↔device memcpys** except the intentional 4-byte count copy.

## Notes
- This is a surgical change layered atop the previous pop-count patch; apply that one first if you haven't.

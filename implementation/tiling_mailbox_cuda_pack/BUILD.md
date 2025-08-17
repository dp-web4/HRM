
# BUILD.md — CUDA Mailbox Skeleton (Jetson-friendly)

This is a **minimal CUDA skeleton** for the two mailbox classes described in the tiling notes:
- `PeripheralBroadcastMailbox` (fixed-size records, many→many)
- `FocusTensorMailbox` (pointer-based zero-copy, few→few)

It compiles two small static libraries you can link into CUDA apps/tests.

## Prereqs
- CUDA toolkit installed (`/usr/local/cuda`)
- CMake ≥ 3.18
- A Jetson Orin (SM 8.7) or adjust for your GPU

## Build
```bash
mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=87 ..    # change to 72 for Xavier, etc.
cmake --build . -j2
```

Artifacts:
- `libpbm.a` — peripheral mailbox device/host helpers
- `libftm.a` — focus mailbox device/host helpers

> This is a **skeleton**: device functions are simple (byte copies, no vectorization).
> Add alignment, vector loads/stores, and error handling for production.

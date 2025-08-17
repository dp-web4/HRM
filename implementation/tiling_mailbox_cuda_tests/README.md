
# CUDA Toy Tests — Peripheral & Focus Mailboxes

Minimal end-to-end tests that **do not depend on PyTorch**. Good for sanity checks on Jetson.

## Build
```bash
mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=87 ..   # adjust to your GPU (e.g., 72 for Xavier)
cmake --build . -j2
```

## Run
```bash
./test_pbm
./test_ftm
```

Expected: both tests print PASS with counts > 0.
```


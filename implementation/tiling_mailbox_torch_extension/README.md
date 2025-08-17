
# PyTorch Extension — Mailbox Bindings (Minimal)

Exposes mailbox init/push/pop via a Torch extension so you can prototype in Python.

## Build (in this folder)
```bash
python setup.py build_ext --inplace
```

If you hit arch issues, set:
```bash
export TORCH_CUDA_ARCH_LIST="8.7"   # Or your Jetson/Xavier cap
```

## Test
```bash
python test_ext.py
```
Expected: peripheral pop count > 0 and a focus record round-trips.

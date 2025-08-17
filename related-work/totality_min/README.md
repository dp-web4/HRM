
# totality_min — Minimal Totality/SubThought Testbed

A tiny, dependency-free prototype of a "Totality/SubThought" world model to experiment with **schemes**, **canvases**, **activation**, and simple **imagination/augmentation**. Designed to pair with SAGE as a **Cognitive Sensor**.

## Files
- `totality_core.py` — in-memory store and core types (Scheme, Canvas, Link) and ops (read, imagine, activate, write)
- `transforms.py` — tiny helpers for context shifts and semantic variation (placeholder for real transforms)
- `demo.py` — runs a small scenario end-to-end
- `tests/test_totality.py` — basic unit tests

## Quickstart

```bash
python totality_min/demo.py
```

Expected output: seed schemes, imagined canvases, activation change, and a committed abstraction.

## Run tests

```bash
python -m pytest -q
```

(If `pytest` isn't available, you can just inspect the store after running `demo.py`.)

## Notes
- No external dependencies. Replace the naive imagination ops with your own augmentation engine as needed.
- This is intentionally *minimal* so it can run on constrained devices (Jetson, etc.).

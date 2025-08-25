# Jetson TinyVAE Test Plan (Motion-Crop → TinyVAE → Telemetry)

## Goals
- Verify end‑to‑end pipeline on Jetson (IMX219 → attention crop → TinyVAE → telemetry).
- Compare FP16 vs FP32 latency and stability.
- Validate trust metric behavior across static vs motion scenes.
- Produce CSV logs for later analysis and SNARC ingestion.

## Hardware/Env
- Jetson (Nano/Orin) with CSI cameras
- CUDA available
- Python 3.10+, PyTorch with CUDA
- OpenCV installed with GStreamer support

## Scenarios
1. **Static scene** (no motion): expect low attention regions, stable latents, higher trust_kl.
2. **Slow motion** (hand sweep): attention bbox tracks; stable latency, small recon_error.
3. **Rapid motion**: larger crops, recon_error increases; trust_recon dips; time_ms stable.
4. **Variable lighting**: check normalization toggle effects on recon_error/latent_norm.

## Switches
- `use_fp16={True, False}`
- `normalize={True, False}` with configurable `mean/std`
- `latent_dim={32, 64}` (optional sweep)
- Input channels: RGB (default)

## Metrics (per frame)
- time_ms
- reconstruction_error
- kl_divergence
- trust
- latent_norm
- crop bbox (x, y, w, h)
- mode flags (fp16, normalize, latent_dim)

## Pass/Fail Heuristics
- time_ms median < 6–8 ms on Orin Nano with FP16, < 12–15 ms FP32 (crop 64×64)
- trust in static scene > 0.6 (tunable), doesn’t clip to 0
- No crashes switching FP16 on/off, steady GPU memory usage

## Procedure
1. Run smoke test (`tinyvae_smoketest.py`).
2. Run bench (`run_jetson_bench.py`) in each scenario for ~30s.
3. Inspect CSV in `/tmp/tinyvae/bench_*.csv` and quick console summary.
4. Optionally replay latents and recon in a later session to visualize fidelity.

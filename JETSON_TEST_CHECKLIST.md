# SAGE on Jetson — Quick Test Checklist
**Target:** Run the IRP demos on a Jetson (Nano/Xavier/Orin).  
**Artifacts written by this checklist:** JSONL telemetry + `tegrastats` logs.

---

## 0) Pre-flight
- Ensure you’re on a recent JetPack (TensorRT/CUDA/CuDNN).  
- Confirm GPU is visible:
  ```bash
  python3 - << 'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
  ```
- Optional perf mode (adjust per device):
  ```bash
  sudo nvpmodel -q          # list modes
  sudo nvpmodel -m 0        # pick max perf mode for your board
  sudo jetson_clocks        # lock clocks for stable measurements
  ```
  > If `nvpmodel`/`jetson_clocks` aren’t present, skip or install `nvpmodel` tools for your board.

## 1) Environment knobs
Set conservative defaults first:
```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDNN_V8_API_ENABLE=1
```

Prefer FP16:
- Use your demo flags (e.g., `--fp16`) or set `torch.set_float32_matmul_precision("high")` inside the demo.

## 2) Start `tegrastats` and run demos
From repo root (adjust paths if different):
```bash
mkdir -p out
tegrastats --interval 500 --logfile out/tegrastats.log &
TEGRA_PID=$!

# Vision latent IRP (example flags; tweak to your demo)
python3 demos/vision_latent_irp.py   --fp16 --max-steps 16 --early-stop --jsonl out/vision.jsonl

# Language span-mask IRP
python3 demos/language_span_mask_irp.py   --fp16 --max-steps 12 --early-stop --jsonl out/language.jsonl

kill $TEGRA_PID
sync
```

## 3) Acceptance gates (edit to your dataset)
- **Vision:** early-stop saves ≥×2 compute vs max-steps with <1% mIoU drop.  
- **Language:** meaning-latent stabilizes ≤N steps with no significant drop in answer EM/F1.

## 4) Quick telemetry sanity (JSONL)
If you have `tools/parse_telemetry.py`:
```bash
python3 tools/parse_telemetry.py out/vision.jsonl
python3 tools/parse_telemetry.py out/language.jsonl
```

You should see:
- Monotonicity ratio ~0.8–1.0 on `E`.
- Negative dE median < 0 (descending energy).
- Steps << max-steps when early-stop triggers.

## 5) What to record
- Commit hash, demo flags, device model (`/proc/device-tree/model`), JetPack version (`dpkg -l | grep 'nvidia-l4t-core'`).  
- Ambient temp / cooling notes (thermal throttling can skew results).

## 6) Troubleshooting
- **CUDA OOM:** lower `--batch-size`, reduce `--max-steps`, use smaller latent, ensure swap is enabled as last resort.  
- **Thermal throttling:** add a fan/heat-sink, reduce steps, raise interval between runs.  
- **Throughput regressions:** confirm `jetson_clocks`, FP16 on, and that TensorRT isn’t inadvertently off for your ops.

---

**Tip:** Keep JSONL + `tegrastats` logs in the PR so the diff includes both quality and energy-side evidence.

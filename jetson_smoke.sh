#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-out}"
VISION_DEMO="${VISION_DEMO:-demos/vision_latent_irp.py}"
LANG_DEMO="${LANG_DEMO:-demos/language_span_mask_irp.py}"
VISION_FLAGS="${VISION_FLAGS:---fp16 --max-steps 16 --early-stop}"
LANG_FLAGS="${LANG_FLAGS:---fp16 --max-steps 12 --early-stop}"

mkdir -p "$OUT_DIR"

# Try to start tegrastats if present
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 500 --logfile "$OUT_DIR/tegrastats.log" &
  TEGRA_PID=$!
else
  echo "[warn] tegrastats not found; continuing without it" >&2
  TEGRA_PID=""
fi

cleanup() {
  if [[ -n "${TEGRA_PID}" ]]; then
    kill "${TEGRA_PID}" || true
  fi
}
trap cleanup EXIT

echo "[info] Running vision demo..."
python3 "$VISION_DEMO" $VISION_FLAGS --jsonl "$OUT_DIR/vision.jsonl"

echo "[info] Running language demo..."
python3 "$LANG_DEMO" $LANG_FLAGS --jsonl "$OUT_DIR/language.jsonl"

echo "[info] Done. Outputs in $OUT_DIR"

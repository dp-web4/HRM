#!/bin/bash
# run.sh — Launch HRM experiment with standardized resume protocol and logging.

# ---- User-editable fields ----
EXP_NAME="EXP-01_baseline"
CKPT_DIR="runs/${EXP_NAME}/checkpoints"   # directory with hrm_arc_step_*.pt
RESUME=""                                 # override with specific ckpt path
LOGDIR="runs/${EXP_NAME}_$(date +%Y%m%d-%H%M)"
STATUS_JSON="${LOGDIR}/status.json"

# ---- Setup ----
mkdir -p "$LOGDIR/checkpoints" "$LOGDIR/logs"

# ---- Pick resume checkpoint ----
if [ -z "$RESUME" ] && [ -d "$CKPT_DIR" ]; then
  RESUME=$(python resume_helper.py --ckpt_dir "$CKPT_DIR" --dry-run | grep Selected | awk '{print $3}')
fi

echo "[run.sh] Using resume checkpoint: $RESUME"

# ---- Run training ----
PYTHONUNBUFFERED=1 \
python -u train_arc_full_nova.py \
  --exp_name "$EXP_NAME" \
  --resume "$RESUME" \
  --optimizer_policy load \
  --schedule_policy load \
  --allow_partial \
  --status_json_path "$STATUS_JSON" \
  2>&1 | tee "$LOGDIR/logs/train.log"

# Notes:
# - train_arc_full_nova.py should import resume_helper and call load_resume_bundle()
# - Logs + checkpoints will go under $LOGDIR
# - status.json will be refreshed every 60s with heartbeat_writer()

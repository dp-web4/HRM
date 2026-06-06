#!/usr/bin/env bash
# Orchestrator for S143 (thermal message-shape) -> S142 (persona ablation re-run).
# Launched via `systemd-run --user --scope` so it lives in the user slice, NOT the
# autonomous-thor-sage.service cgroup. This is the fix for the recurring "detached
# experiment dies at session end" loop: setsid does NOT escape systemd's default
# KillMode=control-group teardown, but a transient user scope does (same mechanism
# the raising sessions already use via thor-raising.service).
set -u
cd "$(dirname "$0")"
LOG=chain_$(date +%Y%m%d_%H%M%S).log
exec > "$LOG" 2>&1
echo "[chain] start $(date)"

# 1. Wait for any active raising session to free the GPU (avoid contention timeouts
#    that corrupted prior runs). Poll for the raising python process by name.
while pgrep -f "ollama_raising_session" >/dev/null 2>&1; do
  echo "[chain] $(date) waiting for raising session to free GPU"
  sleep 30
done
echo "[chain] $(date) GPU free; settling 20s"
sleep 20

# 2. S143 to completion.
echo "[chain] $(date) S143 start"
python3 s143_thermal_message_shape.py
s143_done=$(python3 -c "import json;print(json.load(open('s143_thermal_message_shape.json')).get('complete'))" 2>/dev/null)
echo "[chain] $(date) S143 complete=$s143_done"

# 3. S142 re-run to completion (only if GPU still free; raising may have re-armed).
while pgrep -f "ollama_raising_session" >/dev/null 2>&1; do
  echo "[chain] $(date) waiting for raising session before S142"
  sleep 30
done
sleep 20
echo "[chain] $(date) S142 start"
python3 s142_addendum_ablation.py
s142_done=$(python3 -c "import json;print(json.load(open('s142_addendum_ablation.json')).get('complete'))" 2>/dev/null)
echo "[chain] $(date) S142 complete=$s142_done"

echo "[chain] DONE $(date) s143=$s143_done s142=$s142_done"
touch CHAIN_DONE

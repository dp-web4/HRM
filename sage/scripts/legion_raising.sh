#!/bin/bash
# Legion SAGE raising session + auto-commit
# Runs a raising session, commits results, pushes to origin.
# Designed to run via systemd timer every 6 hours.

set -e

HRM_DIR="/home/dp/ai-workspace/HRM"
PYTHONPATH="$HRM_DIR"
export PYTHONPATH
PYTHON="/home/dp/miniforge3/bin/python3"

cd "$HRM_DIR"

echo "[Legion-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

# Ensure daemon is running via systemd (not the manual ensure_daemon.sh)
if ! systemctl --user is-active sage-daemon.service >/dev/null 2>&1; then
    echo "[Legion-Raising] Daemon not running, starting via systemctl..."
    systemctl --user start sage-daemon.service
    sleep 5
fi

# Verify daemon health
HEALTH=$(curl -s --max-time 5 http://localhost:8750/health 2>/dev/null || echo "")
if echo "$HEALTH" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='alive'" 2>/dev/null; then
    echo "[Legion-Raising] Daemon healthy"
else
    echo "[Legion-Raising] WARNING: Daemon health check failed, attempting restart..."
    systemctl --user restart sage-daemon.service
    sleep 10
fi

# Pull latest code
git pull --rebase --quiet 2>/dev/null || echo "[Legion-Raising] WARN: git pull failed, continuing with current code"

# Run the raising session (continue from last session number)
$PYTHON -m sage.raising.scripts.legion_raising_session -c 2>&1

# Check if there are new results to commit
CHANGED=0

# Check for new/modified session transcripts
if ! git diff --quiet sage/raising/sessions/legion/ 2>/dev/null; then
    CHANGED=1
fi
if [ -n "$(git ls-files --others --exclude-standard sage/raising/sessions/legion/ 2>/dev/null)" ]; then
    CHANGED=1
fi

# Check for updated identity state
if ! git diff --quiet sage/raising/state/legion_identity.json 2>/dev/null; then
    CHANGED=1
fi

# Check for updated experience buffer
if ! git diff --quiet sage/raising/state/experience_buffer_legion_qwen2_0.5b.json 2>/dev/null; then
    CHANGED=1
fi
if [ -n "$(git ls-files --others --exclude-standard sage/raising/state/experience_buffer_legion_qwen2_0.5b.json 2>/dev/null)" ]; then
    CHANGED=1
fi

if [ "$CHANGED" -eq 0 ]; then
    echo "[Legion-Raising] No new raising data to commit."
    exit 0
fi

# Read session number from identity state
SESSION_NUM=$($PYTHON -c "
import json
with open('$HRM_DIR/sage/raising/state/legion_identity.json') as f:
    print(json.load(f)['identity']['session_count'])
" 2>/dev/null || echo "?")

PHASE=$($PYTHON -c "
import json
with open('$HRM_DIR/sage/raising/state/legion_identity.json') as f:
    print(json.load(f)['development']['phase_name'])
" 2>/dev/null || echo "?")

# Stage and commit
git add sage/raising/sessions/legion/ \
        sage/raising/state/legion_identity.json \
        sage/raising/state/experience_buffer_legion_qwen2_0.5b.json

git commit -m "[Legion-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Automated SAGE-Legion raising session via OllamaIRP
Machine: Legion (Legion Pro 7, RTX 4090, Linux)
Model: Qwen 2 0.5B (alibaba-qwen family)
Phase: $PHASE
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

# Push (with rebase to handle race conditions from other machines)
git pull --rebase --quiet 2>/dev/null || true
git push origin main
echo "[Legion-Raising] Session $SESSION_NUM committed and pushed."

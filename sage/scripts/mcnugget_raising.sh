#!/bin/bash
# McNugget SAGE raising session + auto-commit
# Runs a raising session, commits results, pushes to origin.
# Designed to run via launchd every 6 hours.

set -e

HRM_DIR="/Users/dennispalatov/repos/HRM"
PYTHONPATH="$HRM_DIR"
export PYTHONPATH

cd "$HRM_DIR"

echo "[McNugget-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

# Ensure daemon is running and up-to-date (also pulls latest code)
source "$HRM_DIR/sage/scripts/ensure_daemon.sh"
echo "[McNugget-Raising] Daemon: version=$SAGE_DAEMON_VERSION updated=$SAGE_DAEMON_UPDATED"

# Run the raising session (continue from last session number)
/opt/homebrew/bin/python3 -m sage.raising.scripts.mcnugget_raising_session -c 2>&1

# Instance directory (new layout) and legacy paths
INSTANCE_DIR="sage/instances/mcnugget-gemma3-12b"
LEGACY_STATE="sage/raising/state"
LEGACY_SESSIONS="sage/raising/sessions/mcnugget"

# Check if there are new results to commit
CHANGED=0

# Check instance dir (new layout)
if [ -d "$INSTANCE_DIR" ]; then
    if ! git diff --quiet "$INSTANCE_DIR/" 2>/dev/null; then
        CHANGED=1
    fi
    if [ -n "$(git ls-files --others --exclude-standard "$INSTANCE_DIR/" 2>/dev/null)" ]; then
        CHANGED=1
    fi
fi

# Check legacy paths (transition period)
if ! git diff --quiet "$LEGACY_SESSIONS/" 2>/dev/null; then
    CHANGED=1
fi
if [ -n "$(git ls-files --others --exclude-standard "$LEGACY_SESSIONS/" 2>/dev/null)" ]; then
    CHANGED=1
fi
if ! git diff --quiet "$LEGACY_STATE/mcnugget_identity.json" 2>/dev/null; then
    CHANGED=1
fi
if ! git diff --quiet "$LEGACY_STATE/experience_buffer_mcnugget_gemma3_12b.json" 2>/dev/null; then
    CHANGED=1
fi

if [ "$CHANGED" -eq 0 ]; then
    echo "[McNugget-Raising] No new raising data to commit."
    exit 0
fi

# Read session number from identity state (prefer instance dir)
IDENTITY_FILE="$INSTANCE_DIR/identity.json"
if [ ! -f "$IDENTITY_FILE" ]; then
    IDENTITY_FILE="$LEGACY_STATE/mcnugget_identity.json"
fi

SESSION_NUM=$(/opt/homebrew/bin/python3 -c "
import json
with open('$HRM_DIR/$IDENTITY_FILE') as f:
    print(json.load(f)['identity']['session_count'])
" 2>/dev/null || echo "?")

PHASE=$(/opt/homebrew/bin/python3 -c "
import json
with open('$HRM_DIR/$IDENTITY_FILE') as f:
    print(json.load(f)['development']['phase_name'])
" 2>/dev/null || echo "?")

# Stage — instance dir (new) + legacy paths (transition)
git add "$INSTANCE_DIR/" 2>/dev/null || true
git add "$LEGACY_SESSIONS/" \
        "$LEGACY_STATE/mcnugget_identity.json" \
        "$LEGACY_STATE/experience_buffer_mcnugget_gemma3_12b.json" 2>/dev/null || true

git commit -m "[McNugget-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Automated SAGE-McNugget raising session via OllamaIRP
Machine: McNugget (Mac Mini M4)
Model: Gemma 3 12B (google-gemma family)
Phase: $PHASE
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

# Push
git push origin main
echo "[McNugget-Raising] Session $SESSION_NUM committed and pushed."

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
/opt/homebrew/bin/python3 -m sage.raising.scripts.ollama_raising_session --machine mcnugget -c 2>&1

# Instance directory
INSTANCE_DIR="sage/instances/mcnugget-gemma3-12b"
SNAPSHOT_DIR="$INSTANCE_DIR/snapshots"

# Snapshot live state files (gitignored) to tracked snapshots/ dir
if [ -d "$INSTANCE_DIR" ]; then
    mkdir -p "$HRM_DIR/$SNAPSHOT_DIR"
    for f in identity.json experience_buffer.json peer_trust.json daemon_state.json; do
        if [ -f "$HRM_DIR/$INSTANCE_DIR/$f" ]; then
            cp "$HRM_DIR/$INSTANCE_DIR/$f" "$HRM_DIR/$SNAPSHOT_DIR/$f"
        fi
    done
    echo "[McNugget-Raising] State snapshot saved to $SNAPSHOT_DIR/"
fi

# Check if there are new results to commit
CHANGED=0

# Check instance dir sessions + snapshots
if [ -d "$INSTANCE_DIR" ]; then
    if ! git diff --quiet "$INSTANCE_DIR/" 2>/dev/null; then
        CHANGED=1
    fi
    if [ -n "$(git ls-files --others --exclude-standard "$INSTANCE_DIR/" 2>/dev/null)" ]; then
        CHANGED=1
    fi
fi

if [ "$CHANGED" -eq 0 ]; then
    echo "[McNugget-Raising] No new raising data to commit."
    exit 0
fi

# Read session number from identity state
IDENTITY_FILE="$INSTANCE_DIR/identity.json"

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

# Stage instance dir (sessions + snapshots, gitignored files excluded automatically)
git add "$INSTANCE_DIR/" 2>/dev/null || true

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

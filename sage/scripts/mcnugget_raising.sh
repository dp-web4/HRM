#!/bin/bash
# McNugget SAGE raising session + auto-commit
# Runs a raising session, snapshots state, commits results, pushes to origin.
# Designed to run via launchd every 6 hours.

set -e

HRM_DIR="/Users/dennispalatov/repos/HRM"
PYTHONPATH="$HRM_DIR"
export PYTHONPATH

cd "$HRM_DIR"

echo "[McNugget-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

# Pull latest before running (avoid conflicts)
git pull --rebase origin main 2>&1 || {
    echo "[McNugget-Raising] WARNING: git pull failed, continuing with local state"
}

# Ensure daemon is running and up-to-date
source "$HRM_DIR/sage/scripts/ensure_daemon.sh"
echo "[McNugget-Raising] Daemon: version=$SAGE_DAEMON_VERSION updated=$SAGE_DAEMON_UPDATED"

# Run the raising session (continue from last session number)
/opt/homebrew/bin/python3 -m sage.raising.scripts.ollama_raising_session --machine mcnugget -c 2>&1

# Instance directory
INSTANCE_DIR="sage/instances/mcnugget-gemma3-12b"
SNAPSHOT_DIR="$INSTANCE_DIR/snapshots"

# Snapshot live state files into git-tracked snapshots/ directory
# Uses Python script for archive history + metadata
echo "[McNugget-Raising] Snapshotting state..."
/opt/homebrew/bin/python3 -m sage.scripts.snapshot_state --machine mcnugget

# Read session number and phase from live identity
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

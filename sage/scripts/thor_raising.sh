#!/bin/bash
# Thor SAGE raising session + auto-commit
# Runs a raising session with qwen3.5:27b via Ollama
# Manual execution for now - can be automated later

set -e

SAGE_DIR="/home/dp/ai-workspace/SAGE"
PYTHONPATH="$SAGE_DIR"
export PYTHONPATH

cd "$SAGE_DIR"

echo "[Thor-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

# Pull latest before running (avoid conflicts)
git pull --rebase origin main 2>&1 || {
    echo "[Thor-Raising] WARNING: git pull failed, continuing with local state"
}

# Run the raising session (continue from last session number)
python3 -m sage.raising.scripts.ollama_raising_session --machine thor -c 2>&1

# Instance directory
INSTANCE_DIR="sage/instances/thor-qwen3.5-27b"
SNAPSHOT_DIR="$INSTANCE_DIR/snapshots"

# Snapshot live state files into git-tracked snapshots/ directory
echo "[Thor-Raising] Snapshotting state..."
python3 -m sage.scripts.snapshot_state --machine thor 2>/dev/null || {
    echo "[Thor-Raising] Snapshot script not found, skipping"
}

# Read session number and phase from live identity
IDENTITY_FILE="$INSTANCE_DIR/identity.json"
SESSION_NUM=$(python3 -c "
import json
try:
    with open('$SAGE_DIR/$IDENTITY_FILE') as f:
        print(json.load(f).get('identity', {}).get('session_count', '?'))
except:
    print('?')
" 2>/dev/null || echo "?")

PHASE=$(python3 -c "
import json
try:
    with open('$SAGE_DIR/$IDENTITY_FILE') as f:
        print(json.load(f).get('development', {}).get('phase_name', '?'))
except:
    print('?')
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
    echo "[Thor-Raising] No new raising data to commit."
    exit 0
fi

# Stage instance dir (sessions + snapshots, gitignored files excluded automatically)
git add "$INSTANCE_DIR/" 2>/dev/null || true

git commit -m "[Thor-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Automated thor raising session via OllamaIRP
Machine: Thor (Jetson AGX Thor)
Model: Qwen 3.5 27B (alibaba-qwen family)
Phase: $PHASE
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

# Push
git push origin main
echo "[Thor-Raising] Session $SESSION_NUM committed and pushed."

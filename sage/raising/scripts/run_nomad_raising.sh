#!/bin/bash
# Nomad SAGE raising session + auto-commit
# Runs a raising session, snapshots state, commits results, pushes to origin.
# Schedule: every 6 hours via crontab (0,6,12,18).

set -e

SAGE_DIR="/mnt/c/projects/ai-agents/SAGE"
export PYTHONPATH="$SAGE_DIR"
LOG_DIR="/tmp/nomad-raising-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/raising-$(date +%Y%m%d-%H%M).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[Nomad-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

cd "$SAGE_DIR"

# --- Step 1: Pull latest code ---
echo "[Nomad-Raising] Pulling latest code..."
git pull --ff-only origin main 2>&1 || {
    echo "[Nomad-Raising] WARNING: git pull --ff-only failed, trying rebase..."
    git pull --rebase origin main 2>&1 || {
        echo "[Nomad-Raising] WARNING: git pull failed, continuing with local state"
    }
}

# --- Step 2: Ensure daemon is running and up-to-date ---
export SAGE_PORT="${SAGE_PORT:-8750}"
export SAGE_NO_BROWSER=1
source "$SAGE_DIR/sage/scripts/ensure_daemon.sh"
echo "[Nomad-Raising] Daemon: version=$SAGE_DAEMON_VERSION running=$SAGE_DAEMON_RUNNING updated=$SAGE_DAEMON_UPDATED"

# --- Step 3: Run the raising session ---
echo "[Nomad-Raising] Running raising session..."
python3 -m sage.raising.scripts.ollama_raising_session \
    --machine nomad \
    --model gemma3:4b \
    --turns 6 \
    -c 2>&1

# --- Step 4: Snapshot state ---
INSTANCE_DIR="sage/instances/nomad-gemma3-4b"

echo "[Nomad-Raising] Snapshotting state..."
python3 -m sage.scripts.snapshot_state --machine nomad 2>&1 || {
    echo "[Nomad-Raising] WARNING: snapshot_state failed, continuing"
}

# Read session number and phase from identity
IDENTITY_FILE="$INSTANCE_DIR/identity.json"
SESSION_NUM=$(python3 -c "
import json
with open('$SAGE_DIR/$IDENTITY_FILE') as f:
    print(json.load(f)['identity']['session_count'])
" 2>/dev/null || echo "?")

PHASE=$(python3 -c "
import json
with open('$SAGE_DIR/$IDENTITY_FILE') as f:
    print(json.load(f)['development']['phase_name'])
" 2>/dev/null || echo "?")

# --- Step 5: Regenerate fleet snapshot ---
echo "[Nomad-Raising] Updating SESSION_FOCUS.md..."
python3 -m sage.scripts.generate_primer 2>/dev/null || true

# --- Step 6: Commit and push ---
CHANGED=0
if [ -d "$INSTANCE_DIR" ]; then
    if ! git diff --quiet "$INSTANCE_DIR/" 2>/dev/null; then
        CHANGED=1
    fi
    if [ -n "$(git ls-files --others --exclude-standard "$INSTANCE_DIR/" 2>/dev/null)" ]; then
        CHANGED=1
    fi
fi

if [ "$CHANGED" -eq 0 ]; then
    echo "[Nomad-Raising] No new raising data to commit."
    exit 0
fi

# Stage instance dir + focus
git add "$INSTANCE_DIR/" SESSION_FOCUS.md 2>/dev/null || true

git commit -m "[Nomad-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Automated SAGE-Nomad raising session via OllamaIRP
Machine: Nomad (Desktop, WSL2)
Model: Gemma 3 4B (google-gemma family)
Phase: $PHASE
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

# Push
PAT=$(grep GITHUB_PAT /mnt/c/projects/ai-agents/.env 2>/dev/null | cut -d= -f2)
if [ -n "$PAT" ]; then
    git push "https://dp-web4:${PAT}@github.com/dp-web4/SAGE.git" main
    echo "[Nomad-Raising] Session $SESSION_NUM committed and pushed."
else
    echo "[Nomad-Raising] ERROR: No GITHUB_PAT found, cannot push."
fi

echo "[Nomad-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Done."

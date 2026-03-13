#!/bin/bash
# Sprout SAGE raising session + auto-commit
# Runs a raising session, snapshots state, commits results, pushes to origin.
# Schedule: every 6 hours via crontab.

set -e

SAGE_DIR="/home/sprout/ai-workspace/SAGE"
export PYTHONPATH="$SAGE_DIR"

cd "$SAGE_DIR"

echo "[Sprout-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

# Pull latest before running (avoid conflicts)
git pull --rebase origin main 2>&1 || {
    echo "[Sprout-Raising] WARNING: git pull failed, continuing with local state"
}

# Ensure daemon is running via systemd
if ! systemctl is-active --quiet sage-daemon-sprout; then
    echo "[Sprout-Raising] Starting daemon..."
    sudo systemctl start sage-daemon-sprout
    sleep 15  # Wait for model load
fi

# Run the raising session (continue from last session number)
python3 -m sage.raising.scripts.ollama_raising_session --machine sprout -c 2>&1

# Instance directory
INSTANCE_DIR="sage/instances/sprout-qwen3.5-0.8b"

# Snapshot live state into git-tracked snapshots/ directory
echo "[Sprout-Raising] Snapshotting state..."
python3 -m sage.scripts.snapshot_state --machine sprout

# Read session number and phase from live identity
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

# Regenerate session primer with updated fleet state
echo "[Sprout-Raising] Updating SESSION_PRIMER.md..."
python3 -m sage.scripts.generate_primer 2>/dev/null || true

# Check if there are new results to commit
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
    echo "[Sprout-Raising] No new raising data to commit."
    exit 0
fi

# Stage instance dir + primer
git add "$INSTANCE_DIR/" SESSION_PRIMER.md 2>/dev/null || true

git commit -m "[Sprout-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Automated sprout raising session via OllamaIRP
Machine: Sprout (Jetson Orin Nano 8GB)
Model: Qwen 3.5 0.8B (alibaba-qwen family)
Phase: $PHASE
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

# Push using PAT
PAT=$(grep GITHUB_PAT /home/sprout/ai-workspace/.env 2>/dev/null | cut -d= -f2)
if [ -n "$PAT" ]; then
    git push "https://dp-web4:${PAT}@github.com/dp-web4/SAGE.git" main
    echo "[Sprout-Raising] Session $SESSION_NUM committed and pushed."
else
    git push origin main
    echo "[Sprout-Raising] Session $SESSION_NUM committed and pushed."
fi

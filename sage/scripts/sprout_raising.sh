#!/bin/bash
# Sprout SAGE raising session + auto-commit
# Runs a raising session, snapshots state, commits results, pushes to origin.
# Schedule: every 6 hours via crontab.

set -e

SAGE_DIR="/home/sprout/ai-workspace/SAGE"
export PYTHONPATH="$SAGE_DIR"
LOG_DIR="/tmp/sprout-raising-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/raising-$(date +%Y%m%d-%H%M).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[Sprout-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

cd "$SAGE_DIR"

# --- Step 1: Pull latest code ---
echo "[Sprout-Raising] Pulling latest code..."
git pull --ff-only origin main 2>&1 || {
    echo "[Sprout-Raising] WARNING: git pull --ff-only failed, trying rebase..."
    git stash -q 2>/dev/null
    git pull --rebase origin main 2>&1 || {
        echo "[Sprout-Raising] WARNING: git pull failed, continuing with local state"
    }
    git stash pop -q 2>/dev/null || true
}

# --- Step 2: Ensure daemon is running and up to date ---
if systemctl is-active --quiet sage-daemon-sprout; then
    # Check if daemon is running stale code (started before last git commit)
    DAEMON_PID=$(systemctl show sage-daemon-sprout --property=MainPID --value 2>/dev/null || echo 0)
    if [ "$DAEMON_PID" -gt 0 ] 2>/dev/null; then
        DAEMON_START=$(stat -c %Y /proc/$DAEMON_PID 2>/dev/null || echo 0)
        LAST_COMMIT=$(git log -1 --format=%ct 2>/dev/null || echo 0)
        if [ "$LAST_COMMIT" -gt "$DAEMON_START" ] 2>/dev/null; then
            echo "[Sprout-Raising] Daemon is stale — restarting with latest code..."
            sudo systemctl restart sage-daemon-sprout
            sleep 15  # Wait for model load
        fi
    fi
else
    echo "[Sprout-Raising] Starting daemon..."
    sudo systemctl start sage-daemon-sprout
    sleep 15  # Wait for model load
fi

# Verify daemon is healthy
if ! curl -s http://localhost:8750/health >/dev/null 2>&1; then
    echo "[Sprout-Raising] WARNING: Daemon not responding, waiting 10s..."
    sleep 10
    if ! curl -s http://localhost:8750/health >/dev/null 2>&1; then
        echo "[Sprout-Raising] ERROR: Daemon still not responding, aborting."
        exit 1
    fi
fi

# --- Step 3: Run the raising session ---
echo "[Sprout-Raising] Running raising session..."
python3 -m sage.raising.scripts.ollama_raising_session --machine sprout -c 2>&1

# --- Step 4: Snapshot state ---
INSTANCE_DIR="sage/instances/sprout-qwen3.5-0.8b"

echo "[Sprout-Raising] Snapshotting state..."
python3 -m sage.scripts.snapshot_state --machine sprout 2>&1 || {
    echo "[Sprout-Raising] WARNING: snapshot_state failed, continuing"
}

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

# --- Step 5: Regenerate fleet snapshot ---
echo "[Sprout-Raising] Updating SESSION_FOCUS.md..."
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
    echo "[Sprout-Raising] No new raising data to commit."
    exit 0
fi

# Stage instance dir + focus
git add "$INSTANCE_DIR/" SESSION_FOCUS.md 2>/dev/null || true

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
    echo "[Sprout-Raising] ERROR: No GITHUB_PAT found, cannot push."
fi

echo "[Sprout-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Done."

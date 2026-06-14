#!/bin/bash
# Sprout SAGE raising session + auto-commit
# Runs a raising session, snapshots state, commits results, pushes to origin.
# Schedule: every 6 hours via crontab.

set -e

# Locate the SAGE workspace portably (the machine's user/home is not always "sprout";
# the repo clones as lowercase "sage" with a "SAGE" compat symlink on some hosts).
if [ -d "$HOME/ai-workspace/sage" ]; then
    SAGE_DIR="$HOME/ai-workspace/sage"
elif [ -d "$HOME/ai-workspace/SAGE" ]; then
    SAGE_DIR="$HOME/ai-workspace/SAGE"
elif [ -d "/home/sprout/ai-workspace/SAGE" ]; then
    SAGE_DIR="/home/sprout/ai-workspace/SAGE"
else
    echo "[Sprout-Raising] FATAL: cannot locate SAGE workspace under \$HOME/ai-workspace"
    exit 1
fi
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
    # Track whether THIS run actually stashed anything; only pop if so.
    # Unconditional `git stash pop` could pop a stale stash from a prior
    # run and silently inject merge markers into the working tree.
    STASH_BEFORE=$(git stash list | wc -l)
    git stash -q 2>/dev/null
    git pull --rebase origin main 2>&1 || {
        echo "[Sprout-Raising] WARNING: git pull failed, continuing with local state"
    }
    if [ "$(git stash list | wc -l)" -gt "$STASH_BEFORE" ]; then
        git stash pop -q
    fi
}

# --- Step 2: Ensure Rust daemon is running (OPTIONAL) ---
# sage-daemon-sprout runs the Rust binary (sage-rs/target/release/sage-daemon)
# on port 8760, providing chat history and metabolic state. Raising talks
# directly to Ollama (11434), NOT the daemon, so this is best-effort only and
# must never abort the session (the daemon may be unbuilt on a fresh host).
if systemctl list-unit-files sage-daemon-sprout.service >/dev/null 2>&1 \
   && ! systemctl is-active --quiet sage-daemon-sprout; then
    echo "[Sprout-Raising] Starting daemon (best-effort)..."
    sudo systemctl start sage-daemon-sprout 2>/dev/null || \
        echo "[Sprout-Raising] WARNING: could not start daemon; continuing (raising talks to Ollama directly)"
    sleep 8
fi
if ! curl -s http://localhost:8760/health >/dev/null 2>&1; then
    echo "[Sprout-Raising] NOTE: Rust daemon not responding; continuing (raising talks to Ollama directly)"
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

# Read session number and phase from the snapshot identity (daemon-proof — the
# live identity.json is continuously overwritten by the daemon; snapshots/ is the
# raising-authoritative copy written by the runner).
SNAP_IDENTITY="$INSTANCE_DIR/snapshots/identity.json"
SESSION_NUM=$(python3 -c "
import json
with open('$SAGE_DIR/$SNAP_IDENTITY') as f:
    print(json.load(f)['identity']['session_count'])
" 2>/dev/null || echo "?")

PHASE=$(python3 -c "
import json
with open('$SAGE_DIR/$SNAP_IDENTITY') as f:
    print(json.load(f)['development']['phase_name'])
" 2>/dev/null || echo "?")

# --- Step 5: Regenerate fleet snapshot ---
# NOTE: dream consolidation is NOT run here — the runner (ollama_raising_session)
# invokes run_dream_consolidation() internally at session close. Running it again
# would double-consolidate the same session.
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

# Push via SSH (PAT deprecated — SSH key loaded by ssh-agent)
git push origin main 2>&1 && {
    echo "[Sprout-Raising] Session $SESSION_NUM committed and pushed."
} || {
    echo "[Sprout-Raising] WARNING: git push failed, will retry next session"
}

echo "[Sprout-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Done."

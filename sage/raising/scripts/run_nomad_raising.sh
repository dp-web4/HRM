#!/bin/bash
# Nomad SAGE raising session + auto-commit
# Runs a FLUID raising session, snapshots state, commits results, pushes to origin.
# Schedule: every 6 hours via crontab (0,6,12,18).
#
# v2.0 (2026-04-19): Switched from ollama_raising_session to fluid runner.
# The old runner fed the attractor loop for 25 sessions (S96-S120).
# Fluid runner: no exemplar injection for <=4B, attractor counter-prompting,
# MRH-informed prompt composition.

set -e

SAGE_DIR="/mnt/c/projects/ai-agents/SAGE"
export PYTHONPATH="$SAGE_DIR"
LOG_DIR="/tmp/nomad-raising-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/raising-$(date +%Y%m%d-%H%M).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[Nomad-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session (fluid runner)"

cd "$SAGE_DIR"

# --- Step 1: Pull latest code ---
echo "[Nomad-Raising] Pulling latest code..."
git pull --ff-only origin main 2>&1 || {
    echo "[Nomad-Raising] WARNING: git pull --ff-only failed, trying merge..."
    git pull --no-rebase --no-edit origin main 2>&1 || {
        echo "[Nomad-Raising] WARNING: git pull failed, continuing with local state"
    }
}

# --- Step 2: Source router shadow env + ensure daemon ---
# Cut over to Rust sage-daemon (port 8760) per SAGE/sage-rs/CUTOVER.md
# Sprint 7+ (2026-06-06): same binary multi-machine; SAGE_MACHINE drives
# instance-dir + identity. Manual-start path because WSL2 has no user-systemd.
export SAGE_PORT="${SAGE_PORT:-8760}"
export SAGE_MACHINE=nomad
export SAGE_MODEL=gemma4:e2b
export SAGE_NO_BROWSER=1
if [ -f "$SAGE_DIR/sage/gateway/router-shadow.env" ]; then
    set -a; source "$SAGE_DIR/sage/gateway/router-shadow.env"; set +a
    echo "[Nomad-Raising] Router shadow: SAGE_ROUTER_SHADOW=$SAGE_ROUTER_SHADOW"
fi
source "$SAGE_DIR/sage/scripts/ensure_daemon_rs.sh"
echo "[Nomad-Raising] Daemon: version=$SAGE_DAEMON_VERSION running=$SAGE_DAEMON_RUNNING updated=$SAGE_DAEMON_UPDATED"

# --- Step 3: Run the FLUID raising session ---
# Canonical Nomad raising model = gemma4:e2b (per private-context/machines/fleet/nomad.json
# as of 2026-06-03; sweep default also matches raising per fleetwide policy).
# Prior arc raised gemma3:4b through session 172 (sessions/ archived at
# sage/instances/nomad-gemma3-4b.archive-20260603). Switched 2026-06-03 because
# CBP simultaneously moved to gemma3:4b (its RTX 2060 SUPER can't fit e2b — Windows
# compositor holds ~2.2GB), and Nomad's RTX 4060 Laptop fits e2b GPU-only (verified
# at 16k context, 7553/8188 MiB, 100% GPU). Migration preserves fleet model diversity.
echo "[Nomad-Raising] Running fluid raising session..."
python3 "$SAGE_DIR/sage/raising/scripts/run_session_identity_anchored_fluid.py" \
    --machine nomad \
    --model gemma4:e2b \
    2>&1

# --- Step 4: Snapshot state ---
INSTANCE_DIR="sage/instances/nomad-gemma4-e2b"

echo "[Nomad-Raising] Snapshotting state..."
python3 -m sage.scripts.snapshot_state --machine nomad --model gemma4:e2b 2>&1 || {
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
git add "$INSTANCE_DIR/" SESSION_FOCUS.md SESSION_MAP.yaml 2>/dev/null || true

git commit -m "[Nomad-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

# Push via SSH (PAT is deprecated)
git push origin main 2>&1 || {
    echo "[Nomad-Raising] WARNING: push failed, will retry next session"
}

echo "[Nomad-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Done."

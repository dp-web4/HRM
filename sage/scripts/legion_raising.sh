#!/bin/bash
# Legion SAGE raising session + auto-commit
# Runs a raising session, commits results, pushes to origin.
# Designed to run via systemd timer every 6 hours.

set -e

SAGE_DIR="/home/dp/ai-workspace/SAGE"
PYTHONPATH="$SAGE_DIR"
export PYTHONPATH
PYTHON="/home/dp/miniforge3/bin/python3"

cd "$SAGE_DIR"

# Account routing: synth token for raising sessions (synthesis pool machine)
ENV_FILE="/home/dp/ai-workspace/.env"
if [ -f "$ENV_FILE" ]; then
    CLAUDE_SYNTH_TOKEN=$(grep '^CLAUDE_SYNTH_TOKEN=' "$ENV_FILE" | cut -d= -f2-)
fi
if [ -n "$CLAUDE_SYNTH_TOKEN" ] && [[ "$CLAUDE_SYNTH_TOKEN" != PLACEHOLDER* ]]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_SYNTH_TOKEN"
fi

echo "[Legion-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting raising session"

# Ensure daemon is running via systemd (not the manual ensure_daemon.sh)
if ! systemctl --user is-active sage-daemon.service >/dev/null 2>&1; then
    echo "[Legion-Raising] Daemon not running, starting via systemctl..."
    systemctl --user start sage-daemon.service
    sleep 5
fi

# Verify daemon health
# Port + status string both follow the 2026-06-06 Python->Rust cutover:
# sage-rs listens on 8760 (was 8750) and reports status="ok" (was "alive").
# This script kept both old values until 2026-07-30, so the check could never
# pass — it restarted a healthy daemon every fire and then logged the restart
# as failed. See cbp_raising.sh / ensure_daemon_rs.sh for the fleet convention.
SAGE_PORT="${SAGE_PORT:-8760}"
HEALTH=$(curl -s --max-time 5 "http://localhost:${SAGE_PORT}/health" 2>/dev/null || echo "")
if echo "$HEALTH" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d.get('status') in ('ok','alive')" 2>/dev/null; then
    echo "[Legion-Raising] Daemon healthy"
else
    echo "[Legion-Raising] WARNING: Daemon health check failed, attempting restart..."
    systemctl --user restart sage-daemon.service
    sleep 10
fi

# Pull latest code and check if daemon needs restart
BEFORE_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
# `--autostash`'s apply failure does NOT set the exit code. On an autostash-apply
# conflict `git pull --rebase --autostash` exits **0**, writes conflict markers
# into the working tree, and puts "Applying autostash resulted in conflicts." on
# **stderr** — so the old form here (`2>/dev/null || echo WARN`) was silent in
# exactly the case that matters: the WARN keys on rc, and rc is 0, while
# `2>/dev/null` discarded the only message. Verified in a scratch repo
# 2026-08-01 with a real same-line conflict: rc=0, tree `UU`, markers on disk.
#
# This is the mirror image of the fleet-sync `stash`/`pop || true` defect found
# 2026-07-31 — that one loses data and leaves an honest-looking tree; this one
# keeps the data and corrupts the file. Found by HUB, reproduced here unchanged.
# So: check the TREE, not the exit code.
if ! pull_out=$(git pull --rebase --autostash --quiet 2>&1); then
    echo "[Legion-Raising] WARN: git pull failed, continuing with current code: $(echo "$pull_out" | tail -2 | tr '\n' ' ')"
fi

conflicted=$(git diff --name-only --diff-filter=U 2>/dev/null)
if [ -n "$conflicted" ]; then
    echo "[Legion-Raising] ERROR: autostash apply left conflict markers (pull rc was 0) in:"
    echo "$conflicted" | sed 's/^/    /'
    echo "[Legion-Raising] Local changes are preserved in stash@{0} ('autostash') and are NOT lost."
    echo "[Legion-Raising] Restoring those paths to HEAD so this session cannot commit markers."
    echo "[Legion-Raising] Recover with: git -C $(pwd) checkout 'stash@{0}' -- <path>"
    echo "$conflicted" | while read -r cf; do
        [ -n "$cf" ] || continue
        git checkout HEAD -- "$cf" 2>/dev/null || true
    done
fi
AFTER_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

if [ "$BEFORE_HASH" != "$AFTER_HASH" ]; then
    echo "[Legion-Raising] Code updated ($BEFORE_HASH → $AFTER_HASH), restarting daemon..."
    systemctl --user restart sage-daemon.service
    sleep 5
    HEALTH=$(curl -s --max-time 5 "http://localhost:${SAGE_PORT}/health" 2>/dev/null || echo "")
    if echo "$HEALTH" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d.get('status') in ('ok','alive')" 2>/dev/null; then
        echo "[Legion-Raising] Daemon restarted with new code, healthy"
    else
        echo "[Legion-Raising] WARNING: Daemon restart failed, continuing anyway"
    fi
else
    echo "[Legion-Raising] Code unchanged ($BEFORE_HASH)"
fi

# Run the raising session (continue from last session number)
# Model: Gemma 4 E4B — same model for raising + gameplay (all-track convergence)
# Runner: fluid (MRH blocks, attractor counter-prompting, machine-parameterized)
# Switched from ollama_raising_session 2026-04-20 for fleet consistency
$PYTHON -m sage.raising.scripts.run_session_identity_anchored_fluid --machine legion --model gemma4:e4b 2>&1

# Instance directory
INSTANCE_DIR="sage/instances/legion-gemma4-e4b"

# Snapshot live state files into git-tracked snapshots/ directory
echo "[Legion-Raising] Snapshotting state..."
$PYTHON -m sage.scripts.snapshot_state --machine legion

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
    echo "[Legion-Raising] No new raising data to commit."
    exit 0
fi

# Read session number from identity state
IDENTITY_FILE="$INSTANCE_DIR/identity.json"

SESSION_NUM=$($PYTHON -c "
import json
with open('$SAGE_DIR/$IDENTITY_FILE') as f:
    print(json.load(f)['identity']['session_count'])
" 2>/dev/null || echo "?")

PHASE=$($PYTHON -c "
import json
with open('$SAGE_DIR/$IDENTITY_FILE') as f:
    print(json.load(f)['development']['phase_name'])
" 2>/dev/null || echo "?")

# --- Dream consolidation (Claude reviews the session) ---
echo "[Legion-Raising] Running dream consolidation..."
$PYTHON -m sage.raising.scripts.dream_consolidation \
    --instance "$INSTANCE_DIR" \
    --session "$SESSION_NUM" 2>&1 || {
    echo "[Legion-Raising] Dream consolidation skipped (claude --print not available or timed out)"
}

# Stage instance dir (sessions + snapshots, gitignored files excluded automatically)
git add "$INSTANCE_DIR/" 2>/dev/null || true

git commit -m "[Legion-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Automated SAGE-Legion raising session via OllamaIRP
Machine: Legion (Legion Pro 7, RTX 4090, Linux)
Model: Gemma 3 12B (google-gemma family)
Phase: $PHASE
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

# Push (with retry for race conditions from other machines)
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    # --autostash: sage-daemon concurrently writes other instance dirs (e.g.
    # legion-gemma3-12b). We only `git add` our own $INSTANCE_DIR, so those stay
    # unstaged and a bare `pull --rebase` aborts with "cannot pull with rebase:
    # You have unstaged changes" (exit 128) before any rebase starts — making the
    # `rebase --abort` below a no-op and all retries fail identically.
    git pull --rebase --autostash 2>&1 || {
        echo "[Legion-Raising] Rebase conflict on attempt $i, aborting rebase and retrying..."
        git rebase --abort 2>/dev/null
        sleep $((i * 5))
        continue
    }
    if git push origin main 2>&1; then
        echo "[Legion-Raising] Session $SESSION_NUM committed and pushed."
        break
    fi
    echo "[Legion-Raising] Push failed attempt $i/$MAX_RETRIES, retrying..."
    sleep $((i * 5))
done

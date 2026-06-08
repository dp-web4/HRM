#!/bin/bash
# McNugget Raising — Fluid variant with MRH-informed prompt
#
# Applies Sprout's crystallization fix:
# - No verbatim exemplar injection (prevents self-quotation feedback loop)
# - MRH-structured system prompt (typed blocks, not ad-hoc string concat)
# - Concise turns, single-purpose primer
#
# Uses whatever model the resident daemon runs (auto-detected from
# /health). Designed to run via launchd every 6 hours.

set -e

SAGE_DIR="/Users/dennispalatov/repos/SAGE"
PYTHONPATH="$SAGE_DIR"
export PYTHONPATH

# Fix OpenMP duplicate library crash on macOS
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Router shadow: source-stamp records from raising sessions
export SAGE_SESSION_SOURCE=raising

cd "$SAGE_DIR"

echo "[McNugget-Raising] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting fluid raising session"

# Pull latest
# Track whether THIS run actually stashed anything; only pop if so.
# Unconditional `git stash pop` could pop a stale stash from a prior
# run and silently inject merge markers into the working tree.
STASH_BEFORE=$(git stash list | wc -l)
git stash 2>/dev/null || true
git pull --rebase origin main 2>&1 || {
    echo "[McNugget-Raising] WARNING: git pull failed, continuing with local state"
}
if [ "$(git stash list | wc -l)" -gt "$STASH_BEFORE" ]; then
    git stash pop
fi

# Ensure daemon is running
source "$SAGE_DIR/sage/scripts/ensure_daemon.sh"

# Run raising session via unified launcher + fluid runner
# Uses the identity-anchored fluid variant with MRH block-based prompt
# and Thor S86 anti-crystallization mitigations.
/opt/homebrew/bin/python3 -m sage.session --raising --fluid \
    --machine mcnugget \
    2>&1

# Derive the instance dir from the daemon's actual model rather than
# hardcoding. The session above wrote to whichever instance `sage.session
# --raising --fluid --machine mcnugget` picked (driven by the daemon's
# active model). Hardcoded gemma4-e4b would silently miss when the daemon
# runs gemma3:12b (Sprint 7 default) — dream consolidation would skip
# with "Session file not found". Auto-detect from /health.
DAEMON_MODEL=$(curl -s --max-time 3 "http://localhost:${SAGE_PORT:-8760}/health" 2>/dev/null \
    | /opt/homebrew/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('model','gemma3:12b'))" 2>/dev/null \
    || echo "gemma3:12b")
INSTANCE_SLUG="mcnugget-${DAEMON_MODEL//:/-}"
INSTANCE_DIR="sage/instances/$INSTANCE_SLUG"
echo "[McNugget-Raising] Active instance: $INSTANCE_SLUG (daemon model: $DAEMON_MODEL)"

# Snapshot state
echo "[McNugget-Raising] Snapshotting state..."
/opt/homebrew/bin/python3 -m sage.scripts.snapshot_state \
    --machine mcnugget \
    --instance "$INSTANCE_SLUG" 2>/dev/null || true

# Read session info
SESSION_NUM=$(/opt/homebrew/bin/python3 -c "
import json
with open('$SAGE_DIR/$INSTANCE_DIR/identity.json') as f:
    print(json.load(f)['identity']['session_count'])
" 2>/dev/null || echo "?")

PHASE=$(/opt/homebrew/bin/python3 -c "
import json
with open('$SAGE_DIR/$INSTANCE_DIR/identity.json') as f:
    print(json.load(f)['development']['phase_name'])
" 2>/dev/null || echo "?")

# Dream consolidation
echo "[McNugget-Raising] Dream consolidation..."
/opt/homebrew/bin/python3 -m sage.raising.scripts.dream_consolidation \
    --instance "$INSTANCE_DIR" \
    --session "$SESSION_NUM" 2>&1 || {
    echo "[McNugget-Raising] Dream consolidation skipped"
}

# Commit if changed
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
    echo "[McNugget-Raising] No new data to commit."
    exit 0
fi

git add "$INSTANCE_DIR/" 2>/dev/null || true

git commit -m "[McNugget-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Fluid raising session via $DAEMON_MODEL
Machine: McNugget (Mac Mini M4)
Model: $DAEMON_MODEL
Phase: $PHASE
Runner: sage.session --raising --fluid (auto-detected instance: $INSTANCE_SLUG)
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

git pull --rebase origin main 2>/dev/null || true
git push origin main 2>&1 || {
    echo "[McNugget-Raising] WARNING: push failed, will retry next session"
}
echo "[McNugget-Raising] Session $SESSION_NUM committed and pushed."

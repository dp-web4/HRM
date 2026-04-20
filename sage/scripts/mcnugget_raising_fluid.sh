#!/bin/bash
# McNugget Raising — Fluid variant with MRH-informed prompt
#
# Applies Sprout's crystallization fix:
# - No verbatim exemplar injection (prevents self-quotation feedback loop)
# - MRH-structured system prompt (typed blocks, not ad-hoc string concat)
# - Concise turns, single-purpose primer
#
# Uses gemma4:e4b on the resident daemon via DaemonIRP.
# Designed to run via launchd every 6 hours.

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
git stash 2>/dev/null || true
git pull --rebase origin main 2>&1 || {
    echo "[McNugget-Raising] WARNING: git pull failed, continuing with local state"
}
git stash pop 2>/dev/null || true

# Ensure daemon is running
source "$SAGE_DIR/sage/scripts/ensure_daemon.sh"

# Run raising session with the identity-anchored fluid runner
# McNugget uses gemma4:e4b — larger than Sprout's 0.8B,
# so exemplar gating should allow them through.
# The fluid scaffold's 4-gram diversity filter prevents crystallization
# while still allowing identity anchoring.
/opt/homebrew/bin/python3 -m sage.raising.scripts.ollama_raising_session \
    --machine mcnugget \
    --model gemma4:e4b \
    -c 2>&1

INSTANCE_DIR="sage/instances/mcnugget-gemma4-e4b"

# Snapshot state
echo "[McNugget-Raising] Snapshotting state..."
/opt/homebrew/bin/python3 -m sage.scripts.snapshot_state \
    --machine mcnugget \
    --instance mcnugget-gemma4-e4b 2>/dev/null || true

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

Fluid raising session via Gemma 4 E4B
Machine: McNugget (Mac Mini M4)
Model: Gemma 4 E4B (google-gemma4 family)
Phase: $PHASE
Runner: ollama_raising_session (fluid scaffold pending)
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

git pull --rebase origin main 2>/dev/null || true
git push origin main 2>&1 || {
    echo "[McNugget-Raising] WARNING: push failed, will retry next session"
}
echo "[McNugget-Raising] Session $SESSION_NUM committed and pushed."

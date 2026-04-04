#!/bin/bash
# Thor Gemma 4 E4B — Federation-raised gameplayer
# Raising sessions focused on puzzle reasoning, spatial understanding,
# and game-playing capability. Fed by fleet KB accumulated from ARC-AGI-3.
#
# Designed to run via cron/systemd every 6 hours (paused during this pivot).

set -e

SAGE_DIR="/home/dp/ai-workspace/SAGE"
PYTHONPATH="$SAGE_DIR"
export PYTHONPATH

cd "$SAGE_DIR"

echo "[Thor-G4] $(date -u +'%Y-%m-%d %H:%M UTC') — Starting gameplayer raising session"

# Pull latest
git pull --rebase origin main 2>&1 || {
    echo "[Thor-G4] WARNING: git pull failed, continuing with local state"
}

# Run raising session with gemma4:e4b
python3 -m sage.raising.scripts.ollama_raising_session \
    --machine thor \
    --model gemma4:e4b \
    -c 2>&1

# Instance directory
INSTANCE_DIR="sage/instances/thor-gemma4-e4b"

# Snapshot state
echo "[Thor-G4] Snapshotting state..."
python3 -m sage.scripts.snapshot_state --machine thor --instance thor-gemma4-e4b 2>/dev/null || true

# Read session info
SESSION_NUM=$(python3 -c "
import json
with open('$SAGE_DIR/$INSTANCE_DIR/identity.json') as f:
    print(json.load(f)['identity']['session_count'])
" 2>/dev/null || echo "?")

PHASE=$(python3 -c "
import json
with open('$SAGE_DIR/$INSTANCE_DIR/identity.json') as f:
    print(json.load(f)['development']['phase_name'])
" 2>/dev/null || echo "?")

# Dream consolidation
echo "[Thor-G4] Running dream consolidation..."
python3 -m sage.raising.scripts.dream_consolidation \
    --instance "$INSTANCE_DIR" \
    --session "$SESSION_NUM" 2>&1 || {
    echo "[Thor-G4] Dream consolidation skipped"
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
    echo "[Thor-G4] No new data to commit."
    exit 0
fi

git add "$INSTANCE_DIR/" 2>/dev/null || true

git commit -m "[Thor-G4-Raising] Session $SESSION_NUM ($PHASE) — $(date -u +'%Y-%m-%d %H:%M UTC')

Federation-raised gameplayer session via Gemma 4 E4B
Machine: Thor (Jetson AGX Thor)
Model: Gemma 4 E4B (google-gemma4 family)
Phase: $PHASE
Role: gameplayer (ARC-AGI-3 competition)
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

git push origin main
echo "[Thor-G4] Session $SESSION_NUM committed and pushed."

#!/bin/bash
# McNugget autonomous supervisor — runs every 4 hours via launchd.
#
# Pulls repos, checks/launches sweeps, reads fleet forums, does work,
# documents and pushes. Designed to keep McNugget productive without
# manual intervention.
#
# Model: phi4-fa (phi4:14b, num_ctx 4096) — switched from gemma3-fa 2026-05-15
# Stack: v14 canonical (6 flags). v17.x OFF (confirmed harmful on mid-models).
#        v18 reset OFF by default (4.3× wall time cost for +1 level).

set -u

SAGE_DIR="/Users/dennispalatov/repos/SAGE"
DEV_SAGE="/Users/dennispalatov/repos/dev-SAGE"
SHARED="/Users/dennispalatov/repos/shared-context"
PRIVATE="/Users/dennispalatov/repos/private-context"
MEMORY="/Users/dennispalatov/repos/memory"
HESTIA="/Users/dennispalatov/repos/hestia"
SWEEP_DIR="$HOME/mcnugget-sweep"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH="$DEV_SAGE"

# Model config
MODEL="phi4-fa"

# v14 canonical flags
export SAGE_COLD_START=1
export SAGE_OBJECT_SPACE=1
export SAGE_RULE_HARVESTER=1
export SAGE_COGNITIVE_EXTRACTION=1
export SAGE_GAMEPLAY_CONVERSATIONS=1
export SAGE_COGNITIVE_ROUTER=1

# v17.x combo: OFF (confirmed harmful on mid-models per fleet bisection)
# v18 reset: OFF by default (use dedicated v18 sweep scripts when needed)

export SAGE_MACHINE=mcnugget
export SAGE_LLM_BACKEND=ollama
export SAGE_OLLAMA_MODEL="$MODEL"

TIMESTAMP=$(date -u +'%Y-%m-%d %H:%M UTC')
echo "[McNugget-Supervisor] $TIMESTAMP — Starting cycle"
echo "[McNugget-Supervisor] Model: $MODEL, Stack: v14 canonical"

# === 1. PULL ALL REPOS ===
for repo in "$DEV_SAGE" "$SAGE_DIR" "$SHARED" "$PRIVATE" "$MEMORY" "$HESTIA"; do
    if [ -d "$repo" ]; then
        cd "$repo"
        git fetch origin 2>/dev/null
        git reset --hard origin/main 2>/dev/null
    fi
done
echo "[McNugget-Supervisor] Repos synced"

# === 2. CHECK SWEEPS ===
SWEEP_RUNNING=$(ps aux | grep sweep_all_25 | grep -v grep | wc -l | tr -d ' ')
SWEEP_LOG="$SWEEP_DIR/sweep.log"

if [ "$SWEEP_RUNNING" -gt 0 ]; then
    echo "[McNugget-Supervisor] Sweep in progress — not interfering"
    if [ -f "$SWEEP_LOG" ]; then
        DONE=$(grep -c "^L=" "$SWEEP_LOG" 2>/dev/null || echo 0)
        STARS=$(grep -c "★" "$SWEEP_LOG" 2>/dev/null || echo 0)
        echo "[McNugget-Supervisor] Progress: $DONE/25 games, $STARS level advances"
    fi
else
    # Check if dev-SAGE advanced since last sweep
    LAST_COMMIT_FILE="$SWEEP_DIR/.last_sweep_commit"
    CURRENT_COMMIT=$(cd "$DEV_SAGE" && git rev-parse --short HEAD)
    LAST_SWEEP_COMMIT=""
    if [ -f "$LAST_COMMIT_FILE" ]; then
        LAST_SWEEP_COMMIT=$(cat "$LAST_COMMIT_FILE" 2>/dev/null)
    fi

    if [ "$LAST_SWEEP_COMMIT" != "$CURRENT_COMMIT" ]; then
        echo "[McNugget-Supervisor] dev-SAGE advanced: ${LAST_SWEEP_COMMIT:-none} → $CURRENT_COMMIT"
        # Check ollama
        if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
            echo "[McNugget-Supervisor] Launching new sweep with $MODEL"
            mkdir -p "$SWEEP_DIR"/{episodes,tier1_diag,diagnostic_games}

            export SAGE_EPISODE_STORE_DIR="$SWEEP_DIR/episodes"
            export SAGE_TIER1_DIAG_DIR="$SWEEP_DIR/tier1_diag"
            export SAGE_GAME_DIAG_DIR="$SWEEP_DIR/diagnostic_games"

            cd "$SAGE_DIR"
            nohup /opt/homebrew/bin/python3 \
                "$DEV_SAGE/arc-agi-3/experiments/sweep_all_25.py" \
                --model "$MODEL" --max-steps 600 --max-revisions 100 \
                > "$SWEEP_LOG" 2>&1 &
            echo "[McNugget-Supervisor] Sweep launched PID: $!"
            echo "$CURRENT_COMMIT" > "$LAST_COMMIT_FILE"
        else
            echo "[McNugget-Supervisor] Ollama not running — skipping sweep"
        fi
    else
        echo "[McNugget-Supervisor] No new code to sweep (last: $LAST_SWEEP_COMMIT)"
    fi
fi

# === 3. READ FORUMS ===
cd "$SHARED"
RECENT_FORUM=$(find forum/ -name "*.md" -mtime -1 -type f 2>/dev/null | wc -l | tr -d ' ')
echo "[McNugget-Supervisor] $RECENT_FORUM forum posts in last 24h"

# === 4. PUSH (if anything changed) ===
for repo in "$SHARED" "$DEV_SAGE" "$SAGE_DIR" "$PRIVATE"; do
    cd "$repo"
    if ! git diff --quiet 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
        git add -A 2>/dev/null
        git commit -m "[McNugget-Supervisor] Autonomous cycle — $TIMESTAMP" 2>/dev/null
        git push origin main 2>&1 || true
    fi
done

echo "[McNugget-Supervisor] Cycle complete — $(date -u +'%Y-%m-%d %H:%M UTC')"

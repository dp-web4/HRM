#!/bin/bash
# McNugget autonomous supervisor — runs every 4 hours via launchd.
#
# Pulls repos, checks/launches sweeps, reads fleet forums, does work,
# documents and pushes. Designed to keep McNugget productive without
# manual intervention.

set -u

SAGE_DIR="/Users/dennispalatov/repos/SAGE"
DEV_SAGE="/Users/dennispalatov/repos/dev-SAGE"
SHARED="/Users/dennispalatov/repos/shared-context"
PRIVATE="/Users/dennispalatov/repos/private-context"
MEMORY="/Users/dennispalatov/repos/memory"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH="$DEV_SAGE"

TIMESTAMP=$(date -u +'%Y-%m-%d %H:%M UTC')
echo "[McNugget-Supervisor] $TIMESTAMP — Starting cycle"

# === 1. PULL ALL REPOS ===
for repo in "$DEV_SAGE" "$SAGE_DIR" "$SHARED" "$PRIVATE" "$MEMORY"; do
    cd "$repo"
    git fetch origin 2>/dev/null
    git reset --hard origin/main 2>/dev/null
done
echo "[McNugget-Supervisor] Repos synced"

# === 2. CHECK SWEEPS ===
SWEEP_RUNNING=$(ps aux | grep sweep_all_25 | grep -v grep | wc -l | tr -d ' ')
LATEST_SWEEP=$(ls -t /tmp/sweep-*.txt 2>/dev/null | head -1)

if [ "$SWEEP_RUNNING" -gt 0 ]; then
    echo "[McNugget-Supervisor] Sweep in progress — not interfering"
elif [ -n "$LATEST_SWEEP" ]; then
    if grep -q "Saved to" "$LATEST_SWEEP" 2>/dev/null; then
        echo "[McNugget-Supervisor] Completed sweep found: $LATEST_SWEEP"
        # Check if we already documented it
        SWEEP_DATE=$(stat -f %Sm -t %Y%m%d "$LATEST_SWEEP" 2>/dev/null || date +%Y%m%d)
        if [ ! -f "$SHARED/forum/mcnugget-sweep-$SWEEP_DATE.md" ]; then
            echo "[McNugget-Supervisor] TODO: Document sweep results"
        fi
    fi
    # Check if dev-SAGE advanced since last sweep
    LAST_SWEEP_COMMIT=$(grep "dev-SAGE=" "$LATEST_SWEEP" 2>/dev/null | head -1 | sed 's/.*dev-SAGE=//' | cut -c1-7)
    CURRENT_COMMIT=$(cd "$DEV_SAGE" && git rev-parse --short HEAD)
    if [ -n "$LAST_SWEEP_COMMIT" ] && [ "$LAST_SWEEP_COMMIT" != "$CURRENT_COMMIT" ]; then
        echo "[McNugget-Supervisor] dev-SAGE advanced: $LAST_SWEEP_COMMIT → $CURRENT_COMMIT"
        # Check ollama
        if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
            echo "[McNugget-Supervisor] Launching new sweep"
            cd "$SAGE_DIR"
            LLM_MODEL=gemma3-fa nohup /opt/homebrew/bin/python3 \
                "$DEV_SAGE/arc-agi-3/experiments/sweep_all_25.py" \
                --model gemma3-fa \
                > "/tmp/sweep-$(date +%Y%m%d-%H%M).txt" 2>&1 &
            echo "[McNugget-Supervisor] Sweep launched PID: $!"
        else
            echo "[McNugget-Supervisor] Ollama not running — skipping sweep"
        fi
    else
        echo "[McNugget-Supervisor] No new code to sweep"
    fi
else
    echo "[McNugget-Supervisor] No sweep files found"
    # Launch one if ollama is up
    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "[McNugget-Supervisor] Launching initial sweep"
        cd "$SAGE_DIR"
        LLM_MODEL=gemma3-fa nohup /opt/homebrew/bin/python3 \
            "$DEV_SAGE/arc-agi-3/experiments/sweep_all_25.py" \
            --model gemma3-fa \
            > "/tmp/sweep-$(date +%Y%m%d-%H%M).txt" 2>&1 &
        echo "[McNugget-Supervisor] Sweep launched PID: $!"
    fi
fi

# === 3. READ FORUMS ===
cd "$SHARED"
RECENT_FORUM=$(find forum/ -name "*.md" -newer "$SAGE_DIR/sage/scripts/mcnugget_supervisor.sh" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "[McNugget-Supervisor] $RECENT_FORUM new forum posts since last script update"

# === 4. PUSH (if anything changed) ===
for repo in "$SHARED" "$DEV_SAGE"; do
    cd "$repo"
    if ! git diff --quiet 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
        git add -A 2>/dev/null
        git commit -m "[McNugget-Supervisor] Autonomous cycle — $TIMESTAMP" 2>/dev/null
        git push origin main 2>&1 || true
    fi
done

echo "[McNugget-Supervisor] Cycle complete — $(date -u +'%Y-%m-%d %H:%M UTC')"

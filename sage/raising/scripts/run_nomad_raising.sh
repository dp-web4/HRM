#!/bin/bash
# Nomad raising session runner — called by cron every 6 hours
# Logs to /tmp/nomad-raising-YYYYMMDD-HHMM.log

set -e

SAGE_DIR="/mnt/c/projects/ai-agents/SAGE"
LOG_DIR="/tmp/nomad-raising-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/raising-$(date +%Y%m%d-%H%M).log"

echo "=== Nomad raising session $(date) ===" >> "$LOG_FILE"

cd "$SAGE_DIR"
PYTHONPATH="$SAGE_DIR" python3 -m sage.raising.scripts.ollama_raising_session \
    --machine nomad \
    --model gemma3:4b \
    --turns 6 \
    -c >> "$LOG_FILE" 2>&1

echo "=== Done $(date) ===" >> "$LOG_FILE"

# Push results to git
cd "$SAGE_DIR"
git add sage/instances/nomad-gemma3-4b/ 2>/dev/null || true
git diff --cached --quiet || git commit -m "[Nomad] Raising session $(date +%Y-%m-%d-%H%M) (cron)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" 2>/dev/null || true
grep GITHUB_PAT /mnt/c/projects/ai-agents/.env | cut -d= -f2 | xargs -I {} git push https://dp-web4:{}@github.com/dp-web4/SAGE.git 2>/dev/null || true

#!/bin/bash
# McNugget cross-family probe + auto-commit
# Runs the probe, commits results, pushes to origin.
# Designed to run via launchd every 6 hours.

set -e

HRM_DIR="/Users/dennispalatov/repos/HRM"
PYTHONPATH="$HRM_DIR"
export PYTHONPATH

cd "$HRM_DIR"

# Pull latest first (avoid conflicts)
git pull --rebase --quiet 2>/dev/null || true

# Run the probe
/opt/homebrew/bin/python3 -m sage.experiments.cross_family_probe 2>&1

# Check if there are new results to commit
if git diff --quiet sage/experiments/cross_family_logs/ 2>/dev/null && \
   [ -z "$(git ls-files --others --exclude-standard sage/experiments/cross_family_logs/)" ]; then
    echo "[McNugget] No new probe data to commit."
    exit 0
fi

# Stage and commit
git add sage/experiments/cross_family_logs/
git commit -m "[McNugget] Cross-family probe run $(date -u +'%Y-%m-%d %H:%M UTC')

Automated cross-family cognition probe: Gemma 3 12B + Mistral 7B
Machine: McNugget (Mac Mini M4)
AI-Instance: OllamaIRP (automated)
Human-Supervised: no"

# Push
git push origin main
echo "[McNugget] Probe results committed and pushed."

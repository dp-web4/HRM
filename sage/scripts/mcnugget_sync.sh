#!/bin/bash
# McNugget repo sync — pulls all dp-web4 repos and logs changes.
# Designed to run via launchd on schedule.

REPOS_DIR="/Users/dennispalatov/repos"
LOG_DIR="$REPOS_DIR/HRM/sage/logs/mcnugget"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE="$LOG_DIR/sync.log"

mkdir -p "$LOG_DIR"

echo "[$TIMESTAMP] === McNugget sync ===" >> "$LOG_FILE"

for repo_dir in "$REPOS_DIR"/*/; do
    if [ -d "$repo_dir/.git" ]; then
        repo_name=$(basename "$repo_dir")
        cd "$repo_dir"

        # Get current HEAD before pull
        before=$(git rev-parse HEAD 2>/dev/null)

        # Pull (quiet, rebase to keep history clean)
        git pull --rebase --quiet 2>/dev/null

        # Get HEAD after pull
        after=$(git rev-parse HEAD 2>/dev/null)

        if [ "$before" != "$after" ]; then
            # Count new commits
            count=$(git log --oneline "$before".."$after" 2>/dev/null | wc -l | tr -d ' ')
            latest=$(git log --oneline -1 2>/dev/null)
            echo "  $repo_name: $count new commits — $latest" >> "$LOG_FILE"
        fi
    fi
done

echo "[$TIMESTAMP] === sync complete ===" >> "$LOG_FILE"

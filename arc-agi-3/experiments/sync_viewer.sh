#!/bin/bash
# Keep game_viewer.py in sync between SAGE and ARC-SAGE.
# ARC-SAGE is the competition repo; SAGE is the main research repo.
# Whichever has newer game_viewer.py wins; the other gets updated.
#
# Usage:
#   ./sync_viewer.sh           # sync both directions, report
#   ./sync_viewer.sh --commit  # also stage the synced file in each repo
#
# Called from either repo's experiments/ dir.

set -e

SAGE="/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments/game_viewer.py"
ARC_SAGE="/mnt/c/exe/projects/ai-agents/ARC-SAGE/arc-agi-3/experiments/game_viewer.py"

if [ ! -f "$SAGE" ] || [ ! -f "$ARC_SAGE" ]; then
    echo "Error: one or both paths missing"
    echo "  SAGE:     $SAGE"
    echo "  ARC-SAGE: $ARC_SAGE"
    exit 1
fi

sage_mtime=$(stat -c %Y "$SAGE")
arc_mtime=$(stat -c %Y "$ARC_SAGE")

if cmp -s "$SAGE" "$ARC_SAGE"; then
    echo "game_viewer.py is in sync (identical content)"
    exit 0
fi

if [ "$sage_mtime" -gt "$arc_mtime" ]; then
    echo "SAGE → ARC-SAGE  (SAGE is newer)"
    cp "$SAGE" "$ARC_SAGE"
    newer_repo="ARC-SAGE"
elif [ "$arc_mtime" -gt "$sage_mtime" ]; then
    echo "ARC-SAGE → SAGE  (ARC-SAGE is newer)"
    cp "$ARC_SAGE" "$SAGE"
    newer_repo="SAGE"
else
    echo "Same mtime but different content — pick explicitly"
    diff -u "$SAGE" "$ARC_SAGE" | head -30
    exit 2
fi

if [ "$1" = "--commit" ]; then
    if [ "$newer_repo" = "ARC-SAGE" ]; then
        cd /mnt/c/exe/projects/ai-agents/ARC-SAGE
    else
        cd /mnt/c/exe/projects/ai-agents/SAGE
    fi
    git add arc-agi-3/experiments/game_viewer.py
    echo "Staged in $newer_repo. Commit with: git commit -m 'viewer: sync from other repo'"
fi

echo "Done."

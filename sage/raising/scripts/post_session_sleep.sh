#!/bin/bash
#
# Post-Session Sleep Training Script
#
# Call this after each raising session to potentially trigger sleep training.
# The scheduler will decide if conditions are met (sufficient new experiences,
# enough time elapsed, etc.)
#
# Usage:
#   ./post_session_sleep.sh [--force]
#
# Options:
#   --force    Force sleep cycle even if checks fail (for manual testing)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")/training"

cd "$TRAINING_DIR"

echo "=================================================="
echo "POST-SESSION SLEEP TRAINING CHECK"
echo "=================================================="
echo ""
echo "Time: $(date)"
echo ""

# Check if force flag provided
if [ "$1" = "--force" ]; then
    echo "FORCING sleep cycle (bypassing checks)..."
    python3 sleep_scheduler.py --force
else
    # Normal operation - let scheduler decide
    echo "Checking if sleep cycle should run..."
    echo ""

    # First, show status
    python3 sleep_scheduler.py --check

    echo ""
    echo "Running sleep scheduler..."
    echo ""

    # Run scheduler (will skip if conditions not met)
    python3 sleep_scheduler.py
fi

echo ""
echo "=================================================="
echo "POST-SESSION SLEEP CHECK COMPLETE"
echo "=================================================="

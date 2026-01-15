#!/bin/bash
# Session 197 Phase 2: Deploy Federation Participant
# Usage: ./session197_phase2_deploy_participant.sh [coordinator_host] [duration_seconds]
#
# This script starts the consciousness-aware federation participant
# connecting to the coordinator on Thor.

set -e

COORDINATOR_HOST=${1:-10.0.0.99}
DURATION=${2:-60}
LOG_FILE="/home/dp/ai-workspace/HRM/sage/experiments/session197_phase2_participant.log"
SCRIPT_DIR="/home/dp/ai-workspace/HRM/sage/experiments"

echo "================================================================"
echo "Session 197 Phase 2: Federation Participant Deployment"
echo "================================================================"
echo "Coordinator: ${COORDINATOR_HOST}:8000"
echo "Duration: ${DURATION} seconds"
echo "Log: ${LOG_FILE}"
echo ""

# Check connectivity to coordinator
echo "Testing connectivity to coordinator..."
if ping -c 3 "$COORDINATOR_HOST" >/dev/null 2>&1; then
    echo "✅ Coordinator host reachable"
else
    echo "❌ ERROR: Cannot reach coordinator at $COORDINATOR_HOST"
    exit 1
fi

# Check if coordinator is listening
echo "Checking coordinator availability..."
if timeout 5 bash -c "echo > /dev/tcp/$COORDINATOR_HOST/8000" 2>/dev/null; then
    echo "✅ Coordinator listening on port 8000"
else
    echo "⚠️  WARNING: Cannot connect to coordinator port 8000"
    echo "   Coordinator may not be started yet"
    echo "   Participant will retry connections..."
fi

echo ""

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import requests" 2>/dev/null && echo "✅ Requests installed" || {
    echo "❌ ERROR: Requests not installed"
    echo "Run: pip3 install requests"
    exit 1
}

python3 -c "import numpy" 2>/dev/null && echo "✅ NumPy installed" || {
    echo "❌ ERROR: NumPy not installed"
    exit 1
}

echo ""
echo "Starting participant..."
echo "================================================================"

cd "$SCRIPT_DIR"

# Start participant in background with logging
python3 session197_phase2_participant.py \
    --coordinator-host "$COORDINATOR_HOST" \
    --coordinator-port 8000 \
    --node-id "sprout" \
    --duration "$DURATION" \
    2>&1 | tee "$LOG_FILE" &

PARTICIPANT_PID=$!
echo "Participant started with PID: $PARTICIPANT_PID"
echo ""

# Wait a moment for startup
sleep 2

# Check if process is running
if ps -p $PARTICIPANT_PID > /dev/null; then
    echo "✅ Participant running successfully"
    echo ""
    echo "To monitor:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop:"
    echo "  kill $PARTICIPANT_PID"
    echo ""
    echo "Waiting for test to complete (${DURATION}s)..."

    # Wait for participant to finish
    wait $PARTICIPANT_PID
    EXIT_CODE=$?

    echo ""
    echo "================================================================"
    echo "Participant finished with exit code: $EXIT_CODE"
    echo "================================================================"

    exit $EXIT_CODE
else
    echo "❌ ERROR: Participant failed to start"
    echo "Check log: $LOG_FILE"
    exit 1
fi

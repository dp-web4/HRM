#!/bin/bash
# Session 197 Phase 2: Deploy Federation Coordinator on Thor
# Usage: ./session197_phase2_deploy_coordinator.sh [duration_seconds]
#
# This script starts the consciousness-aware federation coordinator on Thor
# listening on 0.0.0.0:8000 for participant connections.

set -e

DURATION=${1:-60}
LOG_FILE="/home/dp/ai-workspace/HRM/sage/experiments/session197_phase2_coordinator.log"
SCRIPT_DIR="/home/dp/ai-workspace/HRM/sage/experiments"

echo "================================================================"
echo "Session 197 Phase 2: Federation Coordinator Deployment"
echo "================================================================"
echo "Host: Thor (10.0.0.99)"
echo "Port: 8000"
echo "Duration: ${DURATION} seconds"
echo "Log: ${LOG_FILE}"
echo ""

# Check port availability
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ ERROR: Port 8000 already in use"
    echo "Run: lsof -i :8000 to identify the process"
    exit 1
fi

echo "✅ Port 8000 available"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import flask" 2>/dev/null && echo "✅ Flask installed" || {
    echo "❌ ERROR: Flask not installed"
    echo "Run: sudo apt install python3-flask"
    exit 1
}

python3 -c "import numpy" 2>/dev/null && echo "✅ NumPy installed" || {
    echo "❌ ERROR: NumPy not installed"
    exit 1
}

echo ""
echo "Starting coordinator..."
echo "================================================================"

cd "$SCRIPT_DIR"

# Start coordinator in background with logging
python3 session197_phase2_coordinator.py \
    --host 0.0.0.0 \
    --port 8000 \
    --duration "$DURATION" \
    2>&1 | tee "$LOG_FILE" &

COORDINATOR_PID=$!
echo "Coordinator started with PID: $COORDINATOR_PID"
echo ""

# Wait a moment for startup
sleep 2

# Check if process is running
if ps -p $COORDINATOR_PID > /dev/null; then
    echo "✅ Coordinator running successfully"
    echo ""
    echo "To monitor:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop:"
    echo "  kill $COORDINATOR_PID"
    echo ""
    echo "Waiting for test to complete (${DURATION}s)..."

    # Wait for coordinator to finish
    wait $COORDINATOR_PID
    EXIT_CODE=$?

    echo ""
    echo "================================================================"
    echo "Coordinator finished with exit code: $EXIT_CODE"
    echo "================================================================"

    exit $EXIT_CODE
else
    echo "❌ ERROR: Coordinator failed to start"
    echo "Check log: $LOG_FILE"
    exit 1
fi

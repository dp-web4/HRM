#!/bin/bash
#
# Start SAGE Resident Daemon for Thor
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/dp/ai-workspace/HRM/logs/sage-daemon"
PID_FILE="/tmp/sage-daemon.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "SAGE daemon already running (PID: $PID)"
        echo "Use stop_sage.sh to stop it first"
        exit 1
    fi
fi

echo "==================================="
echo "Starting SAGE Resident Daemon"
echo "==================================="
echo ""
echo "Model: Phi-4-mini-instruct (7B)"
echo "Port: 8765"
echo "Logs: $LOG_DIR"
echo ""

# Start daemon in background
nohup python3 "$SCRIPT_DIR/sage_server.py" \
    --model /home/dp/ai-workspace/HRM/model-zoo/phi-4-mini \
    --port 8765 \
    > "$LOG_DIR/sage-daemon.log" 2>&1 &

# Save PID
echo $! > "$PID_FILE"

echo "SAGE daemon starting (PID: $!)"
echo ""
echo "Waiting for startup..."
sleep 5

# Check if it started successfully
if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
    echo "✓ SAGE daemon started successfully"
    echo ""
    echo "Health check: http://localhost:8765/health"
    echo "View logs: tail -f $LOG_DIR/sage-daemon.log"
    echo ""
    echo "Use stop_sage.sh to shut down"
else
    echo "✗ SAGE daemon failed to start"
    echo "Check logs: $LOG_DIR/sage-daemon.log"
    rm -f "$PID_FILE"
    exit 1
fi

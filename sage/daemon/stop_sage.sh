#!/bin/bash
#
# Stop SAGE Resident Daemon
#

PID_FILE="/tmp/sage-daemon.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "SAGE daemon not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "SAGE daemon not running (stale PID file)"
    rm -f "$PID_FILE"
    exit 0
fi

echo "Stopping SAGE daemon (PID: $PID)..."

# Send SIGTERM for graceful shutdown
kill -TERM $PID

# Wait for shutdown (up to 30 seconds)
for i in {1..30}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✓ SAGE daemon stopped"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p $PID > /dev/null 2>&1; then
    echo "Daemon didn't stop gracefully, force killing..."
    kill -9 $PID
    rm -f "$PID_FILE"
    echo "✓ SAGE daemon forcefully stopped"
fi

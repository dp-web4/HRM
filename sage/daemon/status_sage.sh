#!/bin/bash
#
# Check SAGE Daemon Status
#

PID_FILE="/tmp/sage-daemon.pid"

echo "==================================="
echo "SAGE Resident Daemon - Status"
echo "==================================="
echo ""

# Check PID file
if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING"
    echo "PID file not found"
    exit 1
fi

PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p $PID > /dev/null 2>&1; then
    echo "Status: NOT RUNNING"
    echo "Stale PID file (process $PID not found)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Status: RUNNING"
echo "PID: $PID"
echo ""

# Try to get health info via API
echo "Health Check:"
curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null || echo "  API not responding"

echo ""
echo "Process Info:"
ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd

echo ""
echo "GPU Memory:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$PID" || echo "  Not using GPU (or nvidia-smi not available)"

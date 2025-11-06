#!/bin/bash
# Simple session monitor - shows progress every 10s

while true; do
    clear
    echo "=== SAGE SESSION #3 MONITOR ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""

    # Check if session is running
    if pgrep -f "sage_session_3_7b.py" > /dev/null; then
        echo "Status: RUNNING ✓"

        # Get last few lines from log
        echo ""
        echo "--- Latest output ---"
        tail -15 /home/dp/ai-workspace/HRM/sage/irp/session_3_log.txt 2>/dev/null | grep -E "Turn|SAGE:|Energy:|Trust:|Complete" || echo "Waiting for output..."

        # Memory usage
        echo ""
        echo "--- System Resources ---"
        mem=$(ps aux | grep "sage_session_3_7b" | grep -v grep | awk '{print $6/1024/1024 " GB"}')
        cpu=$(ps aux | grep "sage_session_3_7b" | grep -v grep | awk '{print $3 "%"}')
        echo "RAM: ${mem:-N/A}"
        echo "CPU: ${cpu:-N/A}"

        # GPU
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        echo "GPU: ${gpu:-N/A}%"

    else
        echo "Status: COMPLETED or NOT RUNNING"
        echo ""
        echo "--- Final Summary ---"
        grep -E "SESSION.*COMPLETE|Trust.*→|Energy:" /home/dp/ai-workspace/HRM/sage/irp/session_3_log.txt 2>/dev/null | tail -10
        break
    fi

    sleep 10
done

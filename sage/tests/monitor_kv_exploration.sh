#!/bin/bash
# Monitor KV cache exploration progress and report when complete

LOG_FILE="/tmp/kv_cache_exploration.log"
CHECK_INTERVAL=30

echo "Monitoring KV cache exploration..."
echo "Log: $LOG_FILE"
echo ""

while true; do
    if [ -f "$LOG_FILE" ]; then
        # Check if exploration completed
        if grep -q "Investigation complete!" "$LOG_FILE"; then
            echo "✅ KV Cache Exploration COMPLETE"
            echo ""
            echo "=== FINAL SUMMARY ==="
            tail -100 "$LOG_FILE" | grep -A 50 "SUMMARY"
            echo ""
            echo "Full log available at: $LOG_FILE"
            exit 0
        fi

        # Check for errors
        if grep -q "ERROR\|FAILED\|Traceback" "$LOG_FILE"; then
            echo "⚠️  Error detected in exploration"
            tail -50 "$LOG_FILE"
            exit 1
        fi

        # Show progress
        LINES=$(wc -l < "$LOG_FILE")
        LAST_LINE=$(tail -1 "$LOG_FILE")
        echo "[$(date +%H:%M:%S)] Lines: $LINES | $LAST_LINE"
    else
        echo "Waiting for log file to appear..."
    fi

    sleep $CHECK_INTERVAL
done

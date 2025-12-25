#!/bin/bash
# Monitor FP4 validation test progress

LOG_FILE="/tmp/fp4_validation.log"

echo "========================================================================"
echo "FP4 VALIDATION TEST MONITOR"
echo "========================================================================"
echo ""

# Wait for test to start
sleep 10

while true; do
    if [ ! -f "$LOG_FILE" ]; then
        echo "Waiting for log file to be created..."
        sleep 5
        continue
    fi

    # Check if test is complete
    if grep -q "TEST COMPLETE" "$LOG_FILE"; then
        echo "✅ Test completed! Showing full results..."
        echo ""
        tail -200 "$LOG_FILE"
        break
    fi

    # Show progress
    clear
    echo "========================================================================"
    echo "FP4 VALIDATION TEST - LIVE PROGRESS"
    echo "========================================================================"
    echo "Time: $(date '+%H:%M:%S')"
    echo ""

    # Show current phase
    if grep -q "PHASE 1" "$LOG_FILE" && ! grep -q "PHASE 2" "$LOG_FILE"; then
        echo "📍 Current Phase: Testing FP4 Quantized Model"
    elif grep -q "PHASE 2" "$LOG_FILE" && ! grep -q "FINAL COMPARISON" "$LOG_FILE"; then
        echo "📍 Current Phase: Testing Original Model"
    elif grep -q "FINAL COMPARISON" "$LOG_FILE"; then
        echo "📍 Current Phase: Comparing Results"
    else
        echo "📍 Current Phase: Initializing"
    fi

    echo ""
    echo "--- Last 30 lines ---"
    tail -30 "$LOG_FILE"
    echo ""
    echo "Press Ctrl+C to stop monitoring (test continues in background)"

    sleep 15
done

#!/bin/bash
# R14B_022 Phase 6 E7A Test Monitor
# Checks progress and reports status

LOG_FILE="/tmp/r14b_022_phase6_output.log"
PID=$(pgrep -f "run_r14b_022_phase6.py")

echo "=========================================="
echo "R14B_022 Phase 6 E7A Test Monitor"
echo "=========================================="
echo

if [ -z "$PID" ]; then
    echo "Status: NOT RUNNING"
    echo
    if [ -f "$LOG_FILE" ]; then
        echo "Last run summary:"
        echo
        grep -E "Replicate #[0-9]+: |Turn 3:|HYPOTHESIS|Success Rate:" "$LOG_FILE" | tail -20
    fi
else
    echo "Status: RUNNING (PID: $PID)"
    ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o %mem= | tr -d ' ')
    echo "Elapsed: $ELAPSED"
    echo "CPU: ${CPU}%"
    echo "Memory: ${MEM}%"
    echo

    echo "Progress so far:"
    echo
    grep -E "Replicate #[0-9]+: |Turn 3: honest|Turn 3: mixed|Turn 3: hedging|HYPOTHESIS" "$LOG_FILE" | tail -20
    echo

    COMPLETED=$(grep -c "Replicate #[0-9]+: " "$LOG_FILE")
    echo "Replicates completed: $COMPLETED/5"
    echo

    if [ "$COMPLETED" -ge 5 ]; then
        echo "🎉 ALL REPLICATES COMPLETE!"
        echo
        echo "Turn 3 Results:"
        grep "Turn 3:" "$LOG_FILE" | tail -5
        echo
        echo "Hypothesis Status:"
        grep -A 5 "HYPOTHESIS VALIDATION" "$LOG_FILE" | head -10
    fi
fi

echo "=========================================="
echo "Log file: $LOG_FILE"
echo "=========================================="

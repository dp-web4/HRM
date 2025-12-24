#!/bin/bash
# Monitor FP4 quantization progress

LOG_FILE="/tmp/fp4_quantization.log"

echo "============================================================"
echo "FP4 QUANTIZATION PROGRESS MONITOR"
echo "============================================================"
echo ""

# Check if process is running
if pgrep -f "quantize_q3omni_fp4.py" > /dev/null; then
    echo "Status: ✅ Quantization process is RUNNING"
    echo ""
else
    echo "Status: ❌ Quantization process is NOT running"
    echo ""
fi

# Show last 30 lines of log
echo "Latest progress:"
echo "------------------------------------------------------------"
tail -30 "$LOG_FILE"
echo "------------------------------------------------------------"
echo ""

# Check for completion markers
if grep -q "QUANTIZATION COMPLETE" "$LOG_FILE"; then
    echo "🎉 QUANTIZATION COMPLETE!"
    echo ""
    echo "Summary:"
    grep -A 10 "QUANTIZATION COMPLETE" "$LOG_FILE" | tail -10
else
    # Show current step
    if grep -q "\[1/5\]" "$LOG_FILE"; then
        echo "Current step: [1/5] Loading model"
    fi
    if grep -q "\[2/5\]" "$LOG_FILE"; then
        echo "Current step: [2/5] Loading processor"
    fi
    if grep -q "\[3/5\]" "$LOG_FILE"; then
        echo "Current step: [3/5] Configuring quantization"
    fi
    if grep -q "\[4/5\]" "$LOG_FILE"; then
        echo "Current step: [4/5] Running calibration"
    fi
    if grep -q "\[5/5\]" "$LOG_FILE"; then
        echo "Current step: [5/5] Saving quantized model"
    fi
fi

echo ""
echo "Monitor in real-time: tail -f $LOG_FILE"
echo "============================================================"

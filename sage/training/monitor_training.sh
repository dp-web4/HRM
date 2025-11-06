#!/bin/bash
#
# Monitor Vision VAE Training
#
# Shows live training progress and GPU utilization
#

echo "==================================="
echo "Vision Puzzle VAE Training Monitor"
echo "==================================="
echo ""

# Check if training is running
PID_FILE="sage/training/vision_vae_training.pid"
LOG_FILE="sage/training/vision_vae_training.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training is RUNNING (PID: $PID)"
    else
        echo "✗ Training process not found (may have finished or crashed)"
    fi
else
    echo "✗ No PID file found - training may not have started"
fi

echo ""
echo "--- Recent Log Output ---"
tail -30 "$LOG_FILE"

echo ""
echo "--- GPU Status ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
else
    echo "nvidia-smi not available"
fi

echo ""
echo "To follow training live: tail -f $LOG_FILE"
echo "To kill training: kill $(cat $PID_FILE 2>/dev/null || echo 'N/A')"

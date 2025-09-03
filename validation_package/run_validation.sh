#!/bin/bash
# Example validation script for HRM models

# Set paths
MODEL_PATH="${1:-checkpoints/hrm_arc_best.pt}"
DATA_PATH="${2:-arc-aug-100}"
DEVICE="${3:-cuda}"

echo "================================================"
echo "HRM ARC Validation"
echo "================================================"
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"
echo "Device: $DEVICE"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    exit 1
fi

# Run validation
python validate_arc.py \
    --model "$MODEL_PATH" \
    --data "$DATA_PATH" \
    --device "$DEVICE" \
    --batch-size 8

echo ""
echo "Validation complete!"
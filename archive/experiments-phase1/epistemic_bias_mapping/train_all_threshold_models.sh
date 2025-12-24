#!/bin/bash
# Sequential training of all threshold detection models
# Each model trains to completion before starting the next

set -e  # Exit on error

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Create logs directory
mkdir -p training_logs

echo "================================================================================"
echo "Training All Threshold Detection Models"
echo "================================================================================"
echo "This will train 4 models sequentially:"
echo "  - 40 examples (18 epochs) ~ 30 min"
echo "  - 60 examples (14 epochs) ~ 30 min"
echo "  - 80 examples (11 epochs) ~ 30 min"
echo "  - 100 examples (9 epochs) ~ 30 min"
echo ""
echo "Total estimated time: ~2 hours"
echo "================================================================================"
echo ""

# Train each model
for size in 40 60 80 100; do
    echo ""
    echo "================================================================================"
    echo "Starting training for ${size}-example model"
    echo "================================================================================"
    echo "Start time: $(date)"
    echo ""

    LOG_FILE="training_logs/${size}examples_$(date +%Y%m%d_%H%M%S).log"

    # Train the model (with output to both console and log)
    python3 train_threshold_models.py $size 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ ${size}-example model training complete!"
        echo "  Log: $LOG_FILE"
        echo "  End time: $(date)"
    else
        echo ""
        echo "❌ ${size}-example model training failed with exit code $EXIT_CODE"
        echo "  Log: $LOG_FILE"
        exit $EXIT_CODE
    fi

    echo ""
done

echo ""
echo "================================================================================"
echo "All Models Trained Successfully!"
echo "================================================================================"
echo "Trained models:"
echo "  - threshold_models/40examples_model/final_model"
echo "  - threshold_models/60examples_model/final_model"
echo "  - threshold_models/80examples_model/final_model"
echo "  - threshold_models/100examples_model/final_model"
echo ""
echo "Next step: Run experiment_orchestrator.py to test all models"
echo "================================================================================"

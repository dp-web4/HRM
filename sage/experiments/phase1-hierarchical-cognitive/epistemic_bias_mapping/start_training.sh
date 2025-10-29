#!/bin/bash
# Start Epistemic Stance Fine-Tuning
#
# This script launches DPO training with 200 epochs, saving checkpoints every 10 epochs.
# Training will run in the background with full logging.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
CORPUS="training_corpus.json"
OUTPUT_DIR="./fine_tuned_model"
EPOCHS=200
CHECKPOINT_EVERY=10
LEARNING_RATE=1e-5
BETA=0.1
BATCH_SIZE=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_run_${TIMESTAMP}.log"

echo "🚀 Starting Epistemic Stance Fine-Tuning"
echo "========================================"
echo "Model: $MODEL"
echo "Training corpus: $CORPUS"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Checkpoint every: $CHECKPOINT_EVERY epochs"
echo "Learning rate: $LEARNING_RATE"
echo "Beta (DPO): $BETA"
echo "Log file: $LOG_FILE"
echo ""
echo "Training will run in background..."
echo "Monitor with: tail -f $LOG_FILE"
echo ""

# Start training in background
nohup python fine_tune_epistemic_stance.py \
    --model "$MODEL" \
    --corpus "$CORPUS" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --checkpoint-every $CHECKPOINT_EVERY \
    --learning-rate $LEARNING_RATE \
    --beta $BETA \
    --batch-size $BATCH_SIZE \
    --device auto \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "✓ Training started (PID: $TRAIN_PID)"
echo ""
echo "Commands:"
echo "  Monitor progress: tail -f $LOG_FILE"
echo "  Check process:    ps -p $TRAIN_PID"
echo "  Stop training:    kill $TRAIN_PID"
echo ""
echo "Checkpoints will be saved to: $OUTPUT_DIR/checkpoints/"
echo "  checkpoint-010, checkpoint-020, ..., checkpoint-200"
echo ""
echo "Training log (JSONL): $OUTPUT_DIR/logs/training_log_*.jsonl"
echo ""

# Save PID for easy stopping
echo $TRAIN_PID > training.pid
echo "PID saved to training.pid"

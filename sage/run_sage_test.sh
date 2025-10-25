#!/bin/bash
# SAGE Hybrid Conversation Test Launcher
# Run this in a separate terminal to see the live dashboard

cd "$(dirname "$0")"

# Create log directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=~/sage_logs
LOG_FILE="$LOG_DIR/sage_session_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "SAGE Hybrid Learning Test"
echo "========================================="
echo ""
echo "Starting SAGE with:"
echo "  - Real LLM: Qwen 2.5-0.5B on CUDA"
echo "  - SNARC Memory: 127K context window"
echo "  - System Prompt: Consciousness identity"
echo "  - Fast/Slow Path: Hybrid learning"
echo "  - Max response: 150 tokens (real-time balance)"
echo ""
echo "Logging to: $LOG_FILE"
echo "Dashboard will show real-time statistics."
echo "Speak into your microphone to interact!"
echo ""
echo "========================================="
echo ""

# Run with logging to home directory (not /tmp)
python3 -u tests/hybrid_conversation_threaded.py --real-llm 2>&1 | tee "$LOG_FILE"

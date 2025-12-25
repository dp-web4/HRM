#!/bin/bash
# Monitor weight-only FP4 quantization progress

LOG_FILE="/tmp/fp4_weight_only.log"

echo "=== FP4 Weight-Only Quantization Monitor ==="
echo "Log file: $LOG_FILE"
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found!"
    exit 1
fi

# Show current status
echo "📊 Current Status:"
echo "─────────────────────────────────────────────────────────"

# Check for each stage
if grep -q "\[1/4\] Loading Q3-Omni model" "$LOG_FILE"; then
    echo "✅ Stage 1/4: Loading model"
fi

if grep -q "✅ Model loaded successfully" "$LOG_FILE"; then
    echo "✅ Stage 1/4 Complete: Model loaded"

    # Show model size
    grep "Model size:" "$LOG_FILE" | tail -1
    grep "Original memory:" "$LOG_FILE" | tail -1
fi

if grep -q "\[2/4\] Loading processor" "$LOG_FILE"; then
    echo "✅ Stage 2/4: Loading processor"
fi

if grep -q "✅ Processor loaded" "$LOG_FILE"; then
    echo "✅ Stage 2/4 Complete: Processor loaded"
fi

if grep -q "\[3/4\] Applying FP4 weight-only quantization" "$LOG_FILE"; then
    echo "🔄 Stage 3/4: Quantizing (this takes 5-10 min)"
fi

if grep -q "✅ Model quantized successfully" "$LOG_FILE"; then
    echo "✅ Stage 3/4 Complete: Quantization done"
fi

if grep -q "\[4/4\] Saving quantized model" "$LOG_FILE"; then
    echo "💾 Stage 4/4: Saving model"
fi

if grep -q "✅ Model saved successfully" "$LOG_FILE"; then
    echo "✅ Stage 4/4 Complete: Model saved"
fi

if grep -q "QUANTIZATION COMPLETE" "$LOG_FILE"; then
    echo ""
    echo "🎉 QUANTIZATION COMPLETE!"
    echo "─────────────────────────────────────────────────────────"

    # Show results
    grep "Original model:" "$LOG_FILE" | tail -1
    grep "Quantized model:" "$LOG_FILE" | tail -1
    grep "Compression:" "$LOG_FILE" | tail -1
    grep "Savings:" "$LOG_FILE" | tail -1
fi

# Check for errors
if grep -q "❌" "$LOG_FILE"; then
    echo ""
    echo "⚠️  ERRORS DETECTED:"
    echo "─────────────────────────────────────────────────────────"
    grep "❌" "$LOG_FILE"
fi

# Show last 10 lines
echo ""
echo "📝 Last 10 lines:"
echo "─────────────────────────────────────────────────────────"
tail -10 "$LOG_FILE"

echo ""
echo "=== End of Status ==="
echo ""
echo "To see full log: tail -f $LOG_FILE"

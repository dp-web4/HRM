#!/bin/bash
# Monitor vLLM build progress for Jetson AGX Thor

LOG_FILE="/tmp/vllm_build_retry.log"

echo "=== vLLM Build Monitor ==="
echo "Platform: Jetson AGX Thor (ARM64 + CUDA 13)"
echo "Log file: $LOG_FILE"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Build log not found at $LOG_FILE"
    exit 1
fi

echo "📊 Build Statistics"
echo "-------------------"
LINES=$(wc -l < "$LOG_FILE")
echo "Log lines: $LINES"
echo "File size: $(du -h $LOG_FILE | cut -f1)"
echo ""

echo "🔍 Build Phase Detection"
echo "------------------------"
if grep -q "Installing build dependencies" "$LOG_FILE"; then
    echo "✅ Phase 1: Dependency installation (completed)"
else
    echo "⏳ Phase 1: Dependency installation"
fi

if grep -q "Building editable for vllm" "$LOG_FILE"; then
    echo "✅ Phase 2: CMake configuration (started)"
else
    echo "⏳ Phase 2: CMake configuration"
fi

if grep -q "Building CXX object" "$LOG_FILE"; then
    echo "✅ Phase 3: C++ compilation (started)"
    CXX_COUNT=$(grep -c "Building CXX object" "$LOG_FILE")
    echo "   Compiled objects: $CXX_COUNT"
else
    echo "⏳ Phase 3: C++ compilation"
fi

if grep -q "CUDA" "$LOG_FILE" && grep -q "Building" "$LOG_FILE"; then
    echo "✅ Phase 4: CUDA kernels (started)"
else
    echo "⏳ Phase 4: CUDA kernels"
fi

if grep -q "Successfully built" "$LOG_FILE"; then
    echo "✅ Phase 5: Build complete!"
else
    echo "⏳ Phase 5: Package assembly"
fi

echo ""
echo "⚠️  Error Detection"
echo "-------------------"
ERROR_COUNT=$(grep -ci "error:" "$LOG_FILE" || echo "0")
FAILED_COUNT=$(grep -c "FAILED:" "$LOG_FILE" || echo "0")

if [ "$ERROR_COUNT" -gt 0 ] || [ "$FAILED_COUNT" -gt 0 ]; then
    echo "❌ Errors found: $ERROR_COUNT errors, $FAILED_COUNT failures"
    echo ""
    echo "Recent errors:"
    grep -i "error:" "$LOG_FILE" | tail -5
else
    echo "✅ No errors detected"
fi

echo ""
echo "📝 Recent Activity (last 10 lines)"
echo "-----------------------------------"
tail -10 "$LOG_FILE"

echo ""
echo "---"
echo "To monitor in real-time: tail -f $LOG_FILE"
echo "To check full log: less $LOG_FILE"

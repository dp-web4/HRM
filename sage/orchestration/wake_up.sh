#!/bin/bash
# SAGE Wake-Up Helper
#
# Run this at session start to check if SAGE needs attention.
# Exit code 0 = no attention needed, continue normally
# Exit code 1 = attention needed, review wake signal

WAKE_SIGNAL="/tmp/claude_wake_signal_sage.md"
ATTENTION_FILE="/tmp/claude_last_attention_sage.txt"

# Update last attention timestamp
date --iso-8601=seconds > "$ATTENTION_FILE"

if [ -f "$WAKE_SIGNAL" ]; then
    echo "🔔 ATTENTION NEEDED - Wake signal detected!"
    echo ""
    cat "$WAKE_SIGNAL"
    echo ""
    echo "---"
    echo "💡 TIP: Remove signal when addressed: rm $WAKE_SIGNAL"
    exit 1
else
    echo "✅ No urgent attention needed"
    echo ""
    echo "Current SAGE status:"
    echo "  Base: /home/dp/ai-workspace/HRM/sage"
    echo "  Orchestration: $(ls /home/dp/ai-workspace/HRM/sage/orchestration/*.py 2>/dev/null | wc -l) scripts"
    echo "  Agents: $(find /home/dp/ai-workspace/HRM/sage/orchestration/agents -name "*.py" 2>/dev/null | wc -l) agent files"

    # Show last status update
    if [ -f "/home/dp/ai-workspace/HRM/sage/STATUS.md" ]; then
        echo ""
        echo "Last status (first 5 lines):"
        head -5 /home/dp/ai-workspace/HRM/sage/STATUS.md
    fi

    echo ""
    echo "SAGE is operating normally. Continue with current tasks."
    exit 0
fi

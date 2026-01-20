#!/bin/bash
##############################################################################
# Monitor v2.0 Deployment and Session Progress
#
# Usage: ./monitor_v2_deployment.sh [session_number]
#
# Monitors SAGE session execution and v2.0 deployment effectiveness.
# If session_number provided, analyzes that specific session.
# Otherwise, monitors next scheduled session.
#
# Created: 2026-01-20 06:06 PST (Thor Autonomous Session)
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION=${1:-"next"}

echo "======================================================================"
echo "SAGE v2.0 Deployment Monitor"
echo "======================================================================"
echo ""

# Show current schedule
echo "Current Schedule:"
python3 schedule_next_session.py --run
echo ""

# If monitoring next session, wait for it
if [ "$SESSION" = "next" ]; then
    # Get next session number
    NEXT_SESSION=$(python3 schedule_next_session.py --run 2>/dev/null | grep "PRIMARY Session" | awk '{print $3}' | head -1)

    if [ -z "$NEXT_SESSION" ]; then
        echo "❌ Could not determine next session number"
        exit 1
    fi

    echo "Monitoring for Session $NEXT_SESSION..."
    SESSION_FILE="../sessions/text/session_${NEXT_SESSION}.json"

    echo "Waiting for session to complete..."
    echo "(File: $SESSION_FILE)"
    echo ""

    # Wait for session file (timeout after 30 minutes)
    TIMEOUT=1800  # 30 minutes
    ELAPSED=0
    INTERVAL=10

    while [ ! -f "$SESSION_FILE" ] && [ $ELAPSED -lt $TIMEOUT ]; do
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        echo -ne "\rWaiting... ${ELAPSED}s / ${TIMEOUT}s"
    done
    echo ""

    if [ ! -f "$SESSION_FILE" ]; then
        echo "⚠️  Session file not created after $TIMEOUT seconds"
        echo "Check if session is running or if there's an error"
        exit 1
    fi

    echo "✅ Session $NEXT_SESSION completed"
    echo ""

    SESSION=$NEXT_SESSION
fi

# Format session number with leading zeros
SESSION_FORMATTED=$(printf "%03d" $SESSION)
SESSION_FILE="../sessions/text/session_${SESSION_FORMATTED}.json"

if [ ! -f "$SESSION_FILE" ]; then
    echo "❌ Session file not found: $SESSION_FILE"
    exit 1
fi

echo "======================================================================"
echo "Analyzing Session $SESSION"
echo "======================================================================"
echo ""

# Show session metadata
echo "Session Metadata:"
echo "-----------------"
cat "$SESSION_FILE" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
print(f\"  Session: {data.get('session', 'N/A')}\")
print(f\"  Phase: {data.get('phase', 'N/A')}\")
print(f\"  Generation Mode: {data.get('generation_mode', 'N/A')}\")
print(f\"  Start: {data.get('start', 'N/A')}\")
print(f\"  End: {data.get('end', 'N/A')}\")

# Check for v2.0 indicators
mode = data.get('generation_mode', '')
if 'v2' in mode.lower() or 'v2.0' in mode.lower():
    print(\"\n✅ v2.0 CONFIRMED\")
elif 'identity_anchored' in mode.lower():
    print(\"\n⚠️  v1.0 detected (not v2.0)\")
else:
    print(f\"\n⚠️  Unknown generation mode: {mode}\")
"
echo ""

# Run integrated coherence analysis
echo "Running Integrated Coherence Analysis..."
echo "-----------------"
python3 ../../analysis/integrated_coherence_analyzer.py "$SESSION_FILE"
echo ""

# Compare to previous session if available
PREV_SESSION=$((SESSION - 1))
PREV_SESSION_FORMATTED=$(printf "%03d" $PREV_SESSION)
PREV_SESSION_FILE="../sessions/text/session_${PREV_SESSION_FORMATTED}.json"

if [ -f "$PREV_SESSION_FILE" ]; then
    echo "======================================================================"
    echo "Comparison: Session $PREV_SESSION vs Session $SESSION"
    echo "======================================================================"
    echo ""

    python3 -c "
import json
import sys

# Load both sessions
with open('$PREV_SESSION_FILE') as f:
    prev = json.load(f)
with open('$SESSION_FILE') as f:
    curr = json.load(f)

# Extract SAGE responses
prev_responses = [t['text'] for t in prev.get('conversation', []) if t.get('speaker') == 'SAGE']
curr_responses = [t['text'] for t in curr.get('conversation', []) if t.get('speaker') == 'SAGE']

# Basic metrics
def calc_metrics(responses):
    if not responses:
        return None
    word_counts = [len(r.split()) for r in responses]
    incomplete = sum(1 for r in responses if r.endswith('...') or r.endswith('…'))
    return {
        'avg_words': sum(word_counts) / len(word_counts),
        'total_responses': len(responses),
        'incomplete_count': incomplete,
        'incomplete_pct': (incomplete / len(responses)) * 100
    }

prev_metrics = calc_metrics(prev_responses)
curr_metrics = calc_metrics(curr_responses)

if prev_metrics and curr_metrics:
    print(f\"Metric                    S{PREV_SESSION:03d}        S{SESSION:03d}        Change\")
    print(\"-\" * 65)
    print(f\"Avg word count:           {prev_metrics['avg_words']:6.1f}     {curr_metrics['avg_words']:6.1f}     {curr_metrics['avg_words'] - prev_metrics['avg_words']:+6.1f}\")
    print(f\"Total responses:          {prev_metrics['total_responses']:6d}     {curr_metrics['total_responses']:6d}     {curr_metrics['total_responses'] - prev_metrics['total_responses']:+6d}\")
    print(f\"Incomplete responses:     {prev_metrics['incomplete_count']:6d}     {curr_metrics['incomplete_count']:6d}     {curr_metrics['incomplete_count'] - prev_metrics['incomplete_count']:+6d}\")
    print(f\"Incomplete %:             {prev_metrics['incomplete_pct']:6.1f}%    {curr_metrics['incomplete_pct']:6.1f}%    {curr_metrics['incomplete_pct'] - prev_metrics['incomplete_pct']:+6.1f}%\")
    print()

    # Interpretation
    word_change = curr_metrics['avg_words'] - prev_metrics['avg_words']
    incomplete_change = curr_metrics['incomplete_pct'] - prev_metrics['incomplete_pct']

    print(\"Interpretation:\")
    if word_change < -10:
        print(\"  ✅ Word count reduced (less bloat)\")
    elif word_change > 10:
        print(\"  ⚠️  Word count increased (more bloat)\")
    else:
        print(\"  ➡️  Word count stable\")

    if incomplete_change < -20:
        print(\"  ✅ Incomplete responses significantly reduced\")
    elif incomplete_change > 20:
        print(\"  ⚠️  Incomplete responses increased\")
    else:
        print(\"  ➡️  Incomplete response rate stable\")
"
    echo ""
fi

echo "======================================================================"
echo "Recommendations"
echo "======================================================================"
echo ""

# Session-specific recommendations
python3 -c "
import json
import sys

with open('$SESSION_FILE') as f:
    data = json.load(f)

mode = data.get('generation_mode', '')

if $SESSION == 32:
    print(\"Session 032 (First v2.0 run - if deployed):\")
    if 'v2' in mode.lower():
        print(\"  ✅ v2.0 confirmed running\")
        print(\"  Expected: Response quality more stable than S031\")
        print(\"  Expected: Self-reference likely still 0% (takes 2-3 sessions)\")
        print(\"  Expected: Exemplar library populated for S033+\")
        print(\"\")
        print(\"  Next: Monitor S033 for 5-15% self-reference emergence\")
    else:
        print(\"  ❌ v2.0 NOT running - still on v1.0 or earlier\")
        print(\"  URGENT: Deploy v2.0 before S033\")
        print(\"  Run: ./switch_to_v2.sh\")
elif $SESSION == 33:
    print(\"Session 033 (v2.0 accumulation expected):\")
    print(\"  CRITICAL: Look for 5-15% self-reference emergence\")
    print(\"  Expected: Coherence 0.48-0.58\")
    print(\"  Expected: Quality stable or improving\")
    print(\"\")
    if 'v2' not in mode.lower():
        print(\"  ⚠️  If v2.0 not running, identity emergence unlikely\")
elif $SESSION >= 34:
    print(f\"Session {SESSION:03d} (Basin escape trajectory expected):\")
    print(\"  Expected: Self-reference 15-30%\")
    print(\"  Expected: Coherence 0.60-0.75 (VERIFIED level)\")
    print(\"  Expected: Sustained improvement trend\")
    print(\"\")
    print(\"  Success criteria: Coherence consistently >0.50 (STANDARD)\")
"

echo "======================================================================"
echo "Monitoring Complete"
echo "======================================================================"

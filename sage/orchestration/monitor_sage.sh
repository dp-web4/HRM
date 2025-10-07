#!/bin/bash
# SAGE Monitoring Cycle
#
# Autonomous monitoring script that:
# 1. Calculates salience of SAGE development state
# 2. Generates wake signal if attention needed
# 3. Logs monitoring activity
#
# Run this periodically (e.g., every 4 hours via cron)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CYCLE_LOG="/tmp/sage_monitor_cycle.log"
CYCLE_STATE="/tmp/sage_monitor_state.json"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CYCLE_LOG"
}

# Get current cycle number
get_cycle_number() {
    if [ -f "$CYCLE_STATE" ]; then
        python3 -c "import json; print(json.load(open('$CYCLE_STATE')).get('cycle', 0))" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Increment cycle number
increment_cycle() {
    local current=$(get_cycle_number)
    local next=$((current + 1))
    if [ -f "$CYCLE_STATE" ]; then
        python3 -c "import json; s=json.load(open('$CYCLE_STATE')); s['cycle']=$next; json.dump(s, open('$CYCLE_STATE','w'))"
    else
        echo "{\"cycle\": $next}" > "$CYCLE_STATE"
    fi
    echo $next
}

# Main monitoring cycle
main() {
    local cycle_num=$(increment_cycle)

    log "===== SAGE Monitor Cycle $cycle_num START ====="

    # Phase 1: Calculate Salience
    log "Phase 1: Calculating salience..."

    # Run salience calculator (may return exit code 1 if attention needed)
    local salience_json
    salience_json=$(python3 "$SCRIPT_DIR/salience_calculator.py" 2>&1) || true

    if [ -n "$salience_json" ] && echo "$salience_json" | python3 -c "import sys, json; json.load(sys.stdin)" &>/dev/null; then
        log "✅ Salience calculation complete"

        # Extract key metrics
        local score=$(echo "$salience_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('salience_score', 0.0))")
        local attention_needed=$(echo "$salience_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('attention_needed', False))")

        log "   Salience Score: $score"
        log "   Attention Needed: $attention_needed"

        # Phase 2: Generate Wake Signal (if needed)
        log "Phase 2: Wake signal generation..."

        # Run wake signal generator (may return exit code 1 if signal created)
        echo "$salience_json" | python3 "$SCRIPT_DIR/wake_signal_generator.py" || true

        if [ "$attention_needed" = "True" ]; then
            log "🔔 WAKE SIGNAL GENERATED - Attention required"
        else
            log "✅ No wake signal needed - SAGE operating normally"
        fi

    else
        log "❌ Error calculating salience: $salience_json"
    fi

    # Phase 3: Summary
    log "Phase 3: Cycle summary"

    # Count various metrics
    local training_logs=$(find /home/dp/ai-workspace/HRM/sage/training -name "*.log" 2>/dev/null | wc -l || echo 0)
    local agent_count=$(find /home/dp/ai-workspace/HRM/sage/orchestration/agents -name "*.py" 2>/dev/null | wc -l || echo 0)
    local wake_signal_exists="No"
    [ -f "/tmp/claude_wake_signal_sage.md" ] && wake_signal_exists="Yes"

    log "   Training logs: $training_logs"
    log "   Active agents: $agent_count"
    log "   Wake signal active: $wake_signal_exists"

    log "===== SAGE Monitor Cycle $cycle_num END ====="
    log ""
}

# Run main function
main

# Exit with code 0 (success)
exit 0

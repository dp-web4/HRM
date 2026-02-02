#!/bin/bash
# Wrapper for autonomous Claude sessions that handles the post-completion hang.
#
# The problem: Claude CLI in -p mode completes all work but the node.js process
# doesn't exit cleanly. This wrapper monitors for completion and terminates
# the process after a grace period.
#
# Usage: run_autonomous.sh <track> [args...]
#   track: "latent" or "training"

TRACK="${1:-latent}"
GRACE_PERIOD=120  # seconds to wait after detecting completion before killing
CLAUDE_BIN="/home/sprout/.local/bin/claude"

case "$TRACK" in
  latent)
    WORKDIR="/home/sprout/ai-workspace"
    SENTINEL_DIR="/home/sprout/ai-workspace/HRM/sage/raising/sessions/latent_exploration"
    PROMPT="Latent behavior exploration session on Sprout. Run these commands in sequence:
1. source ~/ai-workspace/Memory/epistemic/tools/session_start.sh
2. cd /home/sprout/ai-workspace/HRM/sage/raising/scripts && python3 latent_behavior_exploration.py -c -n 5
3. source ~/ai-workspace/Memory/epistemic/tools/session_end.sh 'Sprout latent exploration session'
IMPORTANT: Run the python script directly - do NOT disable CUDA or force CPU. The script handles GPU detection. After running session_end.sh, you are done."
    ;;
  training)
    WORKDIR="/home/sprout/ai-workspace/HRM/sage/raising/tracks/training"
    SENTINEL_DIR="/home/sprout/ai-workspace/HRM/sage/raising/tracks/training/sessions"
    PROMPT="Raising training session on Sprout. Run these commands in sequence:
1. source ~/ai-workspace/Memory/epistemic/tools/session_start.sh
2. python3 training_session.py -c
3. source ~/ai-workspace/Memory/epistemic/tools/session_end.sh 'Sprout training session'
IMPORTANT: Do NOT add --cpu flag or disable CUDA. The Jetson has a GPU and scripts handle detection. After running session_end.sh, you are done."
    ;;
  conversation)
    WORKDIR="/home/sprout/ai-workspace/HRM/sage/raising/scripts"
    SENTINEL_DIR="/home/sprout/ai-workspace/HRM/sage/raising/sessions/text"
    # LoRA re-enabled 2026-02-01 after adding collapse prevention safeguards:
    # - SleepScheduler: min_experiences=30, min_new=10, min_hours=12
    # - ExperienceCollector: repetition detection filters collapsed responses
    # - cycle_003 deleted, scheduler rolled back to cycle_002
    PROMPT="Run a multi-turn conversation session with SAGE. Run these commands in sequence:
1. source ~/ai-workspace/Memory/epistemic/tools/session_start.sh
2. cd /home/sprout/ai-workspace/HRM/sage/raising/scripts && python3 autonomous_conversation.py -c --turns 8 --sleep
3. Read the session transcript that was just created in /home/sprout/ai-workspace/HRM/sage/raising/sessions/text/ (the newest session_NNN.json file)
4. Write a session analysis to /home/sprout/ai-workspace/private-context/autonomous-sessions/sprout-conversation-SNNN-YYYYMMDD.md where NNN is the session number. Include:
   - Session number, date, phase
   - Summary of conversation quality and notable responses
   - Whether LoRA adapters were used
   - Any filtered responses (collapse prevention triggered)
   - Key quotes from SAGE worth noting
5. source ~/ai-workspace/Memory/epistemic/tools/session_end.sh 'Sprout conversation session'
IMPORTANT: Run the python script directly. Do NOT disable CUDA or force CPU. After session_end.sh, you are done."
    ;;
  *)
    echo "Unknown track: $TRACK"
    exit 1
    ;;
esac

# Snapshot existing files before starting
BEFORE=$(ls "$SENTINEL_DIR"/*.json 2>/dev/null | wc -l)

# Start Claude in background
cd "$WORKDIR"
$CLAUDE_BIN --no-session-persistence --dangerously-skip-permissions -p "$PROMPT" &
CLAUDE_PID=$!
echo "Claude started with PID $CLAUDE_PID for track: $TRACK"

# Monitor loop: check if Claude exited or if new session file appeared
COMPLETION_DETECTED=0
COMPLETION_TIME=0

while true; do
    # Check if Claude already exited on its own
    if ! kill -0 $CLAUDE_PID 2>/dev/null; then
        echo "Claude exited on its own"
        wait $CLAUDE_PID
        exit $?
    fi

    # Check for new session files (completion signal)
    AFTER=$(ls "$SENTINEL_DIR"/*.json 2>/dev/null | wc -l)
    if [ "$AFTER" -gt "$BEFORE" ] && [ "$COMPLETION_DETECTED" -eq 0 ]; then
        echo "New session file detected - work complete"
        COMPLETION_DETECTED=1
        COMPLETION_TIME=$SECONDS
    fi

    # If completion detected, give grace period then kill
    if [ "$COMPLETION_DETECTED" -eq 1 ]; then
        ELAPSED=$((SECONDS - COMPLETION_TIME))
        if [ "$ELAPSED" -ge "$GRACE_PERIOD" ]; then
            echo "Grace period ($GRACE_PERIOD s) expired, terminating Claude"
            kill $CLAUDE_PID 2>/dev/null
            sleep 5
            # Force kill if still running
            if kill -0 $CLAUDE_PID 2>/dev/null; then
                kill -9 $CLAUDE_PID 2>/dev/null
            fi
            wait $CLAUDE_PID 2>/dev/null
            exit 0
        fi
    fi

    sleep 10
done

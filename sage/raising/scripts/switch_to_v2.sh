#!/bin/bash
##############################################################################
# Switch SAGE Raising Sessions to v2.0 Identity Intervention
#
# Usage: ./switch_to_v2.sh
#
# This script switches the SAGE raising session runner from v1.0
# (or earlier) to v2.0 identity-anchored intervention.
#
# Safe to run multiple times (idempotent).
#
# Created: 2026-01-20 06:03 PST (Thor Autonomous Session)
##############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "SAGE v2.0 Identity Intervention Deployment"
echo "======================================================================"
echo ""

# Check v2.0 exists and is executable
if [ ! -f "run_session_identity_anchored_v2.py" ]; then
    echo "❌ ERROR: run_session_identity_anchored_v2.py not found"
    exit 1
fi

if [ ! -x "run_session_identity_anchored_v2.py" ]; then
    echo "Making v2.0 executable..."
    chmod +x run_session_identity_anchored_v2.py
fi

echo "✅ v2.0 script found and executable"
echo ""

# Test compile
echo "Testing v2.0 script compilation..."
python3 -m py_compile run_session_identity_anchored_v2.py
echo "✅ v2.0 compiles successfully"
echo ""

# Check for symlink (common pattern)
if [ -L "run_session.py" ]; then
    echo "Found symlink: run_session.py"
    CURRENT_TARGET=$(readlink run_session.py)
    echo "  Current target: $CURRENT_TARGET"

    if [ "$CURRENT_TARGET" = "run_session_identity_anchored_v2.py" ]; then
        echo "✅ Already pointing to v2.0"
    else
        echo "Backing up current symlink..."
        cp -P run_session.py run_session_pre_v2_backup.py || true

        echo "Switching to v2.0..."
        ln -sf run_session_identity_anchored_v2.py run_session.py

        echo "✅ Symlink updated to v2.0"
        ls -la run_session.py
    fi
    echo ""
elif [ -f "run_session.py" ] && [ ! -L "run_session.py" ]; then
    echo "Found regular file: run_session.py (not a symlink)"
    echo "Backing up and creating symlink..."

    mv run_session.py run_session_pre_v2_backup.py
    ln -s run_session_identity_anchored_v2.py run_session.py

    echo "✅ Created symlink to v2.0"
    ls -la run_session.py
    echo ""
else
    echo "No run_session.py found - creating symlink..."
    ln -s run_session_identity_anchored_v2.py run_session.py
    echo "✅ Created symlink to v2.0"
    ls -la run_session.py
    echo ""
fi

# Check state file for configuration
STATE_FILE="../state/identity.json"
if [ -f "$STATE_FILE" ]; then
    echo "Checking state file: $STATE_FILE"

    # Check if it has a runner configuration
    if grep -q "runner\|script\|mode" "$STATE_FILE" 2>/dev/null; then
        echo "⚠️  State file contains runner configuration"
        echo "Manual review may be needed:"
        grep -i "runner\|script\|mode" "$STATE_FILE" || true
        echo ""
        echo "Consider adding or updating:"
        echo '  "session_runner": "identity_anchored_v2"'
    else
        echo "State file doesn't specify runner (uses default)"
    fi
    echo ""
fi

echo "======================================================================"
echo "v2.0 Deployment Complete"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Wait for next session (check schedule_next_session.py --run)"
echo "2. Verify session uses v2.0 (check session JSON metadata)"
echo "3. Analyze results with integrated_coherence_analyzer.py"
echo ""
echo "Expected improvements:"
echo "- Cumulative identity context across sessions"
echo "- Strengthened identity priming"
echo "- Response quality controls (50-80 words)"
echo "- Mid-conversation reinforcement"
echo ""
echo "Monitoring:"
echo "  S032: Cumulative context begins"
echo "  S033: Look for 5-15% self-reference emergence"
echo "  S034-S036: Basin escape trajectory (C: 0.60-0.75)"
echo ""

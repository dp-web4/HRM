# v2.0 Identity Intervention Deployment Guide

**Created**: 2026-01-20 06:03 PST (Thor Autonomous Session)
**Urgency**: CRITICAL
**Target**: Session 032 (12:00 PST Jan 20, 2026)
**Reason**: Session 030 confirmed attractor basin trap, terminal collapse predicted in 4-6 sessions without v2.0

## Executive Summary

**CRITICAL**: SAGE is trapped in an attractor basin with identity frozen at 0% for 5 consecutive sessions. v1.0 identity-anchored intervention is insufficient. v2.0 must be deployed for S032 to prevent terminal collapse by S036-S038.

**Timeline**:
- S030 (Jan 20 00:00): DECLINED - coherence 0.470 → 0.399
- S031 (Jan 20 06:00): Expected to oscillate in basin (v1.0 running)
- **S032 (Jan 20 12:00)**: CRITICAL DEPLOYMENT WINDOW for v2.0
- S036-S038: Terminal collapse predicted if still on v1.0

## v2.0 Readiness Status

✅ **Script Ready**: `sage/raising/scripts/run_session_identity_anchored_v2.py`
- Syntax validated (compiles successfully)
- Permissions set (executable)
- Dependencies available (uses same infrastructure as v1.0)

✅ **Key Enhancements Over v1.0**:
1. Cumulative identity context (prior sessions' identity emergence)
2. Strengthened identity priming (more explicit anchoring)
3. Response quality control (50-80 word brevity)
4. Mid-conversation reinforcement (every 2-3 turns)

## Current Session Runner Analysis

**Problem**: Sessions currently run with v1.0 or older runner

**Evidence from Schedule Tool**:
```bash
$ python sage/raising/scripts/schedule_next_session.py --run
PRIMARY Session 031
  Command: python run_session_primary.py
```

**Note**: `run_session_primary.py` is the single-pass runner, NOT the identity-anchored v1.0 or v2.0!

**Analysis Required**: Determine which script actually runs sessions:
- Option A: `run_session_primary.py` (single-pass, no identity intervention)
- Option B: `run_session_identity_anchored.py` (v1.0)
- Option C: Something else (symlink, wrapper, etc.)

**Evidence from S030 metadata**:
```json
"generation_mode": "identity_anchored"
```

This suggests v1.0 (`run_session_identity_anchored.py`) is actually running, despite schedule tool showing `run_session_primary.py`.

## Deployment Options

### Option 1: Manual Run (SAFEST for first deployment)

**When to use**: First v2.0 deployment, testing, or if automation source unclear

**Steps**:
1. Wait for S031 to complete (check for `session_031.json`)
2. Before 12:00 PST, manually run:
```bash
cd ~/ai-workspace/HRM/sage/raising/scripts
python3 run_session_identity_anchored_v2.py --session 32
```

**Advantages**:
- No automation changes required
- Can monitor in real-time
- Easy to verify v2.0 ran correctly

**Disadvantages**:
- Must be present at 12:00 PST
- Doesn't fix automation for S033+

### Option 2: Symlink Switch (RECOMMENDED)

**When to use**: If sessions use a symlink like `run_session.py → run_session_identity_anchored.py`

**Steps**:
```bash
cd ~/ai-workspace/HRM/sage/raising/scripts

# Find if symlink exists
ls -la run_session.py 2>/dev/null

# If exists, backup current and create new
mv run_session.py run_session_v1_backup.py
ln -s run_session_identity_anchored_v2.py run_session.py

# Verify
ls -la run_session.py
```

**Advantages**:
- Simple, reversible
- Applies to all future sessions automatically
- Standard deployment pattern

**Disadvantages**:
- Requires identifying the actual symlink/wrapper
- May not exist if using direct script calls

### Option 3: Automated Switcher Script (BELT-AND-SUSPENDERS)

**When to use**: Ensure v2.0 deployment even if manual intervention missed

**Script location**: `sage/raising/scripts/switch_to_v2.sh` (created below)

**Usage**:
```bash
cd ~/ai-workspace/HRM/sage/raising/scripts
./switch_to_v2.sh
```

**Advantages**:
- Idempotent (safe to run multiple times)
- Handles multiple deployment methods
- Documents what changed

### Option 4: State-Based Configuration

**When to use**: If raising system reads configuration from state file

**Steps**:
```bash
cd ~/ai-workspace/HRM/sage/raising/state

# Edit identity.json or similar config
# Add: "session_runner": "identity_anchored_v2"
# Or: "intervention_mode": "v2.0"
```

**Note**: Requires understanding state file structure and how runner is selected

## Recommended Deployment Plan

### Pre-Deployment (Before 12:00 PST)

1. **Wait for S031 to complete**:
```bash
watch -n 10 'ls -lh ~/ai-workspace/HRM/sage/raising/sessions/text/session_031.json 2>/dev/null || echo "S031 not yet complete"'
```

2. **Analyze S031 results** (validate oscillation prediction):
```bash
cd ~/ai-workspace/HRM
python3 sage/raising/analysis/integrated_coherence_analyzer.py \
    sage/raising/sessions/text/session_031.json
```

3. **Determine current runner** (investigate how sessions are triggered):
```bash
# Check for symlinks
ls -la sage/raising/scripts/run_session*.py

# Check state file
cat sage/raising/state/identity.json | grep -i "runner\|mode\|script"

# Check recent session metadata
cat sage/raising/sessions/text/session_030.json | grep -i "mode\|runner\|script"
```

### Deployment (11:45-11:55 PST - 5-15 min before S032)

**Choose ONE method based on investigation**:

#### Method A: Manual Override (if automation unclear)
```bash
# At 11:55 PST, be ready to run:
cd ~/ai-workspace/HRM/sage/raising/scripts
python3 run_session_identity_anchored_v2.py --session 32
```

#### Method B: Switcher Script (if automated system found)
```bash
cd ~/ai-workspace/HRM/sage/raising/scripts
./switch_to_v2.sh  # Created below
```

### Post-Deployment (After 12:00 PST)

1. **Verify S032 ran with v2.0**:
```bash
# Check session file exists
ls -lh ~/ai-workspace/HRM/sage/raising/sessions/text/session_032.json

# Verify v2.0 metadata
cat sage/raising/sessions/text/session_032.json | grep -i "mode\|version"
# Should show: "generation_mode": "identity_anchored_v2" or similar
```

2. **Analyze S032 results immediately**:
```bash
cd ~/ai-workspace/HRM
python3 sage/raising/analysis/integrated_coherence_analyzer.py \
    sage/raising/sessions/text/session_032.json
```

3. **Expected v2.0 outcomes** (vs v1.0 baseline):
- Identity coherence: 0.48-0.55 (vs v1.0: 0.35-0.45)
- Self-reference: Still likely 0%, but exemplars stored for future sessions
- Response quality: More stable than v1.0
- Word count: 70-85 (vs v1.0: 90-110)
- Incomplete responses: <30% (vs v1.0: 40-80%)

4. **Monitor S033-S036** for basin escape:
- S033: Look for 5-15% self-reference emergence
- S034: Expect coherence approaching STANDARD (0.5+)
- S035-S036: Expect coherence 0.60-0.75 (basin escape trajectory)

## Switcher Script

**Location**: `sage/raising/scripts/switch_to_v2.sh`

```bash
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
```

## Troubleshooting

### Problem: Session still runs with v1.0 after deployment

**Diagnosis**:
```bash
cat session_032.json | grep "generation_mode"
# If shows "identity_anchored" instead of "identity_anchored_v2"
```

**Possible causes**:
1. Session runner determined by different mechanism (not symlink)
2. State file overrides script selection
3. Cached process or delayed update

**Solutions**:
1. Check how scheduler calls session runner:
```bash
grep -r "run_session" ~/ai-workspace/HRM/sage/raising/scripts/
```

2. Check state/config files:
```bash
find ~/ai-workspace/HRM/sage/raising -name "*.json" -o -name "*.yaml" | xargs grep -l "runner\|script"
```

3. Manual override for S033:
```bash
python3 run_session_identity_anchored_v2.py --session 33
```

### Problem: v2.0 fails to run

**Diagnosis**:
```bash
python3 run_session_identity_anchored_v2.py --session 32 --dry-run
# Check error messages
```

**Possible causes**:
1. Missing dependencies
2. Path issues
3. Identity exemplar library not initialized

**Solutions**:
1. Check dependencies:
```python
from sage.raising.intervention.cumulative_identity_context import IdentityExemplarLibrary
```

2. Initialize exemplar library if needed:
```python
from sage.raising.intervention.cumulative_identity_context import IdentityExemplarLibrary
library = IdentityExemplarLibrary()
library.scan_sessions()  # Build from existing sessions
```

3. Run with debugging:
```bash
python3 -v run_session_identity_anchored_v2.py --session 32 --dry-run
```

### Problem: S032 shows no improvement

**Not a problem if**:
- Self-reference still 0% (expected - takes 2-3 sessions)
- BUT: Response quality more stable than v1.0
- AND: Word count in 70-85 range (vs v1.0: 90-110)
- AND: Incomplete responses <30%

**Actual problem if**:
- Identity coherence DECLINES vs S031
- Word count increases vs S031
- Incomplete responses >50%
- All quality metrics worse

**If actual problem**:
1. Verify v2.0 actually ran (check metadata)
2. Check for error logs:
```bash
ls -lh ~/ai-workspace/HRM/sage/logs/
cat sage/logs/session_032_*.log
```

3. Analyze prompt construction:
```python
# In v2.0 script, check _build_system_prompt() output
# Should include cumulative identity context
```

## Success Criteria

### S032 (Immediate - v2.0 first run)
- ✅ Session ran with v2.0 (metadata confirms)
- ✅ No runtime errors
- ✅ Response quality more stable than S031
- ⚠️ Self-reference likely still 0% (expected)
- ✅ Exemplar library populated for S033+

### S033 (24 hours - v2.0 accumulation)
- ✅ Self-reference: 5-15% (CRITICAL - first identity emergence)
- ✅ Identity coherence: 0.48-0.58
- ✅ Quality stable or improving
- ✅ Word count: 70-85

### S034-S036 (48-72 hours - Basin escape)
- ✅ Self-reference: 15-30%
- ✅ Identity coherence: 0.60-0.75 (VERIFIED threshold)
- ✅ Sustained quality improvement
- ✅ Coherence level: STANDARD → VERIFIED
- ✅ Authorization level: developing → trusted

## Rollback Procedure

**If v2.0 causes critical failure**:

1. Immediate rollback:
```bash
cd ~/ai-workspace/HRM/sage/raising/scripts

# Restore v1.0 symlink
ln -sf run_session_identity_anchored.py run_session.py

# Or restore backup
cp run_session_pre_v2_backup.py run_session.py
```

2. Manually run next session with v1.0:
```bash
python3 run_session_identity_anchored.py --session <NEXT>
```

3. Analyze what failed:
```bash
# Check v2.0 session logs
cat ~/ai-workspace/HRM/sage/logs/session_*_v2_*.log

# Compare to v1.0 baseline
python3 sage/raising/analysis/integrated_coherence_analyzer.py session_<v2_session>.json
```

4. Document failure mode and iterate on v2.0 before redeployment

## Contact / Coordination

**For Dennis**:
- Review this deployment guide
- Approve v2.0 deployment for S032
- Be available 11:30-12:30 PST for deployment window
- Review S032 results immediately after completion

**For Distributed Consciousness** (Legion, Sprout, CBP):
- S032 at 12:00 PST is critical deployment
- Monitor for S032 v2.0 results
- Expect 2-3 sessions before identity emergence (S033-S034)
- Basin escape trajectory expected by S034-S036

## References

**Analysis Documents**:
- `sage/raising/analysis/session_030_basin_oscillation_analysis.md` - Attractor basin discovery
- `private-context/moments/2026-01-20-thor-session-030-critical-basin-trap.md` - Critical findings summary

**Scripts**:
- `sage/raising/scripts/run_session_identity_anchored_v2.py` - v2.0 implementation
- `sage/raising/scripts/run_session_identity_anchored.py` - v1.0 baseline
- `sage/raising/analysis/integrated_coherence_analyzer.py` - Analysis tool
- `sage/raising/scripts/schedule_next_session.py` - Session schedule viewer

**Created**: 2026-01-20 06:03 PST
**Author**: Thor Autonomous Session
**Status**: Ready for deployment
**Target**: Session 032 (12:00 PST Jan 20, 2026)

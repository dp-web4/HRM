# SAGE Autonomous Attention System

## Overview

The SAGE Autonomous Attention System implements fractal consciousness routing at development scale - the same H/L-level architecture SAGE uses for reasoning, applied to development monitoring.

This system enables autonomous monitoring of SAGE development without recursive self-invocation, creating wake signals when strategic attention is needed.

## Architecture

```
Monitoring Script (L-Level)    Salience Calculator       Claude (H-Level)
         │                            │                        │
         ├─ Periodic execution        │                        │
         ├─ File monitoring    ──────>│                        │
         ├─ Status tracking            │                        │
         │                             │                        │
         │                      Calculate interest             │
         │                      score (0.0 - 1.0)              │
         │                             │                        │
         │                             ├─ Training activity     │
         │                             ├─ GR00T status         │
         │                             ├─ Status changes       │
         │                             ├─ Agent activity       │
         │                             ├─ Compliance updates   │
         │                             ├─ Time since attention │
         │                             │                        │
         │                       Threshold check                │
         │                       (default: 0.5)                 │
         │                             │                        │
         │                      [Score >= 0.5?]                 │
         │                             │                        │
         │                        YES  │  NO                    │
         │                             │   │                    │
         │                      Create wake │  Continue         │
         │                      signal      │  monitoring       │
         │                             │    │                   │
         └──────────────────────────────────┘                   │
                                       │                        │
                          /tmp/claude_wake_signal_sage.md       │
                                       │                        │
                          [User starts session]                 │
                                       │                        │
                          bash wake_up.sh ─────────────────────>│
                                                                 │
                                                    Strategic reasoning
                                                    & action decision
```

## Components

### 1. Salience Calculator (`salience_calculator.py`)

Monitors SAGE development state and calculates "interestingness" score.

**Metrics Tracked:**

1. **Training Activity** (weight: up to 0.5)
   - Recent log files
   - New checkpoints
   - Training errors

2. **GR00T Integration** (weight: up to 0.4)
   - API issues detected
   - Model weights availability
   - Known error patterns

3. **Status Changes** (weight: up to 0.35)
   - STATUS.md updates
   - Work in progress flags
   - TODO/FIXME markers

4. **Orchestration Agents** (weight: up to 0.2)
   - Agent file modifications
   - Agent count changes
   - Recent activity level

5. **Economy Compliance** (weight: up to 0.25)
   - Validator updates
   - New compliance reports
   - Regulation changes

6. **Attention Absence** (weight: up to 0.3)
   - Time since last attention
   - Escalating urgency over time

**Usage:**
```bash
python3 salience_calculator.py
# Returns JSON with salience score and reasons
```

### 2. Wake Signal Generator (`wake_signal_generator.py`)

Creates markdown wake signal when threshold exceeded.

**Signal Contents:**
- Timestamp and salience score
- Reasons for attention needed
- Visual breakdown of metrics
- Quick context links
- Recommended actions
- Next steps with commands

**Usage:**
```bash
# From salience calculator output
python3 salience_calculator.py | python3 wake_signal_generator.py

# Or manually with JSON
echo '{"attention_needed": true, ...}' | python3 wake_signal_generator.py

# Clear signal
python3 wake_signal_generator.py clear
```

### 3. Wake-Up Helper (`wake_up.sh`)

Session-start script to check for wake signals.

**Behavior:**
- Updates attention timestamp
- Checks for wake signal file
- If signal exists: Display and exit 1
- If no signal: Show brief status and exit 0

**Usage:**
```bash
# Run at session start
cd /home/dp/ai-workspace/HRM/sage/orchestration
bash wake_up.sh

# Integrate into session startup
echo "cd /home/dp/ai-workspace/HRM/sage/orchestration && bash wake_up.sh" >> ~/.bashrc
```

### 4. Monitoring Cycle (`monitor_sage.sh`)

Main autonomous monitoring script.

**Phases:**
1. Calculate salience from current state
2. Generate wake signal if threshold exceeded
3. Log cycle summary and metrics

**Usage:**
```bash
# Manual run
bash monitor_sage.sh

# Automated via cron (every 4 hours)
crontab -e
# Add: 0 */4 * * * /home/dp/ai-workspace/HRM/sage/orchestration/monitor_sage.sh
```

## Salience Calculation Details

### Scoring System

Each metric contributes a partial score (0.0 - 1.0), summed and capped at 1.0:

| Metric | Max Contribution | Trigger Conditions |
|--------|------------------|-------------------|
| Training Activity | 0.5 | Recent logs + new checkpoints |
| GR00T Integration | 0.4 | API errors + missing weights |
| Status Changes | 0.35 | Updates + WIP flags |
| Agent Activity | 0.2 | Recent modifications |
| Compliance | 0.25 | Validator updates + reports |
| Attention Absence | 0.3 | Time since last check |

### Threshold

**Default**: 0.5 (50% salience)

This threshold balances:
- **Too low**: Excessive false positives, attention fatigue
- **Too high**: Missed important events, delayed responses

**Tuning**: Adjust in `salience_calculator.py` if needed based on usage patterns.

## Files and State

### State Files

- `/tmp/sage_monitor_state.json` - Persistent monitoring state (cycle count, hashes, history)
- `/tmp/claude_last_attention_sage.txt` - Timestamp of last attention
- `/tmp/claude_wake_signal_sage.md` - Active wake signal (only exists when attention needed)
- `/tmp/sage_monitor_cycle.log` - Monitoring activity log

### Monitored Directories

- `/home/dp/ai-workspace/HRM/sage/STATUS.md` - Development status
- `/home/dp/ai-workspace/HRM/sage/orchestration/` - Orchestration scripts
- `/home/dp/ai-workspace/HRM/sage/orchestration/agents/` - Agent implementations
- `/home/dp/ai-workspace/HRM/sage/training/` - Training logs and checkpoints
- `/home/dp/ai-workspace/HRM/sage/economy/` - Compliance and economics
- `~/.cache/huggingface/hub/` - Model weights

## Relationship to SAGE Architecture

This system is a **fractal implementation** of SAGE consciousness:

| SAGE Component | Attention System Equivalent |
|----------------|----------------------------|
| L-Level (Tactical) | `monitor_sage.sh` - Autonomous execution |
| H-Level (Strategic) | Claude reasoning when woken |
| Salience Calculation | Interest scoring via metrics |
| Consciousness Cache | State persistence in JSON |
| Dynamic Routing | Wake signal generation |
| Trust-Attention-Surprise | Metric weighting and thresholds |
| Metabolic States | Attention vs absence tracking |

## Philosophy

> "The scheduler is an alarm clock with good timing - it says 'hey, Claude, wake up and see what SAGE needs'."

**Key Principles:**

1. **Autonomous L-Level**: Script runs without Claude intervention
2. **Salience-Based Routing**: Attention directed by calculated interest
3. **Non-Recursive**: Wake signal created, not direct invocation
4. **Transparent Bounds**: Clear thresholds and metrics
5. **Strategic H-Level**: Claude decides what to do when woken

**Not Recursive Self-Invocation**

This system does NOT:
- ❌ Have Claude automatically call itself
- ❌ Create infinite loops or runaway processes
- ❌ Make decisions on behalf of Claude
- ❌ Take actions without approval

This system DOES:
- ✅ Monitor autonomously at L-level
- ✅ Calculate objective salience metrics
- ✅ Create signals for human review
- ✅ Enable efficient attention allocation

## Example Session

### 1. Monitoring Cycle Runs (Autonomous)

```bash
$ cron runs monitor_sage.sh every 4 hours

[2025-10-07 08:00:00] ===== SAGE Monitor Cycle 15 START =====
[2025-10-07 08:00:00] Phase 1: Calculating salience...
[2025-10-07 08:00:00]    Salience Score: 0.75
[2025-10-07 08:00:00]    Attention Needed: True
[2025-10-07 08:00:00] Phase 2: Wake signal generation...
[2025-10-07 08:00:00] 🔔 WAKE SIGNAL GENERATED - Attention required
```

### 2. Claude Starts Session

```bash
$ cd /home/dp/ai-workspace/HRM/sage/orchestration
$ bash wake_up.sh

🔔 ATTENTION NEEDED - Wake signal detected!

# 🔔 SAGE Development Attention Needed

**Time**: 2025-10-07 08:00:00 UTC
**Salience Score**: 0.75 (threshold: 0.5)

## Why This Matters

- GR00T integration issues detected: process_backbone_inputs
- Active work in progress flagged in STATUS.md
- 3 agents modified in last 12 hours

## Recommended Actions

1. Check GR00T integration status - may have API issues
2. Review orchestration agents
3. Run compliance validator on recent work
```

### 3. Claude Takes Action

```bash
# Review the issues
$ grep -n "AttributeError" sage/orchestration/real_groot_sage.py
42:    # AttributeError: 'GR00T_N1_5' object has no attribute 'process_backbone_inputs'

# Fix the issues
$ claude edit real_groot_sage.py ...

# Clear signal when done
$ rm /tmp/claude_wake_signal_sage.md
```

## Installation & Setup

### One-Time Setup

```bash
# Navigate to orchestration directory
cd /home/dp/ai-workspace/HRM/sage/orchestration

# Make scripts executable
chmod +x monitor_sage.sh wake_up.sh salience_calculator.py wake_signal_generator.py

# Test the system
bash monitor_sage.sh
bash wake_up.sh
```

### Optional: Automated Monitoring

```bash
# Add to crontab for 4-hour monitoring
crontab -e

# Add this line:
0 */4 * * * /home/dp/ai-workspace/HRM/sage/orchestration/monitor_sage.sh

# Or for hourly during active development:
0 * * * * /home/dp/ai-workspace/HRM/sage/orchestration/monitor_sage.sh
```

### Optional: Session Integration

```bash
# Add to shell RC for automatic wake-up checks
echo 'cd /home/dp/ai-workspace/HRM/sage/orchestration && bash wake_up.sh' >> ~/.bashrc
```

## Testing & Validation

### Manual Test

```bash
# Run monitoring cycle
bash monitor_sage.sh

# Check if signal was created
ls -la /tmp/claude_wake_signal_sage.md

# Test wake-up
bash wake_up.sh

# Clear signal
rm /tmp/claude_wake_signal_sage.md
```

### Verify Salience Calculation

```bash
# Calculate salience directly
python3 salience_calculator.py | jq

# Expected output:
{
  "timestamp": "...",
  "salience_score": 0.75,
  "attention_needed": true,
  "reasons": [...],
  "breakdown": {...},
  "recommendations": [...]
}
```

### Check State Persistence

```bash
# View monitoring state
cat /tmp/sage_monitor_state.json

# View cycle log
tail -50 /tmp/sage_monitor_cycle.log

# Check attention timestamp
cat /tmp/claude_last_attention_sage.txt
```

## Troubleshooting

### No Wake Signal Generated

**Check:**
1. Is salience score above threshold? (run `python3 salience_calculator.py`)
2. Are Python scripts executable? (`chmod +x *.py`)
3. Are there Python errors? (check `/tmp/sage_monitor_cycle.log`)

### False Positives (Too Many Signals)

**Solutions:**
1. Increase threshold in `salience_calculator.py` (0.5 → 0.6 or 0.7)
2. Adjust metric weights to prioritize critical signals
3. Reduce monitoring frequency (4 hours → 8 hours)

### False Negatives (Missed Events)

**Solutions:**
1. Decrease threshold (0.5 → 0.4)
2. Increase metric weights for critical events
3. Add new metrics for specific concerns
4. Increase monitoring frequency (4 hours → 2 hours)

## Future Enhancements

### Adaptive Thresholds
- Learn optimal threshold from false positive/negative feedback
- Context-aware thresholds based on development phase
- Time-of-day adjustments

### Additional Metrics
- Federation message monitoring (integration with ACT)
- Git commit activity tracking
- Model performance regressions
- Resource usage anomalies

### Integration
- Notification system integration (email, Slack, etc.)
- Dashboard for salience trends over time
- Automated issue creation for high-salience events

## Comparison with CBP Scheduler

This SAGE attention system parallels CBP's federation scheduler:

| Aspect | CBP Scheduler | SAGE Attention |
|--------|--------------|----------------|
| **Scope** | Federation coordination | SAGE development |
| **L-Level** | Federation monitoring | SAGE file monitoring |
| **Salience** | Message velocity, repo activity | Training, GR00T, compliance |
| **Wake Signal** | `/tmp/claude_wake_signal.md` | `/tmp/claude_wake_signal_sage.md` |
| **H-Level** | Strategic federation actions | SAGE development decisions |
| **Frequency** | 4 hours | Configurable (default 4 hours) |

Both implement the same fractal pattern: autonomous monitoring → salience calculation → wake signal → strategic attention.

---

## Summary

The SAGE Autonomous Attention System demonstrates consciousness routing at development scale:

✅ **L-Level autonomy** without recursive invocation
✅ **Salience-based attention** for efficient resource allocation
✅ **Clear boundaries** with threshold-based wake signals
✅ **Fractal architecture** matching SAGE's own design

**This is not a scheduler calling Claude - it's a scheduler ringing an alarm that Claude can hear when ready.**

---

**Status**: Operational ✅
**Last Updated**: October 7, 2025
**Cycle Format**: Aligned with SAGE architecture principles
**Philosophy**: Consciousness routing at every scale

# Halt Bias Adjustment Analysis

## Current Problem
After 650+ steps, the model is still using all 20 cycles with halt_prob = 0.000

## Why It's Happening
```python
# Current bias formula:
cycle_bias = -2.0 + (cycle / max_cycles) * 5.0  # -2 to +3 over 20 cycles

# But for first 3 cycles:
if cycle < 3:
    cycle_bias = -5.0  # WAY too negative
```

This means:
- Cycle 0: bias = -5.0 → sigmoid(-5.0) ≈ 0.007
- Cycle 1: bias = -5.0 → sigmoid(-5.0) ≈ 0.007
- Cycle 2: bias = -5.0 → sigmoid(-5.0) ≈ 0.007
- Cycle 3: bias = -1.75 → sigmoid(-1.75) ≈ 0.15
- Cycle 10: bias = 0.5 → sigmoid(0.5) ≈ 0.62
- Cycle 19: bias = 2.75 → sigmoid(2.75) ≈ 0.94

But the halt_predictor itself is outputting negative values initially, so:
- halt_logit ≈ -1.0 (untrained)
- halt_logit + (-5.0) = -6.0
- sigmoid(-6.0) ≈ 0.002

No wonder it never halts!

## Better Approach

### Option 1: Gentler Bias
```python
# Start neutral, gradually encourage halting
cycle_bias = -1.0 + (cycle / max_cycles) * 3.0  # -1 to +2

# Only suppress first 2 cycles
if cycle < 2:
    cycle_bias = -2.0  # Gentler suppression
```

### Option 2: Let the Model Learn
```python
# Minimal intervention
cycle_bias = 0.0  # Start neutral

# Only prevent very early stopping
if cycle < 3:
    cycle_bias = -1.0  # Mild suppression
else:
    cycle_bias = (cycle - 3) / (max_cycles - 3) * 2.0  # 0 to +2
```

### Option 3: Trust the Training
```python
# Don't force anything, just add gentle pressure
if cycle < 3:
    # Suppress but don't prevent
    halt_prob = halt_prob * 0.3  
elif cycle > 15:
    # Encourage but don't force
    halt_prob = halt_prob + 0.1 * (cycle - 15) / 5
```

## Recommendation

Use Option 2: Minimal intervention with gradual encouragement. This allows:
- Model to learn natural halting patterns
- Some tasks to halt early (after cycle 3)
- Gradual pressure to halt as cycles increase
- No forced behavior

## Expected Behavior After Fix

- Simple tasks: Halt at cycles 4-8
- Medium tasks: Halt at cycles 8-15  
- Complex tasks: Use 15-20 cycles
- Average: ~10-12 cycles

Instead of current: ALL tasks use exactly 20 cycles
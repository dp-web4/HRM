# Visual + Color Learning Solver Analysis

## What We Built

Combined visual pattern matching with color effectiveness learning:
- **Visual memory**: Store/compare frame snapshots, track similarity to initial/goal states
- **Color learning**: Two-phase EXPLORE→EXPLOIT, learn which colors cause changes
- **Integration**: Update driver preferences dynamically based on color effectiveness

## Test Results (lp85)

### Visual Solver (500 steps, single run)
- Level: 0/8 (stuck at level 0)
- Color 8: 100% effective (66/66 changes)
- Issue: Deterministic clicking of effective color without spatial strategy

### Smart Scorer (3 runs × 300 steps)
- Level: 1/8
- Color 8: Likely also ~100% effective
- Uses random clicking with volume

### Historical (mcnugget 4b94fb38)
- Level: 4/8 achieved
- Strategy: Multiple runs (10+ runs) with randomness
- L1@60, L2@99, L3@117, L4@212 steps

## Key Insight

**Color effectiveness alone ≠ puzzle solving**

Knowing Color 8 causes changes doesn't mean we're clicking the RIGHT Color 8 cells in the RIGHT sequence. lp85 is a rotation puzzle requiring specific button sequences.

## Why Visual Solver Struggles

1. **Over-determinism**: Once Color 8 learned as 100% effective, solver becomes too focused
2. **No spatial awareness**: Doesn't track WHICH Color 8 cells (buttons) matter
3. **No goal comparison**: Doesn't compare current state to goal state visually
4. **Single-run**: No volume/randomness to stumble upon correct sequences

## Possible Enhancements

### Option 1: Add Randomness (Quick)
- Keep random cell selection even for preferred colors
- Multi-run with different seeds
- Match smart_scorer's volume approach

### Option 2: Spatial Sequence Learning (Medium)
- Track which specific (x,y) clicks led to level ups
- Build button→outcome mapping
- Replay winning sequences

### Option 3: Goal-Directed Search (Advanced)
- Extract goal state from game metadata
- Compute visual similarity to GOAL (not just initial)
- Click colors that move closer to goal state
- Requires understanding game's goal representation

### Option 4: Hybrid (Best)
- Goal-directed search with spatial tracking
- Visual memory of progress states
- Sequence replay when near-goal states detected

## Recommendation

Start with **Option 1** (randomness) to validate the infrastructure works, then add **Option 3** (goal-awareness) for strategic improvement.

The visual memory foundation is solid - we just need better action selection strategy.

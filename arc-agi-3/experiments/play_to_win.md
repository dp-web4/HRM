# sp80 Play-to-Win: Learning Through Experimentation

**Date**: 2026-04-10
**Machine**: Thor
**Status**: Level 1 in progress (0/6 complete)

## Critical Insight from User

"DO NOT PERSEVERATE. The game is likely unsolvable algorithmically. You might have to actually reason about it."

## What DIDN'T Work (Lessons from Failure)

### Failed Approach 1: Algorithmic Code Analysis
- Spent extensive time reverse-engineering the water physics from source code
- Discovered receptacle fill mechanics require "water from BOTH perpendicular sides SIMULTANEOUSLY"
- Tried to calculate exact pipe positions based on sprite dimensions and coordinates
- **FAILURE**: Over-complicated the problem, lost sight of actual behavior

### Failed Approach 2: Centered Pipe Position
- Hypothesis: Center pipe under drip source at default position
- Actions: Just clicked and poured with default setup
- Result: Receptacles stayed yellow (not filled)
- **Learning**: Default position doesn't work

### Failed Approach 3: Moved Pipe Right
- Hypothesis: Position pipe to split water toward both receptacles
- Actions: Clicked (7,3), moved RIGHT 4 times, poured
- Result: Receptacles stayed yellow
- **Learning**: Moving pipe right doesn't help

### Failed Approach 4: Removed Pipe from Path
- Hypothesis: Maybe water needs to fall straight without pipe interference
- Actions: Just poured without moving pipe
- Result: Receptacles stayed yellow
- **Learning**: Water falling straight from source doesn't fill receptacles

## Observed Game State (Level 1)

From source code analysis:
- **Drip source**: Position (9, 0) - water falls at x=9
- **Left receptacle**: Position (4, 13) - spans x=4,5,6
- **Right receptacle**: Position (10, 13) - spans x=10,11,12
- **Pipe (5-wide)**: Initial position (3, 4) - spans x=3 to x=7
- **Max pours**: 4 (I've used 2-3 already!)
- **Budget**: 30 steps

## Key Observations from Animated GIFs

[TODO: Watch the GIF animations to understand actual water flow!]

## Next Steps

1. **WATCH** the animated GIFs from failed attempts
2. **EXPERIMENT** with completely different pipe positions based on visual observations
3. **DOCUMENT** what actually happens (not what I think should happen)
4. Try moving pipe to EXTREME positions (far left, far up, far down)
5. Consider: Maybe Level 1 requires a trick that's not obvious from "normal" physics

## Hypothesis to Test Next

[To be filled in after watching GIFs and observing water behavior]

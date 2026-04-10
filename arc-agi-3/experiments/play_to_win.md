# sp80 Play-to-Win: Progress Log

**Date**: 2026-04-10
**Status**: L1-L3 solved live, L4-L6 in progress

## Solutions Verified Live Against SDK

### Level 1 (rot=0, grid=16x16, 30 steps)
- **Config**: Pipe at (6,4) — 3 rights from default
- **Actions**: CLICK(7,3) RIGHT RIGHT RIGHT SELECT
- **5 actions, verified live**

### Level 2 (rot=180, grid=16x16, 45 steps)
- **Config**: pipe0 stays at (6,6), pipe1→(4,3), pipe2→(4,11)
- **Actions**: CLICK(33,25) RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN CLICK(13,17) RIGHT*7 SELECT
- **18 actions, verified live**

### Level 3 (rot=180, grid=16x16, 100 steps)
- **Config**: p0→(11,3), p1→(3,7), p2→(0,5), p3→(9,9)
- **Actions**: p2 LEFT → p1 LEFT 5 → p0 RIGHT 10 UP 5 → p3 LEFT UP → SELECT
- **~28 display actions, verified live**
- **Key insight**: Grid boundary constraint — 4w pipe max x = gw-4 = 12, NOT 14.

## L4-L6 Analysis (Unsolved)

### The Fundamental L4 Challenge
- Grid 20x20, rot=0, 120 steps
- 2 sources: x=7 from top, x=5 from fixed pipe (adbrqflmwi at (2,9))
- 4 cups at x=3, x=9, x=13, x=17 at y=17
- The fixed pipe p0 spans x=2-8 at y=9. ANY water entering it exits at x=1 and x=9.
- x=1 stream → always reaches danger zone at y=19 unless caught by another pipe
- Splash from receptacle edges creates stray water that reaches danger

### What Didn't Work (L4)
1. Random search: 3.5M+ random pipe configurations — found 4/4 fills but ALWAYS with danger
2. Analytical routing: every pipe chain eventually produces uncaught stray water
3. Negative-x pipe positions (x=-1, -2) for catching x=1 stream — still danger from other sources
4. The core constraint: 4 movable pipes + 2 sources + fixed pipe producing x=1 + splash mechanics → not enough pipes to catch ALL stray water

### Hypothesis for L4 Solution
- May require a pipe configuration where water streams merge before hitting receptacles, avoiding splash
- Or: the simulation doesn't perfectly model some mechanic (e.g., splash blocking when water already exists at splash position)
- Or: there's a creative use of pipe overlap or pipe-pipe cascading that creates safe routing

### Next Steps
1. Try interactive exploration: position pipes, pour, observe actual game behavior
2. Compare game behavior with simulation to find discrepancies
3. Try L5 and L6 — might be easier than L4 (L5 has L-pipes which add routing options)

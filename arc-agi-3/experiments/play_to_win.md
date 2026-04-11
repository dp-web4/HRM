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

### L4 SOLVED — The Breakthrough

**Root cause of failure:** I assumed `adbrqflmwi` (p0, the 7-wide pipe with drip source) was FIXED because it lacks the `sys_click` tag. But the game's click handler uses `rxjmwfcjyw()` which returns ALL `ksmzdcblcz`-tagged sprites regardless of `sys_click`. **p0 IS movable!** That gives us 5 pipes, not 4.

**Second bug:** The water flow simulator didn't update drip source positions when pipes with `syaipsfndp` tag were moved. Fixed by recomputing drip positions in `simulate_pour()`.

**L4 solution (verified live):**
- p0(7w): (2,9)→(10,10). Drip moves to x=13→cup3. Pipe exits 9(cup2)+17(cup1).
- p2(5w): (5,5)→(4,5). Catches source x=7. Exits 3(cup0)+9.
- p4(4w): (14,10)→(9,8). Catches x=9 cascade. Exits 8+13.
- p1(5w): (12,5)→(4,14). Catches x=8. Exits 3(cup0)+9(cup2). Safe!
- p3(4w): (12,13)→(0,3). Out of the way.
- **All 4 cups filled, no danger!**

**Key lessons:**
1. Don't assume sprites are immovable just because they lack a UI tag. Read the actual click handler code.
2. When sprites have multiple tags (pipe + drip source), moving them moves ALL their properties.
3. The 7w pipe at x=10 is magical: exits at x=9 (cup2) AND x=17 (cup1), drip at x=13 (cup3).

### Next Steps
1. L5 has L-pipes (deflectors) — new mechanic to model
2. L6 has 3 danger zones (sides + bottom) — more constrained routing

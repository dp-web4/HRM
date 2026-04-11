#!/usr/bin/env python3
"""Verify L6 solution using the physics model (no game engine needed).

Traces each step with full state, checking for death and win conditions.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

WALLS = set()
UP_SPIKES = set()
DN_SPIKES = set()
ALL_SPIKES = set()
GEM = (3, 25)

grid_rows = {
    0:  "..WWWWWWWWWWW....",
    1:  "..WWWWWWWWWWW....",
    2:  "..WWWWWWWWWWW....",
    3:  "..WWWWWWWWWWW....",
    4:  "..WWWWWWWOW^W....",
    5:  "..WWWWWWWOW.W....",
    6:  "..WWWWWWWO..W....",
    7:  "..WWWWWWWO..W....",
    8:  "..WW.O^.^OW.W....",
    9:  "..WW.OO.BOW.W....",
    10: "..WW.O.B.OW.W....",
    11: "..WW.O.v.vW.W....",
    12: "..WW.WWWWWW.W....",
    13: "..WW.WOOO.W.W....",
    14: "..WW.WOOO.W.W....",
    15: "..WW..OOO...W....",
    16: "..WW..OOO...W....",
    17: "..WW.vOOOv..W....",
    18: "..WWWWWWWW..W....",
    19: "..WW....OW..W....",
    20: "..WW........W....",
    21: "..WW....OW..W....",
    22: "..WWWWWWWW..W....",
    23: "..WW........W....",
    24: "..WW......v.W....",
    25: "..WW.*.WW.W.W....",
    26: "..WW...WW...W....",
    27: "..WW...WWWWWW....",
    28: "..WWWWWWWWWWW....",
}

for y, row in grid_rows.items():
    for i, ch in enumerate(row):
        x = i - 2
        if ch == 'W':
            WALLS.add((x, y))
        elif ch == '^':
            UP_SPIKES.add((x, y))
            ALL_SPIKES.add((x, y))
        elif ch == 'v':
            DN_SPIKES.add((x, y))
            ALL_SPIKES.add((x, y))

OB_BLOCKS = [
    (7, 4), (7, 5), (7, 6), (7, 7),
    (3, 8), (7, 8),
    (3, 9), (4, 9), (6, 9), (7, 9),
    (3, 10), (5, 10), (7, 10),
    (3, 11),
    (4, 13), (5, 13), (6, 13),
    (4, 14), (5, 14), (6, 14),
    (4, 15), (5, 15), (6, 15),
    (4, 16), (5, 16), (6, 16),
    (4, 17), (5, 17), (6, 17),
    (6, 19),
    (6, 21),
]
OB_SET = set(OB_BLOCKS)

# Track which blocks are B (solid). Initially (6,9) and (5,10) are B.
STATIC_SOLID = WALLS | ALL_SPIKES


class State:
    def __init__(self):
        self.px, self.py = 3, 19
        self.grav_up = True
        self.b_blocks = {(6, 9), (5, 10)}  # Start: these are B (solid)
        self.dead = False
        self.won = False

    def is_solid(self, x, y):
        if (x, y) in STATIC_SOLID:
            return True
        if (x, y) in OB_SET:
            return (x, y) in self.b_blocks
        return False

    def is_dead_check(self, x, y, grav_up):
        if grav_up:
            return (x, y - 1) in UP_SPIKES
        else:
            return (x, y + 1) in DN_SPIKES

    def fall(self, x, y):
        """Fall from (x,y) in current gravity. Returns (fx, fy, won, dead)."""
        dy = -1 if self.grav_up else 1
        lx, ly = x, y
        cx, cy = x, y + dy
        while True:
            if (cx, cy) == GEM:
                return cx, cy, True, False
            if self.is_solid(cx, cy):
                return lx, ly, False, self.is_dead_check(lx, ly, self.grav_up)
            if cy < -5 or cy > 35:
                return lx, ly, False, False
            lx, ly = cx, cy
            cy += dy

    def move_lr(self, dx):
        """Move left (-1) or right (+1)."""
        nx, ny = self.px + dx, self.py
        if (nx, ny) == GEM:
            self.px, self.py = nx, ny
            self.won = True
            return
        if self.is_solid(nx, ny):
            return  # blocked
        fx, fy, won, dead = self.fall(nx, ny)
        if won:
            self.px, self.py = fx, fy
            self.won = True
            return
        if dead:
            self.px, self.py = fx, fy
            self.dead = True
            return
        self.px, self.py = fx, fy

    def flip_gravity(self):
        """Flip gravity and apply fall."""
        self.grav_up = not self.grav_up
        dy = -1 if self.grav_up else 1
        if self.is_solid(self.px, self.py + dy):
            if self.is_dead_check(self.px, self.py, self.grav_up):
                self.dead = True
            return
        fx, fy, won, dead = self.fall(self.px, self.py)
        if won:
            self.px, self.py = fx, fy
            self.won = True
            return
        if dead:
            self.px, self.py = fx, fy
            self.dead = True
            return
        self.px, self.py = fx, fy

    def toggle_ob(self, bx, by):
        """Toggle O↔B block at (bx, by)."""
        pos = (bx, by)
        if pos not in OB_SET:
            print(f"  ERROR: ({bx},{by}) is not an OB block!")
            return
        was_solid = pos in self.b_blocks
        if was_solid:
            self.b_blocks.discard(pos)
            # Check if this was our floor/ceiling
            dy = -1 if self.grav_up else 1
            if pos == (self.px, self.py + dy):
                # Floor/ceiling removed! Player falls
                fx, fy, won, dead = self.fall(self.px, self.py)
                if won:
                    self.px, self.py = fx, fy
                    self.won = True
                    return
                if dead:
                    self.px, self.py = fx, fy
                    self.dead = True
                    return
                self.px, self.py = fx, fy
        else:
            self.b_blocks.add(pos)
            # If player is at this position (inside new solid), that's bad
            if pos == (self.px, self.py):
                print(f"  WARNING: Player trapped in solid block at ({bx},{by})!")

    def __str__(self):
        g = "UP" if self.grav_up else "DN"
        b = sorted(self.b_blocks)
        status = " DEAD!" if self.dead else " WIN!" if self.won else ""
        return f"({self.px:2d},{self.py:2d}) grav={g} B={b}{status}"


# Define solution
R = ('R',)
L = ('L',)
def G(gy): return ('G', 0, gy)  # G flip at x=0, y=gy
def T(bx, by): return ('T', bx, by)  # Toggle OB block

solution = [
    # Phase 1: Start to platform area (12 steps)
    R,                  # 1. → (4,19)
    R,                  # 2. → (5,19)
    R,                  # 3. → (6,19) [O transparent]
    T(6, 21),           # 4. toggle (6,21) O→B
    G(19),              # 5. flip DOWN → (6,20) [floor B(6,21)]
    R,                  # 6. → (7,20) [floor wall(7,21)]
    G(20),              # 7. flip UP → stays (7,20)
    R,                  # 8. → (8,15) via fall UP
    L,                  # 9. → (7,13) via fall UP
    L,                  # 10. → (6,13) O
    L,                  # 11. → (5,13) O
    L,                  # 12. → (4,13) O

    # Phase 2: Platform to x=2 shaft (5 steps)
    G(13),              # 13. flip DOWN → (4,17)
    T(4, 15),           # 14. toggle (4,15) O→B
    G(17),              # 15. flip UP → (4,16) [ceiling B(4,15)]
    L,                  # 16. → (3,15) via fall UP [ceiling wall(3,14)]
    L,                  # 17. → (2,8) via fall UP [ceiling wall(2,7)]

    # Phase 3: Navigate y=8-11 to reach x=7 (14 steps)
    R,                  # 18. → (3,8) O [ceiling wall(3,7)]
    G(8),               # 19. flip DOWN → (3,11) [floor wall(3,12)]
    R,                  # 20. → (4,11) [floor wall(4,12)]
    T(4, 9),            # 21. toggle (4,9) O→B
    G(11),              # 22. flip UP → (4,10) [ceiling B(4,9)]
    T(5, 10),           # 23. toggle (5,10) B→O
    R,                  # 24. → (5,8) via fall UP [ceiling wall(5,7)]
    T(5, 10),           # 25. toggle (5,10) O→B
    G(8),               # 26. flip DOWN → (5,9) [floor B(5,10)]
    T(6, 9),            # 27. toggle (6,9) B→O
    R,                  # 28. → (6,11) via fall DOWN [floor wall(6,12)]
    T(6, 9),            # 29. toggle (6,9) O→B
    G(11),              # 30. flip UP → (6,10) [ceiling B(6,9)]
    R,                  # 31. → (7,4) via fall UP through O column [ceiling wall(7,3)]

    # Phase 4: Through x=7 to x=9 (4 steps)
    T(7, 8),            # 32. toggle (7,8) O→B
    G(5),               # 33. flip DOWN → (7,7) [floor B(7,8)]
    R,                  # 34. → (8,7) [floor wall(8,8)]
    R,                  # 35. → (9,26) via fall DOWN [floor wall(9,27)]

    # Phase 5: Navigate to gem (8 steps)
    L,                  # 36. → (8,26) [floor wall(8,27)]
    L,                  # 37. → (7,26) [floor wall(7,27)]
    G(26),              # 38. flip UP → (7,23) [ceiling wall(7,22)]
    L,                  # 39. → (6,23)
    L,                  # 40. → (5,23)
    L,                  # 41. → (4,23)
    L,                  # 42. → (3,23)
    G(23),              # 43. flip DOWN → gem (3,25)!
]

# Execute and verify
state = State()
print(f"L6 Solution Verification ({len(solution)} steps)")
print(f"Start: {state}")
print()

for i, step in enumerate(solution):
    old = f"({state.px:2d},{state.py:2d})"
    if step[0] == 'R':
        state.move_lr(1)
        action = "R"
    elif step[0] == 'L':
        state.move_lr(-1)
        action = "L"
    elif step[0] == 'G':
        state.flip_gravity()
        action = f"G({step[2]})"
    elif step[0] == 'T':
        state.toggle_ob(step[1], step[2])
        bstate = "B→O" if (step[1], step[2]) not in state.b_blocks else "O→B"
        action = f"T({step[1]},{step[2]}) {bstate}"
    else:
        print(f"  Unknown action: {step}")
        continue

    new = f"({state.px:2d},{state.py:2d})"
    grav = "UP" if state.grav_up else "DN"
    print(f"  {i+1:2d}. {action:20s}: {old} → {new}  grav={grav}")

    if state.dead:
        print(f"\n  DEAD at step {i+1}! Player at ({state.px},{state.py})")
        sys.exit(1)
    if state.won:
        print(f"\n  WIN at step {i+1}! Reached gem at ({state.px},{state.py})")
        print(f"  Solution: {len(solution)} steps (baseline: 86)")
        break
else:
    print(f"\n  Solution incomplete! Player at ({state.px},{state.py})")
    print(f"  B blocks: {sorted(state.b_blocks)}")

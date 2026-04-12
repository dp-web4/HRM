"""
re86 Level 8 Solution
=====================
Shape-matching puzzle: position 2 deformable pieces to match target pixels.

Target:
  P0: 16x10 hollow rect, color 6, at position (6,45)
  P1: 7x19 hollow rect, color 11, at position (9,39)

Key challenge:
  Paint zones c6 at (5,20) and c11 at (5,5) are above a nearly-complete barrier
  of paint zones at rows 28-32. The gap in the barrier (cols 44-49) is only 6 cols wide.

Solution strategy (discovered through analysis):
  1. Deform piece to 16x10 using wall at (34,55) below barrier
  2. Cross barrier going UP (piece gets repainted but shape preserved)
  3. At wall (58,1) above barrier: dy-deform 16x10 -> 19x7 -> 22x4
  4. The 22x4 piece (only 4 rows tall) fits in the safe corridor at y=15
     (rows 15-18) between blocking zones c9 at rows 8-12 and c12 at rows 21-25
  5. Navigate to paint zone, paint correct color
  6. Return at y=15 (free movement corridor) to x=36
  7. Go UP to y=3, then RIGHT to dx-deform at wall (58,1): 22x4 -> ... -> 4x22
  8. Go through gap (cols 44-49) back below barrier
  9. Reshape to target dimensions using wall at (34,55)
  10. Position at target coordinates

Total steps: 190 (budget: 400)
"""

import sys
import os

# Add game module to path
script_dir = os.path.dirname(os.path.abspath(__file__))
game_dir = os.path.join(script_dir, 'environment_files', 're86', '4e57566e')
sys.path.insert(0, game_dir)

from re86 import Re86, GameAction
from arcengine import ActionInput

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
SEL = GameAction.ACTION5


def solve_l8():
    game = Re86()
    game.set_level(7)  # L8 is index 7

    steps = 0

    def act(action):
        nonlocal steps
        steps += 1
        game.perform_action(ActionInput(id=action))

    def repeat(action, n):
        for _ in range(n):
            act(action)

    p0 = game.current_level.get_sprites_by_tag('vfaeucgcyr')[0]
    p1 = game.current_level.get_sprites_by_tag('vfaeucgcyr')[1]

    def piece_info(p):
        cp = p.pixels[(p.pixels != 0) & (p.pixels != -1)]
        c = int(cp[0]) if len(cp) > 0 else -1
        return f"({p.x},{p.y}) {p.width}x{p.height} c={c}"

    # ========================================================
    # P0 ROUTE: 13x13 c=12 at (45,42) -> 16x10 c=6 at (6,45)
    # ========================================================

    # Phase 1: Deform 13x13 -> 16x10 via wall at (34,55)
    # Move LEFT 3 to x=36, then DOWN to hit wall
    repeat(LEFT, 3)   # (36,42) 13x13
    act(DOWN)          # wall collision -> (33,45) 16x10
    act(UP)            # move away from wall -> (33,42)

    # Phase 2: Cross barrier going UP to y=6
    repeat(UP, 12)     # -> (33,6) 16x10

    # Phase 3: Move RIGHT to x=45 (safe: piece rows 6-15, wall rows 1-5 no overlap)
    repeat(RIGHT, 4)   # -> (45,6) 16x10

    # Phase 4: dy-deform at wall (58,1): 16x10 -> 19x7 -> 22x4
    act(UP)            # wall collision, bounce back -> (45,6) 19x7
    act(UP)            # wall collision, bounce back -> (42,6) 22x4

    # Phase 5: Navigate DOWN to y=18 for c6 zone overlap
    # c6 zone at (5,20) rows 20-24. 22x4 at y=18: rows 18-21, overlap at 20-21
    repeat(DOWN, 4)    # -> (42,18) 22x4

    # Phase 6: Move LEFT to x=6 (overlap c6 zone at cols 5-9)
    repeat(LEFT, 12)   # -> (6,18) 22x4 c=6 (painted by c6 zone!)

    # Phase 7: Move UP to y=15 (safe corridor: rows 15-18, between c9/c12 zones)
    act(UP)            # -> (6,15) 22x4 c=6

    # Phase 8: Move RIGHT to x=36 (free movement at y=15!)
    repeat(RIGHT, 10)  # -> (36,15) 22x4 c=6

    # Phase 9: Move UP to y=3 (dx-deform altitude)
    repeat(UP, 4)      # -> (36,3) 22x4 c=6

    # Phase 10: Move RIGHT for dx-deforms at wall (58,1)
    # 22x4 -> 19x7 -> 16x10 -> 13x13 -> 10x16 -> 7x19 -> 4x22
    repeat(RIGHT, 6)   # -> (54,-6) 4x22 c=6

    # Phase 11: Move LEFT to gap position x=45
    repeat(LEFT, 3)    # -> (45,-6) 4x22 c=6

    # Phase 12: Move DOWN through gap (cols 44-49, no barrier zones)
    repeat(DOWN, 15)   # -> (45,39) 4x22 c=6

    # Phase 13: Reshape 4x22 -> 16x10 via wall at (34,55)
    # Move UP 2 for clearance, LEFT 3 to wall x, DOWN 4 for dy-deforms
    repeat(UP, 2)      # -> (45,33)
    repeat(LEFT, 3)    # -> (36,33)
    repeat(DOWN, 4)    # 4x22 -> 7x19 -> 10x16 -> 13x13 -> 16x10 at (30,45)

    # Phase 14: Move LEFT to target x=6
    repeat(LEFT, 8)    # -> (6,45) 16x10 c=6

    assert p0.x == 6 and p0.y == 45, f"P0 position wrong: {piece_info(p0)}"
    assert p0.width == 16 and p0.height == 10, f"P0 shape wrong: {piece_info(p0)}"
    print(f"P0 done [{steps} steps]: {piece_info(p0)}")

    # ========================================================
    # SWITCH TO P1
    # ========================================================
    act(SEL)

    # ========================================================
    # P1 ROUTE: 13x13 c=10 at (48,39) -> 7x19 c=11 at (9,39)
    # ========================================================

    # Phase 1: Deform 13x13 -> 16x10 via wall at (34,55)
    repeat(LEFT, 4)    # (36,39)
    repeat(DOWN, 2)    # wall collision -> (33,45) 16x10
    act(UP)            # move away -> (33,42)

    # Phase 2: Cross barrier going UP to y=6
    repeat(UP, 12)     # -> (33,6) 16x10

    # Phase 3: Move RIGHT to x=45
    repeat(RIGHT, 4)   # -> (45,6) 16x10

    # Phase 4: dy-deform at wall (58,1): 16x10 -> 19x7 -> 22x4
    act(UP)            # -> (45,6) 19x7
    act(UP)            # -> (42,6) 22x4

    # Phase 5: Move LEFT to x=6 (c11 zone at (5,5) rows 5-9)
    # 22x4 at y=6: rows 6-9. c11 overlap at 6-9.
    repeat(LEFT, 12)   # -> (6,6) 22x4 c=11 (painted!)

    # Phase 6: Move DOWN to y=15 (safe corridor)
    # c11 at rows 5-9: piece at y=9 (rows 9-12) no overlap. Safe!
    repeat(DOWN, 3)    # -> (6,15) 22x4 c=11

    # Phase 7: Move RIGHT to x=36 (free movement at y=15)
    repeat(RIGHT, 10)  # -> (36,15) 22x4 c=11

    # Phase 8: Move UP to y=3
    repeat(UP, 4)      # -> (36,3) 22x4 c=11

    # Phase 9: Move RIGHT for dx-deforms -> 4x22
    repeat(RIGHT, 6)   # -> (54,-6) 4x22 c=11

    # Phase 10: Move LEFT to gap x=45
    repeat(LEFT, 3)    # -> (45,-6) 4x22 c=11

    # Phase 11: Move DOWN through gap
    repeat(DOWN, 15)   # -> (45,39) 4x22 c=11

    # Phase 12: Reshape 4x22 -> 7x19 via wall at (34,55)
    # Only need 1 dy-deform: 4x22 -> 7x19
    repeat(UP, 2)      # -> (45,33)
    repeat(LEFT, 3)    # -> (36,33)
    act(DOWN)          # -> 7x19

    # Phase 13: Position at (9,39)
    # Current position after reshape: need to check and adjust
    x_diff = p1.x - 9
    y_diff = p1.y - 39
    if x_diff > 0:
        repeat(LEFT, x_diff // 3)
    elif x_diff < 0:
        repeat(RIGHT, (-x_diff) // 3)
    if y_diff > 0:
        repeat(UP, y_diff // 3)
    elif y_diff < 0:
        repeat(DOWN, (-y_diff) // 3)

    assert p1.x == 9 and p1.y == 39, f"P1 position wrong: {piece_info(p1)}"
    assert p1.width == 7 and p1.height == 19, f"P1 shape wrong: {piece_info(p1)}"
    print(f"P1 done [{steps} steps]: {piece_info(p1)}")

    # ========================================================
    # VERIFY WIN
    # ========================================================
    won = game.cdjxpfqest()
    print(f"\nTotal steps: {steps} / 400")
    print(f"Win: {won}")

    if won:
        print("\n=== LEVEL 8 SOLVED! ===")
        # Generate action sequence
        return True
    else:
        print("\nWin check failed!")
        return False


def generate_action_sequence():
    """Generate the compact action sequence for replay."""
    # Using (direction, count) tuples
    sequence = [
        # P0 route
        (LEFT, 3), (DOWN, 1), (UP, 1),         # 16x10 deform
        (UP, 12),                                # cross barrier
        (RIGHT, 4),                              # to gap x
        (UP, 2),                                 # 22x4 deform
        (DOWN, 4),                               # to paint y
        (LEFT, 12),                              # paint c6
        (UP, 1),                                 # safe corridor
        (RIGHT, 10),                             # to deform x
        (UP, 4),                                 # deform altitude
        (RIGHT, 6),                              # dx deforms -> 4x22
        (LEFT, 3),                               # to gap
        (DOWN, 15),                              # through gap
        (UP, 2), (LEFT, 3), (DOWN, 4),          # reshape -> 16x10
        (LEFT, 8),                               # to target x
        # SEL
        "SEL",
        # P1 route
        (LEFT, 4), (DOWN, 2), (UP, 1),          # 16x10 deform
        (UP, 12),                                # cross barrier
        (RIGHT, 4),                              # to gap x
        (UP, 2),                                 # 22x4 deform
        (LEFT, 12),                              # paint c11
        (DOWN, 3),                               # safe corridor
        (RIGHT, 10),                             # to deform x
        (UP, 4),                                 # deform altitude
        (RIGHT, 6),                              # dx deforms -> 4x22
        (LEFT, 3),                               # to gap
        (DOWN, 15),                              # through gap
        (UP, 2), (LEFT, 3), (DOWN, 1),          # reshape -> 7x19
        # positioning (computed dynamically)
    ]
    return sequence


if __name__ == "__main__":
    success = solve_l8()
    if not success:
        sys.exit(1)

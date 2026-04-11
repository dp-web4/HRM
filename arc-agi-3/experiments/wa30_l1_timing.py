#!/usr/bin/env python3
"""wa30 L1: Exhaustive search over approach timing and delivery parameters.

The 5/5 c=1 result is off by exactly 1 step. Try all reasonable approach
orderings (RIGHT/DOWN interleaving) and delivery targets to find a timing
that lets the RED drop 1 step earlier.
"""
import sys, os, time, random
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
from itertools import combinations

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5
)
AN = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R', ACT5: '5'}

L0_SOL = [UP,UP,LEFT,UP,UP,UP,LEFT,LEFT,ACT5,RIGHT,RIGHT,RIGHT,ACT5,UP,RIGHT,RIGHT,
          ACT5,DOWN,LEFT,LEFT,ACT5,DOWN,ACT5,UP,UP,ACT5]


def test_sequence(seq, idle_action=UP):
    """Run a sequence and return (won, at_slot, carried)."""
    arcade = Arcade()
    env = arcade.make('wa30-ee6fef47')
    fd = env.reset()
    for a in L0_SOL:
        fd = env.step(a)
    game = env._game

    total_steps = game.kuncbnslnm.current_steps
    active = len(seq)
    idle = total_steps - active
    if idle < 0:
        return False, 0, 0

    full = list(seq) + [idle_action] * idle
    for i, a in enumerate(full):
        fd = env.step(a)
        if fd.state.name == 'WIN':
            return True, 5, 0

    game = env._game
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = set((x, y) for (x, y) in game.wyzquhjerd)
    at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
    carried = len(game.zmqreragji)
    return False, at_slot, carried


# Strategy: generate all interleavings of 9 RIGHTs and 3 DOWNs for the approach phase.
# After approach, player is at (48,20) regardless. Then ACT5 + UP*2 + carry + drop.
#
# The interleaving changes WHEN the player is at each position during the approach,
# which changes what the RED sees (player as obstacle) and thus RED's pathfinding.

t0 = time.time()
n_tested = 0
best_score = 0
wins = []

# Generate all unique interleavings of 9R + 3D
# Choose 3 positions (out of 12) for the DOWNs
print("Testing all interleavings of approach phase (9R+3D)...")
print(f"Total combos: C(12,3) * 6 slots * 4 idle = {220*6*4}")

for down_positions in combinations(range(12), 3):
    # Build approach sequence
    approach = []
    d_idx = 0
    r_idx = 0
    for i in range(12):
        if i in down_positions:
            approach.append(DOWN)
        else:
            approach.append(RIGHT)

    # After approach: at (48,20), facing depends on last action
    # If last is RIGHT, facing RIGHT=(52,20), NOT the goal!
    # If last is DOWN, facing DOWN=(48,24), which IS the goal.
    # Need facing DOWN for ACT5 to work!
    if approach[-1] != DOWN:
        # Add blocked DOWN to set facing
        approach.append(DOWN)  # +1 step, blocked at (48,24)

    for n_left, n_down in [(9,3), (9,4), (9,5), (8,3), (8,4), (8,5)]:
        carry = [UP]*2 + [LEFT]*n_left + [DOWN]*n_down + [ACT5]
        seq = approach + [ACT5] + carry

        for idle_act in [UP, RIGHT]:  # Only test best idle actions
            n_tested += 1
            won, at_slot, carried = test_sequence(seq, idle_act)

            score = at_slot * 10 - carried
            if won:
                sol_str = ''.join(AN[a] for a in seq)
                idle_str = AN[idle_act]
                print(f"\n  WIN! seq={sol_str} idle={idle_str} ({len(seq)} active)")
                wins.append((seq, idle_act))

            if score > best_score:
                best_score = score
                sol_str = ''.join(AN[a] for a in seq)
                idle_str = AN[idle_act]
                elapsed = time.time() - t0
                print(f"  [{n_tested}] New best: {at_slot}/5 c={carried} seq={sol_str} idle={idle_str} ({elapsed:.1f}s)")

            if n_tested % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{n_tested}] tested, best={best_score}, {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"\nDone! Tested {n_tested}, {elapsed:.1f}s")
if wins:
    print(f"WINS: {len(wins)}")
    for seq, idle in wins:
        print(f"  {''.join(AN[a] for a in seq)} idle={''.join(AN[idle])}")
else:
    print(f"No wins. Best score: {best_score}")

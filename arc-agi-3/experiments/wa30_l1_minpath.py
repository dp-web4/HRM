#!/usr/bin/env python3
"""wa30 L1: Try minimum-length player paths to maximize idle time for AI."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5
)
AN = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R', ACT5: '5'}

arcade = Arcade()
env = arcade.make('wa30-ee6fef47')

L0_SOL = [UP,UP,LEFT,UP,UP,UP,LEFT,LEFT,ACT5,RIGHT,RIGHT,RIGHT,ACT5,UP,RIGHT,RIGHT,
          ACT5,DOWN,LEFT,LEFT,ACT5,DOWN,ACT5,UP,UP,ACT5]

def reset_to_l1():
    fd = env.reset()
    for a in L0_SOL:
        fd = env.step(a)
    return fd

def run_and_check(name, seq, idle_action=UP, verbose=False):
    fd = reset_to_l1()
    game = env._game
    total_steps = game.kuncbnslnm.current_steps
    active = len(seq)
    idle = total_steps - active
    if idle < 0:
        print(f"{name} ({active}a/{idle}i): TOO LONG")
        return False

    full = list(seq) + [idle_action] * idle
    best_score = 0
    won = False

    for i, a in enumerate(full):
        fd = env.step(a)
        game = env._game
        step = i + 1

        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        slots = set((x, y) for (x, y) in game.wyzquhjerd)
        at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
        carried = len(game.zmqreragji)

        if fd.state.name == 'WIN':
            print(f"{name} ({active}a/{idle}i): WIN at step {step}!")
            return True

        score = at_slot * 10 - carried
        if score > best_score:
            best_score = score

        if verbose and (step <= active or at_slot >= 4 or step >= total_steps - 3):
            reds = game.current_level.get_sprites_by_tag("kdweefinfi")
            r_info = ""
            for r in reds:
                ci = f"c({game.nsevyuople[r].x},{game.nsevyuople[r].y})" if r in game.nsevyuople else ""
                r_info += f" r({r.x},{r.y}){ci}"
            not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]
            print(f"  {step:2d} [{at_slot}/5] c={carried}{r_info} not={not_at}")

    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = set((x, y) for (x, y) in game.wyzquhjerd)
    at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
    carried = len(game.zmqreragji)
    not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]
    print(f"{name} ({active}a/{idle}i): {at_slot}/5 c={carried} not={not_at}")
    return False

# Goal at (48,24). Player starts at (12,8). Slots at (12-16, 28-36).
#
# Strategy: go RIGHT to x=48, DOWN to y=20, face goal at (48,24), pickup,
# carry to nearest slot with minimum moves.
#
# After pickup at (48,20) with carry offset (0,4) [carried below]:
#   carried_pos = (player_x, player_y + 4)
#   For carried at (16,28): player at (16,24). LEFT*8, DOWN*1. 10 steps carry.
#   For carried at (12,28): player at (12,24). LEFT*9, DOWN*1. 11 steps carry.
#   For carried at (16,32): player at (16,28). LEFT*8, DOWN*2. 11 steps carry.
#   For carried at (12,32): player at (12,28). LEFT*9, DOWN*2. 12 steps carry.
#   For carried at (16,36): player at (16,32). LEFT*8, DOWN*3. 12 steps carry.
#   For carried at (12,36): player at (12,32). LEFT*9, DOWN*3. 13 steps carry.

# All paths start with: RIGHT*9 + DOWN*3 + ACT5 = 13 steps for approach+pickup

base = [RIGHT]*9 + [DOWN]*3 + [ACT5]  # 13 steps

print("=== Systematic minimum-length paths ===")
print("Player at (12,8) → goal at (48,24)")
print("Approach: RIGHT*9 + DOWN*3 + ACT5 = 13 steps")
print()

# Try all slot destinations
targets = [
    ("16,28", [LEFT]*8 + [DOWN]*1 + [ACT5]),   # 10
    ("12,28", [LEFT]*9 + [DOWN]*1 + [ACT5]),   # 11
    ("16,32", [LEFT]*8 + [DOWN]*2 + [ACT5]),   # 11
    ("12,32", [LEFT]*9 + [DOWN]*2 + [ACT5]),   # 12
    ("16,36", [LEFT]*8 + [DOWN]*3 + [ACT5]),   # 12
    ("12,36", [LEFT]*9 + [DOWN]*3 + [ACT5]),   # 13
]

for slot_name, carry_seq in targets:
    seq = base + carry_seq
    name = f"to_{slot_name.replace(',','_')}"
    run_and_check(name, seq, verbose=True)
    print()

# Also try carry with UP first (offset changes direction)
# After pickup facing DOWN, UP*1 puts player at (48,16), carried at (48,20)
# Still offset (0,4). LEFT then DOWN.
# UP*1 + LEFT*8 + DOWN*2 + ACT5 = 12, carried at (16,28). Total 25.
# UP*1 + LEFT*9 + DOWN*2 + ACT5 = 13, carried at (12,28). Total 26.
# UP*2 + LEFT*8 + DOWN*3 + ACT5 = 14, carried at (16,28). Total 27.

print("=== With UP before LEFT (possibly avoids AI collision) ===")
up_targets = [
    ("u1_16_28", [UP]*1 + [LEFT]*8 + [DOWN]*2 + [ACT5]),   # 12
    ("u1_12_28", [UP]*1 + [LEFT]*9 + [DOWN]*2 + [ACT5]),   # 13
    ("u1_16_32", [UP]*1 + [LEFT]*8 + [DOWN]*3 + [ACT5]),   # 13
    ("u1_12_32", [UP]*1 + [LEFT]*9 + [DOWN]*3 + [ACT5]),   # 14
    ("u2_16_32", [UP]*2 + [LEFT]*8 + [DOWN]*3 + [ACT5]),   # 14
    ("u2_12_32", [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5]),   # 15
    ("u2_16_36", [UP]*2 + [LEFT]*8 + [DOWN]*4 + [ACT5]),   # 15
    ("u2_12_36", [UP]*2 + [LEFT]*9 + [DOWN]*4 + [ACT5]),   # 16
]

for name, carry_seq in up_targets:
    seq = base + carry_seq
    run_and_check(name, seq, verbose=True)
    print()

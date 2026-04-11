#!/usr/bin/env python3
"""wa30 L1: Test different delivery slots to unblock RED's pathfinding."""
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

L0_SOL = [UP,UP,LEFT,UP,UP,UP,LEFT,LEFT,ACT5,RIGHT,RIGHT,RIGHT,ACT5,UP,RIGHT,RIGHT,
          ACT5,DOWN,LEFT,LEFT,ACT5,DOWN,ACT5,UP,UP,ACT5]

def run_test(name, seq, idle_action=UP):
    arcade = Arcade()
    env = arcade.make('wa30-ee6fef47')
    fd = env.reset()
    for a in L0_SOL:
        fd = env.step(a)
    game = env._game

    total_steps = game.kuncbnslnm.current_steps
    active = len(seq)
    idle = total_steps - active

    full = list(seq) + [idle_action] * idle
    print(f"\n=== {name}: {active}a + {idle}i = {total_steps} ===")
    print(f"  {''.join(AN.get(a,'?') for a in seq)}")

    for i, a in enumerate(full):
        fd = env.step(a)
        game = env._game
        step = i + 1

        if fd.state.name == 'WIN':
            sol_str = ''.join(AN.get(a,'?') for a in full[:step])
            print(f"  Step {step}: WIN!")
            print(f"  Full: {sol_str}")
            return True

    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = set((x, y) for (x, y) in game.wyzquhjerd)
    at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
    carried = len(game.zmqreragji)
    not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]
    print(f"  RESULT: {at_slot}/5 c={carried} not={not_at}")
    return False


# Base approach: RIGHT*9 + DOWN*3 + ACT5 + UP*2 + LEFT*N + DOWN*M + ACT5
# Carry offset (0,4): carried at (player_x, player_y + 4)
# For carried at (slot_x, slot_y): player_x = slot_x, player_y = slot_y - 4

# Slots: (12,28), (12,32), (12,36), (16,28), (16,32), (16,36)

# After UP*2: player at (48,12), carried at (48,16)
# LEFT to slot_x, DOWN to slot_y - 4

slots_to_test = [
    ("12_28", 9, 3),  # LEFT*9 to x=12, DOWN*3 to y=24, carried at y=28
    ("12_32", 9, 4),  # LEFT*9, DOWN*4, carried at y=32
    ("12_36", 9, 5),  # LEFT*9, DOWN*5, carried at y=36
    ("16_28", 8, 3),  # LEFT*8 to x=16, DOWN*3, carried at y=28
    ("16_32", 8, 4),  # LEFT*8, DOWN*4, carried at y=32
    ("16_36", 8, 5),  # LEFT*8, DOWN*5, carried at y=36
]

for slot_name, n_left, n_down in slots_to_test:
    path = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*n_left + [DOWN]*n_down + [ACT5]
    run_test(f"slot_{slot_name}", path)

# Also test with different idle actions for the best candidates
print("\n\n=== Idle action variants ===")
for slot_name, n_left, n_down in slots_to_test:
    path = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*n_left + [DOWN]*n_down + [ACT5]
    for idle_name, idle_act in [("R", RIGHT), ("D", DOWN), ("5", ACT5), ("L", LEFT)]:
        run_test(f"{slot_name}_i{idle_name}", path, idle_action=idle_act)

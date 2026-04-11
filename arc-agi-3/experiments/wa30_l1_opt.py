#!/usr/bin/env python3
"""wa30 L1: Optimized UP*2 paths — skip the wasted blocked-DOWN step."""
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

def run_test(name, seq, idle_action=UP, verbose_every=1):
    """Run a complete L1 test with detailed output."""
    fd = env.reset()
    for a in L0_SOL:
        fd = env.step(a)
    game = env._game

    total_steps = game.kuncbnslnm.current_steps
    active = len(seq)
    idle = total_steps - active
    if idle < 0:
        print(f"{name}: TOO LONG ({active} > {total_steps})")
        return False

    full = list(seq) + [idle_action] * idle
    print(f"\n=== {name}: {active} active + {idle} idle = {total_steps} total ===")
    print(f"  Sequence: {''.join(AN.get(a,'?') for a in seq)}")

    for i, a in enumerate(full):
        fd = env.step(a)
        game = env._game
        step = i + 1

        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        slots = set((x, y) for (x, y) in game.wyzquhjerd)
        at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
        carried = len(game.zmqreragji)

        if fd.state.name == 'WIN':
            print(f"  Step {step}: WIN! [{at_slot}/5]")
            return True

        # Detailed output
        if step <= active or step % verbose_every == 0 or at_slot >= 4 or step >= total_steps - 3:
            player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
            reds = game.current_level.get_sprites_by_tag("kdweefinfi")
            r_info = ""
            for r in reds:
                ci = f"c({game.nsevyuople[r].x},{game.nsevyuople[r].y})" if r in game.nsevyuople else ""
                r_info += f" r({r.x},{r.y}){ci}"
            # Player carry
            p_carry = ""
            if player in game.nsevyuople:
                c = game.nsevyuople[player]
                p_carry = f" P_carry({c.x},{c.y})"
            not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]
            print(f"  {step:2d} [{at_slot}/5] c={carried} P({player.x},{player.y}){p_carry}{r_info} not={not_at}")

    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = set((x, y) for (x, y) in game.wyzquhjerd)
    at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
    carried = len(game.zmqreragji)
    not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]
    print(f"  RESULT: {at_slot}/5 c={carried} not={not_at}")
    return False


# Key insight: RIGHT*9 + DOWN*3 gets to (48,20) facing DOWN.
# The extra DOWN in the old path was a wasted step (blocked by goal at (48,24)).
# ACT5 directly after DOWN*3 picks up the goal.
# Then UP*2 to y=12 clears the y=20 obstacle (goal at (40,20)).

# PATH A: to slot (12,28) — shortest carry
# Pickup at (48,20), carry offset (0,4). UP*2: player (48,12), carried (48,16).
# LEFT*9: player (12,12), carried (12,16). DOWN*3: player (12,24), carried (12,28).
# Total: 9+3+1+2+9+3+1 = 28 active, 42 idle
pathA = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5]
run_test("A_to_12_28", pathA)

# PATH B: to slot (16,28)
# LEFT*8 instead of LEFT*9: player (16,12), carried (16,16). DOWN*3: player (16,24), carried (16,28).
# Total: 9+3+1+2+8+3+1 = 27 active, 43 idle
pathB = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*8 + [DOWN]*3 + [ACT5]
run_test("B_to_16_28", pathB)

# PATH C: to slot (12,32) — deeper in slot area
# LEFT*9 + DOWN*4: player (12,28), carried (12,32).
# Total: 9+3+1+2+9+4+1 = 29 active, 41 idle
pathC = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*4 + [ACT5]
run_test("C_to_12_32", pathC)

# PATH D: to slot (16,32)
# LEFT*8 + DOWN*4: player (16,28), carried (16,32).
# Total: 9+3+1+2+8+4+1 = 28 active, 42 idle
pathD = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*8 + [DOWN]*4 + [ACT5]
run_test("D_to_16_32", pathD)

# PATH E: to slot (12,36) — original destination but shorter
# LEFT*9 + DOWN*5: player (12,32), carried (12,36).
# Total: 9+3+1+2+9+5+1 = 30 active, 40 idle
pathE = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*5 + [ACT5]
run_test("E_to_12_36", pathE)

# PATH F: to slot (16,36) — shorter x distance
# LEFT*8 + DOWN*5: player (16,32), carried (16,36).
# Total: 9+3+1+2+8+5+1 = 29 active, 41 idle
pathF = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*8 + [DOWN]*5 + [ACT5]
run_test("F_to_16_36", pathF)

# PATH G: approach from right side — RIGHT*9 + DOWN*4 + RIGHT + ACT5
# Player goes RIGHT*9 to (48,8), DOWN*4 tries to go to (48,24) but blocked at (48,20).
# Actually DOWN*4 = (48,8+16)=(48,24)... but (48,24) has a goal. So (48,20) + blocked.
# Last facing is still DOWN. Same as DOWN*3.
# Not helpful. Skip this.

# PATH H: Different approach entirely — interleave with AI timing
# Go RIGHT and DOWN to reach goal area faster with diagonal path
# RIGHT*5, DOWN*3, RIGHT*4 to (32,20) then go right to (48,20)?
# Wait, at (32,20) we can face DOWN to (32,24) — not a goal.
# Need to be at (48,20) facing (48,24). No shortcut.

# PATH I: UP*3 to go higher and potentially avoid more obstacles
# Total: 9+3+1+3+9+4+1 = 30 active
pathI = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*3 + [LEFT]*9 + [DOWN]*4 + [ACT5]
run_test("I_up3_to_12_32", pathI)

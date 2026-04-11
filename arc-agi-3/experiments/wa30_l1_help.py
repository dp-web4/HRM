#!/usr/bin/env python3
"""wa30 L1: Player helps RED by delivering a second goal during 'idle' time.

Path A delivers (48,24) to slot (12,28) in 28 steps.
Then player has 42 steps left. Use them to also deliver (40,20) to a slot,
removing it from RED's todo list so RED finishes faster.
"""
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

def make_fresh_env():
    arcade = Arcade()
    env = arcade.make('wa30-ee6fef47')
    fd = env.reset()
    for a in L0_SOL:
        fd = env.step(a)
    return env, fd

def run_test(name, seq, idle_action=UP):
    """Run with a FRESH env each time."""
    env, fd = make_fresh_env()
    game = env._game

    total_steps = game.kuncbnslnm.current_steps
    active = len(seq)
    idle = total_steps - active
    if idle < 0:
        print(f"{name}: TOO LONG ({active} > {total_steps})")
        return False

    full = list(seq) + [idle_action] * idle
    print(f"\n=== {name}: {active}a + {idle}i = {total_steps} ===")

    for i, a in enumerate(full):
        fd = env.step(a)
        game = env._game
        step = i + 1

        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        slots = set((x, y) for (x, y) in game.wyzquhjerd)
        at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
        carried = len(game.zmqreragji)

        if fd.state.name == 'WIN':
            sol_str = ''.join(AN.get(a,'?') for a in full[:step])
            print(f"  Step {step}: WIN! [{at_slot}/5]")
            print(f"  Full: {sol_str}")
            return True

        # Print at key moments
        if step <= active or at_slot >= 4 or step >= total_steps - 5:
            player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
            reds = game.current_level.get_sprites_by_tag("kdweefinfi")
            r_info = ""
            for r in reds:
                ci = f"c({game.nsevyuople[r].x},{game.nsevyuople[r].y})" if r in game.nsevyuople else ""
                r_info += f" r({r.x},{r.y}){ci}"
            p_carry = ""
            if player in game.nsevyuople:
                c = game.nsevyuople[player]
                p_carry = f" Pc({c.x},{c.y})"
            not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]
            print(f"  {step:2d} [{at_slot}/5] c={carried} P({player.x},{player.y}){p_carry}{r_info} not={not_at}")

    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = set((x, y) for (x, y) in game.wyzquhjerd)
    at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
    carried = len(game.zmqreragji)
    not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]
    print(f"  RESULT: {at_slot}/5 c={carried} not={not_at}")
    return False


# === First: reproduce path A to verify state budget ===
pathA = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5]
run_test("A_verify", pathA)

# === After path A delivery (28 steps), player is at (12,24), carried dropped ===
# Player can go help with (40,20)!
# From (12,24): need to reach (40,20) and carry to a slot.
#
# Route to (40,20):
#   RIGHT to x=40: RIGHT*7 = (40,24)
#   UP to y=20: Can player move to (40,20)? That position has a goal.
#   Player can't move into a goal position. Need to be adjacent.
#   From (40,24): facing UP = (40,20). ACT5 picks up!
#   But at step 28+7 = step 35, has RED already picked up (40,20)?
#
# From path A trace: RED picks up (40,20) at ~step 60.
# So at step 35, (40,20) should still be there!
#
# Carry to slot: offset (0,-4) [carried above player].
# Need carried at slot. Carried = (player_x, player_y - 4).
# For carried at (16,28): player at (16,32). From (40,24): LEFT*6 + DOWN*2 = 8 steps.
# For carried at (12,28): player at (12,32). LEFT*7 + DOWN*2 = 9 steps.
# For carried at (16,32): player at (16,36). LEFT*6 + DOWN*3 = 9 steps.
# For carried at (12,32): player at (12,36). LEFT*7 + DOWN*3 = 10 steps.
#
# Total for second delivery: RIGHT*7 + ACT5 + LEFT*6 + DOWN*2 + ACT5 = 17 steps
# Grand total: 28 + 17 = 45 active, 25 idle

# PATH H1: deliver (48,24) to (12,28), then (40,20) to (16,28)
# After drop at step 28: player at (12,24)
# RIGHT*7 to (40,24), ACT5 pickup (40,20), LEFT*6 to (16,24) carry to (16,20)?
# Wait, carry offset from facing UP: (0,-4). Carried at (player_x, player_y-4).
# If player at (40,24) and faces UP = (40,20), ACT5 picks up with offset (0,-4).
# Then LEFT*6: player (16,24), carried (16,20). Not a slot!
# For carried at (16,28): player at (16,32). From (40,24): LEFT*6 + DOWN*2.
# But carried offset is (0,-4): carried = (16, 32-4) = (16,28). ✓
# Total: 28 + 7 + 1 + 6 + 2 + 1 = 45

h1 = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] + \
     [RIGHT]*7 + [ACT5] + [LEFT]*6 + [DOWN]*2 + [ACT5]
run_test("H1_two_deliver", h1)

# PATH H2: deliver (48,24) to (12,28), then (40,20) to (12,32)
# After drop at (12,24): RIGHT*7 + ACT5 + LEFT*7 + DOWN*2 + ACT5
# player at (12,32), carried at (12,28)... wait (12,28) already has goal from delivery 1!
# Carried offset (0,-4): carried at (12, 32-4) = (12,28). BLOCKED by existing goal!
# Need different target.
# LEFT*7 + DOWN*3 + ACT5: player (12,36), carried (12,32). 11 steps carry.
h2 = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] + \
     [RIGHT]*7 + [ACT5] + [LEFT]*7 + [DOWN]*3 + [ACT5]
run_test("H2_to_12_32", h2)

# PATH H3: deliver (48,24) to (16,28) first (using LEFT*8 not LEFT*9)
# Then (40,20) to (12,28) — no conflict since (12,28) is free
# Delivery 1: RIGHT*9 + DOWN*3 + ACT5 + UP*2 + LEFT*8 + DOWN*3 + ACT5 = 27
# After step 27: player at (16,24)
# Delivery 2: RIGHT*6 to (40,24), ACT5, LEFT*7 to (12,24), DOWN*1 to (12,28)?
# Carry offset (0,-4): carried = (12, 28-4) = (12,24). NOT a slot.
# Need DOWN*2: player (12,32), carried (12,28). ✓
# Total: 27 + 6 + 1 + 7 + 2 + 1 = 44. Idle = 70 - 44 = 26.
h3 = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*8 + [DOWN]*3 + [ACT5] + \
     [RIGHT]*6 + [ACT5] + [LEFT]*7 + [DOWN]*2 + [ACT5]
run_test("H3_16_28_then_12_28", h3)

# PATH H4: deliver (48,24) to (12,28), then (40,20) to (16,32)
# After drop at (12,24): RIGHT*7 + ACT5 + LEFT*6 + DOWN*2 + ACT5 = 16 steps
# player (16,32), carried (16,28)... wait that hits the goal at (16,28) if RED delivered there
# Actually, carry offset (0,-4): player at (16,32), carried at (16,28).
# If (16,28) is already occupied by a goal from RED's delivery #1 (step 10), the carry
# validation would block the move because carried dest == existing goal position.
# So this won't work.

# PATH H5: carry (40,20) up first, then down on different x
# After (12,24): go UP to (12,20) - blocked? (12,20) should be free after step 28
# Then face right, but (40,20) is far. Player needs to reach adjacent to (40,20).
# Actually player CAN'T pick up from far away. Must be adjacent facing.
# So player must go to (40,24) and face UP, or (40,16) and face DOWN, etc.
# (40,24): RIGHT*7. Simplest.
# Or (36,20): RIGHT*6 + UP*1 = 7 steps. Then face RIGHT = (40,20). ACT5 picks up.
# Carry offset (4,0) [carried to right]. Then LEFT to bring carried left.
# For carried at (16,28): carried_x = player_x + 4. player_x = 12. carried at (16,28)?
# player at (12,28). But carry offset (4,0): carried at (16,28). ✓
# LEFT*6 + DOWN*2 + ACT5 = 9 steps. Total: 28 + 7 + 1 + 9 = 45.
h5 = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] + \
     [RIGHT]*6 + [UP]*1 + [RIGHT] + [ACT5] + [LEFT]*6 + [DOWN]*2 + [ACT5]
run_test("H5_face_right_pickup", h5)

# PATH H6: Two-goal with shorter first delivery
# Deliver (48,24) to (16,32) in 27 steps (LEFT*8 + DOWN*4)
# Then deliver (40,20) to (12,28) in ? steps
# After step 27: player at (16,28), dropped carried at (16,32)
# Go to (40,24): RIGHT*6 = 6 steps
# ACT5 facing UP (but rotation was last DOWN): player at (16,28), then UP*...
# Actually last move was ACT5 (drop). Rotation doesn't change. Before drop, last move
# was DOWN*4, so rotation = 180 (facing down). ACT5 doesn't change rotation.
# After step 27 at (16,28): need to go to (40,24) and face UP toward (40,20).
# RIGHT*6 to (40,28). UP*1 to (40,24). Now facing UP = (40,20). ACT5 pickup.
# Carry offset (0,-4). LEFT*7 to (12,24). Carried at (12,20). Not slot.
# DOWN*2: player (12,32), carried (12,28). ✓
# Total: 27 + 6 + 1 + 1 + 7 + 2 + 1 = 45
h6 = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*8 + [DOWN]*4 + [ACT5] + \
     [RIGHT]*6 + [UP]*1 + [ACT5] + [LEFT]*7 + [DOWN]*2 + [ACT5]
run_test("H6_16_32_then_12_28", h6)

# PATH H7: Most aggressive — three deliveries by player
# Deliver (48,24), (40,20), and (44,40)
# RED only needs to deliver (36,28) and (48,32) — easy!
# But this is probably too many steps. Let's see:
# (48,24) to (12,28): 28 steps
# (40,20) to (16,28): 17 steps (H1 pattern). Total so far: 45
# (44,40) to (12,32): from player at (16,32) after H1...
# RIGHT*7 to (44,32)? No, (44,40) is at y=40.
# This gets complicated. Skip for now.

# PATH H8: What if we DON'T idle UP but instead idle DOWN so player
# stays in slot area, potentially helping AI pathfind?
pathA_idleD = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5]
run_test("A_idle_DOWN", pathA_idleD, idle_action=DOWN)

pathA_idleR = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5]
run_test("A_idle_RIGHT", pathA_idleR, idle_action=RIGHT)

pathA_idleACT = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5]
run_test("A_idle_ACT5", pathA_idleACT, idle_action=ACT5)

#!/usr/bin/env python3
"""wa30 L1: Fixed two-goal delivery — avoid RED collision on carry path."""
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
    print(f"\n=== {name}: {active}a + {idle}i ===")
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

# ===================================================================
# The key problem with the 2-goal approach was: when carrying the
# first goal DOWN through the slot area, the RED is AT (12,32),
# blocking the carried goal's destination.
#
# Solution: After delivering goal 1, WAIT for RED to leave, then do
# delivery 2. Or take a path that avoids the RED entirely.
#
# NEW APPROACH: Deliver both goals via a route that doesn't go through
# the slot area column until the final drop.
#
# Plan: Deliver (48,24) to (12,28) via existing working path (28 steps).
# Then deliver (40,20) to a slot using a path that doesn't need to
# go through x=12, y=28-36.
#
# After delivery 1 (step 28): player at (12,24), goal at (12,28).
# Go to (40,20): RIGHT*7 to (40,24). UP blocked by (40,20). ACT5 pickup.
# Carry offset (0,-4).
#
# Instead of going LEFT all the way to x=12 (which conflicts with
# slot area), go LEFT to x=16 and carry DOWN:
#   LEFT*6 to (16,24), carried (16,20).
#   DOWN*3: player (16,36), carried (16,32). (16,32) IS a slot!
#   But is (16,32) occupied? RED delivered there at step 27-28.
#   YES, (16,32) is occupied! Can't drop there.
#
# OK: DOWN*2: player (16,32), carried (16,28). (16,28) occupied by RED #1!
#
# All x=16 slots are occupied by RED deliveries!
# x=12 slots: (12,28) occupied by player. (12,32) and (12,36) free.
#
# Need to get carried to (12,32) or (12,36).
# Carry offset (0,-4): carried = player_y - 4.
# For carried at (12,32): player at (12,36). Path: LEFT*7 to (12,24).
#   But to get from (12,24) to (12,36): DOWN*3.
#   (12,28) has a goal! Player can't go through.
#   DOWN*1: player (12,28)? Blocked by own delivery! Hmm.
#
# The problem: ALL paths to the lower slots go through (12,28).
# Unless we approach from a different x column.
#
# What if we approach from x=8?
# LEFT*8 to (8,24), carried (8,20). DOWN*3: player (8,36), carried (8,32).
# (8,32) is NOT a slot!
#
# What about LEFT*8 + DOWN*4 + RIGHT: player (8,40) + RIGHT to (12,40)?
# That's getting too complex.
#
# KEY INSIGHT: Use a different carry offset!
# If we pick up from the RIGHT instead of UP:
# Go to (36,20): RIGHT*6 to (36,24), UP to (36,20)... blocked by goal? No!
# (36,20) is NOT a goal position. Goals: (48,32),(36,28),(40,20),(48,24),(44,40).
# (36,20) is free! So we CAN go there.
# From (36,20), face RIGHT: (40,20) = goal. ACT5 picks up.
# Carry offset (4,0): carried to the RIGHT of player.
# Carry LEFT: player (12,20), carried (16,20).
# DOWN*2: player (12,28)... blocked!
#
# Hmm, same issue. Player can't enter (12,28) if goal is there.
# DOWN to (12,24+4)=(12,28) — wait player is at (12,20). DOWN = (12,24).
# Then (12,24) to (12,28) — blocked by goal.
#
# What if carry offset is (4,0) and we go:
# LEFT to x=12: player (12,20), carried (16,20).
# DOWN*2: player (12,28)... BLOCKED.
#
# LEFT to x=8: player (8,20), carried (12,20).
# DOWN*2: player (8,28), carried (12,28). Blocked by goal at (12,28)!
#
# Hmm. The delivered goal at (12,28) blocks ALL paths to lower x=12 slots.
# Unless we approach from below (y > 28).
#
# ALTERNATIVE: Don't deliver to (12,28)! Deliver to (12,32) or (12,36) instead.
# Then (12,28) is free for the second delivery.
#
# But we showed that delivering to (12,32) or (12,36) has the RED blocking
# the carry path. Unless we use a different timing.
#
# Let me try: deliver (48,24) to (12,32), wait a step for RED to leave, then continue.
# The RED blocks at (12,32) only during certain steps. By adding 1 idle step between
# the approach and the carry DOWN phase, the RED might have moved away.
#
# Delivery 1: RIGHT*9+DOWN*3+ACT5+UP*2+LEFT*9 = 24 steps. Player at (12,12).
# Then: wait for RED to clear (12,32). At step 24, RED is at ~(24,32).
# By step 27, RED reaches (12,32). If we wait until step 28+, RED drops and moves.
#
# After LEFT*9 (step 24): player at (12,12), carried at (12,16).
# DOWN: player (12,16), carried (12,20). Step 25.
# DOWN: player (12,20), carried (12,24). Step 26.
# DOWN: player (12,24), carried (12,28). Step 27. (12,28) is a slot! Can we drop here?
# ACT5: drop carried at (12,28). Step 28. DONE! 28 steps.
# This is path A. We already know this works!
#
# OK let me think about this more fundamentally.
#
# The problem: with 1-goal delivery, we get 5/5 c=1 (RED needs 1 more step).
# With 2-goal delivery, path conflicts prevent proper delivery.
#
# What if we do a PARTIAL second delivery — just move (40,20) CLOSER to the slots?
# The RED then picks it up from a closer position, saving steps.

# TEST 1: Player carries (40,20) to (28,20) and drops there
# RED picks up from (28,20) instead of (40,20), saving ~3 steps.
# 28 + 7+1+1 + 2+1 = 28 + 12 = 40 active, 30 idle
# After delivery 1 at (12,24): RIGHT*7 to (40,24). UP blocked. ACT5.
# LEFT*3 to (28,24), carried (28,20). ACT5 drop. But (28,20) not a slot.
# RED would then pathfind to (28,20) and carry to slot.
nudge_A = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*3 + [ACT5]
)
run_test("nudge_A_to_28_20", nudge_A)

# TEST 2: Nudge to (24,20) — save ~4 steps for RED
nudge_B = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*4 + [ACT5]
)
run_test("nudge_B_to_24_20", nudge_B)

# TEST 3: Nudge to (20,20)
nudge_C = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*5 + [ACT5]
)
run_test("nudge_C_to_20_20", nudge_C)

# TEST 4: Nudge to (16,20)
nudge_D = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*6 + [ACT5]
)
run_test("nudge_D_to_16_20", nudge_D)

# TEST 5: Nudge to (12,20)
nudge_E = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*7 + [ACT5]
)
run_test("nudge_E_to_12_20", nudge_E)

# TEST 6: Try idle RIGHT after nudge (player moves away from slot area)
run_test("nudge_B_iR", nudge_B, idle_action=RIGHT)
run_test("nudge_C_iR", nudge_C, idle_action=RIGHT)
run_test("nudge_D_iR", nudge_D, idle_action=RIGHT)

# TEST 7: Don't pick up (40,20), just push player near it during idle
# to change RED's BFS landscape
push_path = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5]
# After delivery, park at (40,24) — near (40,20)
push_park = push_path + [RIGHT]*7
run_test("park_near_40_20", push_park)

# Park at (40,20) offset positions to change RED BFS
push_park2 = push_path + [RIGHT]*7 + [DOWN]*4  # park at (40,40)
run_test("park_40_40", push_park2)

# Park at intermediate position
push_park3 = push_path + [RIGHT]*3  # park at (24,24)
run_test("park_24_24", push_park3)

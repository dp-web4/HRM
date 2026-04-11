#!/usr/bin/env python3
"""wa30 L1: Player delivers TWO goals. RED handles the other 3."""
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

        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        slots = set((x, y) for (x, y) in game.wyzquhjerd)
        at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
        carried = len(game.zmqreragji)

        if fd.state.name == 'WIN':
            sol_str = ''.join(AN.get(a,'?') for a in full[:step])
            print(f"  Step {step}: WIN! [{at_slot}/5]")
            print(f"  Full solution: {sol_str}")
            return True, sol_str, step

        # Print at key steps
        if step <= active or at_slot >= 4 or step >= total_steps - 5 or (step % 10 == 0):
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
    return False, None, None


# ===== TWO-GOAL DELIVERY PLANS =====
#
# Delivery 1: (48,24) → slot (12,28)
#   RIGHT*9 + DOWN*3 + ACT5 + UP*2 + LEFT*9 + DOWN*3 + ACT5 = 28 steps
#   After: player at (12,24)
#
# Delivery 2: (40,20) → slot (12,32)
#   After delivery 1, player at (12,24).
#   Go to (40,24): RIGHT*7 = 7 steps (step 29-35)
#   Face UP toward (40,20): UP blocked by goal → rotation UP. 1 step (step 36)
#   ACT5 pickup: carry offset (0,-4). 1 step (step 37)
#   Carry LEFT to x=12: LEFT*7 = 7 steps (step 38-44). Carried at (12,20).
#   Carry DOWN to slot: DOWN*3 to (12,36), carried (12,32). 3 steps (step 45-47).
#   ACT5 drop at (12,32). 1 step (step 48).
#   Total delivery 2: 7+1+1+7+3+1 = 20 steps
#   Grand total: 28 + 20 = 48 active, 22 idle

plan_2goal_A = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +  # deliver (48,24) to (12,28)
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*7 + [DOWN]*3 + [ACT5]  # deliver (40,20) to (12,32)
)
run_test("2goal_A", plan_2goal_A)

# ===== VARIANT B: Deliver (40,20) to (12,36) instead of (12,32) =====
# After pickup: LEFT*7 + DOWN*4 + ACT5. player at (12,40), carried at (12,36).
# 7+1+1+7+4+1 = 21 steps. Total: 49 active, 21 idle.
plan_2goal_B = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*7 + [DOWN]*4 + [ACT5]
)
run_test("2goal_B_12_36", plan_2goal_B)

# ===== VARIANT C: Deliver (48,24) to (12,32), (40,20) to (12,28) =====
# Delivery 1: RIGHT*9+DOWN*3+ACT5+UP*2+LEFT*9+DOWN*4+ACT5 = 29 steps. Player at (12,28).
# Delivery 2 from (12,28): RIGHT*7 to (40,28).
#   UP*2 to (40,20) — but goal at (40,20) blocks! Stays at (40,24), faces UP.
#   Actually, from (40,28) UP: to (40,24). Is (40,24) clear? Yes.
#   Then UP again: to (40,20) blocked by goal. Faces UP.
#   ACT5: facing (40,20) = goal. Pick up. Carry offset (0,-4).
#   LEFT*7 to (12,24). Carried at (12,20). Not slot.
#   DOWN*1: player (12,28), carried (12,24). (12,24) NOT a slot.
#   DOWN*2: player (12,32), carried (12,28). (12,28) IS slot! ✓
#   But wait, (12,32) already has a goal from delivery 1. Player can't move to (12,32)?
#   Hmm, (12,32) has the player's delivered goal. Player can't enter that cell.
#   So DOWN from (12,28) to (12,32) is blocked by goal at (12,32)!
#
# Let me try (12,28) as delivery slot: carry to (12,32), carried (12,28).
# But from (40,24) after pickup, need to get to (12,32).
# LEFT*7 to (12,24). DOWN*2: player (12,32). But (12,32) has delivery 1!
#
# This is tricky. Skip this variant.

# ===== VARIANT D: Deliver both to x=12 but different approach to (40,20) =====
# From (12,24): DOWN*1 to (12,28) blocked (goal there!).
# Hmm, after delivery 1, player is at (12,24) and goal at (12,28).
# Player can go RIGHT*7 without going through (12,28).
# OK variant A should work. Let me also try:

# ===== VARIANT E: Pick up (40,20) from the right side =====
# From (12,24): RIGHT*8 to (44,24). DOWN to (44,28). LEFT to (40,28)?
# No, (44,28) to (40,28) is LEFT. And facing LEFT = (36,28). Not (40,20).
# This doesn't help.

# ===== VARIANT F: Deliver (48,24) to (12,28), (40,20) to (16,32) =====
# After delivery 1 at (12,24):
# RIGHT*7 to (40,24). UP. ACT5. Carry LEFT*6 to (16,24), carried (16,20).
# DOWN*3: player (16,36), carried (16,32). ✓
# 7+1+1+6+3+1 = 19 steps. Total: 28+19 = 47 active, 23 idle.
plan_2goal_F = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*6 + [DOWN]*3 + [ACT5]
)
run_test("2goal_F_16_32", plan_2goal_F)

# ===== VARIANT G: Deliver (48,24) to (12,28), (40,20) to (16,36) =====
# LEFT*6 + DOWN*4 + ACT5. 7+1+1+6+4+1 = 20. Total: 48, 22 idle.
plan_2goal_G = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*3 + [ACT5] +
    [RIGHT]*7 + [UP] + [ACT5] + [LEFT]*6 + [DOWN]*4 + [ACT5]
)
run_test("2goal_G_16_36", plan_2goal_G)

# ===== VARIANT H: Deliver (48,24) to (12,32), (40,20) to (12,36) =====
# Delivery 1: ...+DOWN*4+ACT5 = 29 steps. Player at (12,28).
# Delivery 2: RIGHT*7 to (40,28). UP to (40,24). UP blocked by (40,20). ACT5 pickup.
#   LEFT*7 to (12,24), carried (12,20). DOWN*4: player (12,40), carried (12,36). ✓
#   But can player go through (12,28) → (12,32)? (12,32) has delivery 1. BLOCKED at (12,32)!
#   So DOWN from (12,28) hits (12,32) and stops. Can't proceed.
#
# Alternative: from (40,24), LEFT*7 to (12,24), carried (12,20).
#   Actually player needs to go DOWN from (12,24) to get carried to slot.
#   (12,24) → DOWN → (12,28). From (12,28) → DOWN → (12,32) BLOCKED by goal.
#   So this doesn't work.

# ===== VARIANT I: Deliver (48,24) to (12,36), (40,20) to (12,28) =====
# Delivery 1: ...+DOWN*5+ACT5 = 30 steps. Player at (12,32).
# But (12,32) is above (12,36), which now has a goal.
# Delivery 2: RIGHT*7 to (40,32). UP*3 to (40,20) — blocked at (40,20).
#   Actually: UP from (40,32): (40,28), then (40,24), then blocked at (40,20).
#   2 UP steps to reach (40,24). Then UP blocked. ACT5 pickup.
#   7+2+1+1 = 11 steps approach.
#   LEFT*7 to (12,24), carried (12,20). DOWN*2: player (12,32). (12,32) free?
#   (12,36) has goal. (12,32) should be free. carried (12,28). ✓
#   7+2 = 9 carry steps + 1 drop = 10.
#   Total: 30 + 11 + 10 = 51 active, 19 idle.
plan_2goal_I = (
    [RIGHT]*9 + [DOWN]*3 + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*5 + [ACT5] +
    [RIGHT]*7 + [UP]*2 + [UP] + [ACT5] + [LEFT]*7 + [DOWN]*2 + [ACT5]
)
run_test("2goal_I_12_36_12_28", plan_2goal_I)

# ===== VARIANT J: Different order — deliver (40,20) FIRST, then (48,24) =====
# (40,20) is closer. Player at (12,8).
# RIGHT*7 to (40,8). DOWN*3 to (40,20) — BLOCKED by goal!
# Last facing = DOWN = (40,24). Not (40,20).
# Need: DOWN*2 to (40,16). DOWN blocked at (40,20). ACT5 facing DOWN=(40,20). Pick up.
# No wait: (40,20) has a goal. From (40,16), DOWN tries (40,20) which is blocked.
# Player stays at (40,16), faces DOWN=(40,20). ACT5 picks up.
# 7+2+1+1 = 11 steps. Carry offset (0,4): carried at (player_x, player_y+4).
# Player at (40,16), carried at (40,20).
# UP*2: player (40,8), carried (40,12).
# LEFT*7: player (12,8), carried (12,12).
# DOWN*4: player (12,24), carried (12,28). ✓
# 2+7+4+1 = 14 carry steps.
# Then delivery 2: (48,24) from (12,24). RIGHT*9 to (48,24) BLOCKED by goal.
# RIGHT*9 to (48,24) — blocked at step... (48,24) has a goal. Player at (44,24) blocked.
# Actually player goes RIGHT from (12,24): (16,24), (20,24)... to (44,24). Then RIGHT blocked by (48,24).
# That's only RIGHT*8 to (44,24). Faces RIGHT = (48,24) = goal. ACT5 pickup!
# 8+1 = 9 steps. Carry offset (4,0): carried at (player_x+4, player_y).
# Player at (44,24), carried at (48,24).
# UP*2: player (44,16), carried (48,16).
# LEFT*8: player (12,16), carried (16,16).
# DOWN*4: player (12,32), carried (16,32). ✓ IS (16,32) a slot? Yes!
# Wait, carry offset (4,0): carried always 4 to right.
# LEFT*8: player from (44,16) → (12,16). Carried from (48,16) → (16,16). ✓
# DOWN*4: player (12,32), carried (16,32). ✓ (16,32) is slot!
# 2+8+4+1 = 15 carry steps.
# But wait, during LEFT, at player x=16, carried at x=20. At player x=12, carried x=16. OK.
# Total: 11+14 + 9+15 = 49 active, 21 idle.
# Hmm, but I need to be careful about where (12,28) is after delivery 1.
# After delivery 1 drops at (12,28): slot occupied.
# Delivery 2: carry (48,24) to (16,32). Player path goes at y=24, then y=16, then y=32.
# At y=24: (12,24) to (44,24) — is (12,28) blocking? Player at y=24, not y=28. Fine.
# At y=16: (44,16) to (12,16) — no goals here.
# At y=32: (12,32) — free (no goal yet).
# Carried at y=24: same. (16,32) — free.
# Actually wait: (16,28) might be occupied by RED's first delivery. (16,32) might be free.
# Let me just test.

plan_2goal_J = (
    # Delivery 1: (40,20) to (12,28)
    [RIGHT]*7 + [DOWN]*2 + [DOWN] + [ACT5] + [UP]*2 + [LEFT]*7 + [DOWN]*4 + [ACT5] +
    # Player at (12,24) — wait, no. After UP*2 from (40,16): (40,8). LEFT*7: (12,8). DOWN*4: (12,24).
    # Hmm, carry offset (0,4): at (12,24), carried (12,28). Drop. Player at (12,24).
    # Delivery 2: (48,24) to (16,32)
    [RIGHT]*8 + [RIGHT] + [ACT5] +  # RIGHT*8 to (44,24), RIGHT blocked, faces RIGHT, ACT5 pickup
    # Carry offset (4,0): player at (44,24), carried (48,24)
    [UP]*2 + [LEFT]*8 + [DOWN]*4 + [ACT5]  # UP*2, LEFT*8 to (12,16) carried (16,16), DOWN*4 to (12,32) carried (16,32)
)
# Count: 7+2+1+1+2+7+4+1 + 8+1+1+2+8+4+1 = 25 + 25 = 50. Total 50a, 20i.
run_test("2goal_J_40_20_first", plan_2goal_J)

# VARIANT K: Same as J but try RIGHT*8+ACT5 directly (skip blocked RIGHT)
# From (12,24), RIGHT*8 = (44,24). Facing RIGHT = (48,24) = goal. ACT5 picks up.
# No need for extra RIGHT since facing is already set by the movement!
# Save 1 step!
plan_2goal_K = (
    [RIGHT]*7 + [DOWN]*2 + [DOWN] + [ACT5] + [UP]*2 + [LEFT]*7 + [DOWN]*4 + [ACT5] +
    [RIGHT]*8 + [ACT5] +
    [UP]*2 + [LEFT]*8 + [DOWN]*4 + [ACT5]
)
# 25 + 24 = 49a, 21i
run_test("2goal_K_optimized", plan_2goal_K)

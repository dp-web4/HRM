#!/usr/bin/env python3
"""wa30 L1: Try path variants that save 1 step to give AI time for final drop."""
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
ACTION_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R', ACT5: '5'}

arcade = Arcade()
env = arcade.make('wa30-ee6fef47')

# L0 solution
L0_SOL = [UP,UP,LEFT,UP,UP,UP,LEFT,LEFT,ACT5,RIGHT,RIGHT,RIGHT,ACT5,UP,RIGHT,RIGHT,
          ACT5,DOWN,LEFT,LEFT,ACT5,DOWN,ACT5,UP,UP,ACT5]

def reset_to_l1():
    fd = env.reset()
    for a in L0_SOL:
        fd = env.step(a)
    return fd

def run_sequence(name, seq, idle_action=UP):
    """Run a sequence for L1, pad with idle_action to fill budget, report status each step."""
    fd = reset_to_l1()
    game = env._game

    total_steps = game.kuncbnslnm.current_steps
    active_steps = len(seq)
    idle_steps = total_steps - active_steps

    if idle_steps < 0:
        print(f"{name}: TOO LONG ({active_steps} > {total_steps})")
        return False

    full_seq = list(seq) + [idle_action] * idle_steps

    won = False
    for i, a in enumerate(full_seq):
        fd = env.step(a)
        game = env._game

        step_num = i + 1
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        slots = set((x, y) for (x, y) in game.wyzquhjerd)
        at_slot = sum(1 for g in goals if (g.x, g.y) in slots)
        carried = len(game.zmqreragji)

        reds = game.current_level.get_sprites_by_tag("kdweefinfi")
        red_info = ""
        for r in reds:
            rx, ry = r.x, r.y
            c_info = ""
            if r in game.nsevyuople:
                c = game.nsevyuople[r]
                c_info = f"c({c.x},{c.y})"
            red_info += f" red=({rx},{ry}){c_info}"

        # Check goals not at slots
        not_at = [(g.x, g.y) for g in goals if (g.x, g.y) not in slots]

        if fd.state.name == 'WIN':
            print(f"{name}: WIN at step {step_num}! [{at_slot}/5]")
            return True

        # Print key steps
        if step_num <= active_steps or at_slot >= 4 or step_num >= total_steps - 5 or carried == 0 and at_slot == 5:
            print(f"  {step_num} [{at_slot}/5] c={carried}{red_info} not={not_at}")

    print(f"{name}: {at_slot}/5 c={carried} not={not_at}")
    return False


# Original to_12_36: 31 active + 39 idle = 70
# RIGHT*9 + DOWN*3 + DOWN + ACT5 + UP*2 + LEFT*9 + DOWN*5 + ACT5 + UP*39
# Problem: 5/5 at step 69 but carried=1, no step left for drop

# === VARIANT 1: UP*1 instead of UP*2 (save 1 step in carry phase) ===
# After pickup at (48,8), go UP*1 to (48,4), then LEFT*9 to (12,4), DOWN*8 to (12,36), drop
v1 = [RIGHT]*9 + [DOWN]*3 + [DOWN] + [ACT5] + [UP]*1 + [LEFT]*9 + [DOWN]*8 + [ACT5]
print(f"\n=== V1: UP*1 carry, DOWN*8 drop ({len(v1)} active) ===")
run_sequence("v1_up1", v1)

# === VARIANT 2: Same but DOWN*7 + ACT5 (drop at y=32 instead of 36) ===
v2 = [RIGHT]*9 + [DOWN]*3 + [DOWN] + [ACT5] + [UP]*1 + [LEFT]*9 + [DOWN]*7 + [ACT5]
print(f"\n=== V2: UP*1 carry, DOWN*7 drop to y=32 ({len(v2)} active) ===")
run_sequence("v2_up1_d7", v2)

# === VARIANT 3: No UP at all — pickup facing down, go LEFT immediately ===
# After DOWN*3 to (48,20), facing down, facing cell is (48,24) = goal
# Pickup: carry offset = (0, CELL) = below player
# Then LEFT*9 to (12,20), then DOWN*4 to (12,36) for carried obj
# Wait... carried offset is (0,4), so carried at (12,24) when player at (12,20)
# Need carried at (12,36) → player at (12,32), so DOWN*3 more from (12,20) = (12,32)
# Actually let's trace: pickup at (48,20), carried at (48,24), offset=(0,4)
# LEFT*9: player (12,20), carried (12,24) — if path is clear
# DOWN*4: player (12,36), carried (12,40) — that's y=40, not a slot
# Hmm, offset (0,4) means carried is BELOW player
# For carried at slot y=36, player needs to be at y=32
# DOWN*3: player (12,32), carried (12,36) — drop here
v3 = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [LEFT]*9 + [DOWN]*3 + [ACT5]
print(f"\n=== V3: pickup facing down, no UP ({len(v3)} active) ===")
run_sequence("v3_noUP", v3)

# === VARIANT 4: RIGHT*9 + DOWN*4 to goal, pickup facing right? No, goal is below ===
# Actually at (48,20), facing is determined by last move direction
# Last move was DOWN, so facing = (48,24) which IS the goal
# Let me try: approach from left side
# RIGHT*8 to (44,8), DOWN*4 to (44,24) — goal is at (48,24), adjacent right
# Last move DOWN, facing (44,28) — not the goal
# Need to face RIGHT: do RIGHT to (48,24)... but that IS the goal position, can't move there
# Hmm, goals are collidable. So from (44,24) RIGHT is blocked by goal at (48,24)
# That means facing would update to RIGHT direction but move fails
# Actually in simulate_fast, if move blocked, we still update rotation
# But in engine, does rotation update on blocked move? Let's check...

# === VARIANT 5: Original path but RIGHT*8 instead of RIGHT*9 ===
# Approach goal from (44,8), DOWN*4 to (44,24)
# Goal at (48,24) is to the right. Can we face RIGHT then pickup?
# Try: RIGHT to (48,24) — blocked by goal. But does rotation update?
# If rotation updates: facing = (48,24) = goal, ACT5 picks up
# carry offset = (4,0), carried to right of player
# LEFT*8: player (12,24), carried (16,24)... not ideal
# Actually let's just try approach from (44,24), try RIGHT+ACT5
v5a = [RIGHT]*8 + [DOWN]*4 + [RIGHT] + [ACT5] + [UP]*1 + [LEFT]*8 + [DOWN]*3 + [ACT5]
print(f"\n=== V5a: approach from left, RIGHT blocked then pickup ({len(v5a)} active) ===")
run_sequence("v5a", v5a)

# === VARIANT 6: Approach from below ===
# RIGHT*9 + DOWN*5 to (48,28), UP to (48,24) blocked by goal
# face UP, ACT5 picks up with offset (0,-4)
# carried above player. LEFT*9 to (12,28), UP*...
# For carried at (12,28) slot, that's already a slot!
# player at (12,28), carried at (12,24) — offset (0,-4)
# Hmm (12,24) is slot? Let me check: slots are at (3-4, 7-9) in cells = (12-16, 28-36)
# (12,24) = cell (3,6) — NOT a slot. Slots start at cell y=7 = y=28
# With offset (0,-4): carried at player_y - 4
# For carried at y=28: player at y=32. For carried at y=32: player at y=36
# Let's try: go to (48,28), face up, pickup, go left, go down to place
v6 = [RIGHT]*9 + [DOWN]*5 + [UP] + [ACT5] + [LEFT]*9 + [DOWN]*1 + [ACT5]
print(f"\n=== V6: approach from below ({len(v6)} active) ===")
run_sequence("v6_below", v6)

# === VARIANT 7: Most direct — RIGHT*9, DOWN*3, pickup, UP*0, LEFT*9, DOWN*5, drop ===
# Same as v3 essentially but let me be precise about the carry offset
# At (48,20), last move DOWN, facing (48,24) = goal. ACT5 pickup.
# Carry offset = facing - player = (0,4). Carried below.
# LEFT*9: player to (12,20), carried to (12,24). Need to check path clearance!
# DOWN*5: player to (12,40), carried to (12,44)... too far
# For carried at (12,36): player at (12,32). DOWN from (12,20) to (12,32) = DOWN*3
# That's v3. Let me also try carry to (16,36)
v7 = [RIGHT]*9 + [DOWN]*3 + [ACT5] + [LEFT]*8 + [DOWN]*3 + [ACT5]
print(f"\n=== V7: carry to (16,36) ({len(v7)} active) ===")
run_sequence("v7_16_36", v7)

# === VARIANT 8: Try to_12_36 but with only DOWN*2 before pickup ===
# From (48,8), DOWN*2 to (48,16). Facing (48,20) — is there a goal there?
# Goal at (48,24). (48,20) is NOT a goal position.
# Nope, need DOWN*3 to reach (48,20) where facing is (48,24)
# What about DOWN*4 to (48,24)? Can't - goal blocks

# === VARIANT 9: Diagonal approach — mix RIGHT and DOWN ===
v9 = [RIGHT]*5 + [DOWN]*3 + [RIGHT]*4 + [ACT5] + [UP]*1 + [LEFT]*9 + [DOWN]*5 + [ACT5]
print(f"\n=== V9: diagonal approach ({len(v9)} active) ===")
run_sequence("v9_diag", v9)

# === VARIANT 10: to_12_36 original but idle=DOWN instead of UP ===
# Maybe DOWN idle keeps player out of AI's path differently
v10 = [RIGHT]*9 + [DOWN]*3 + [DOWN] + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*5 + [ACT5]
print(f"\n=== V10: original path, idle=DOWN ({len(v10)} active) ===")
run_sequence("v10_idleDOWN", v10, idle_action=DOWN)

# === VARIANT 11: original path, idle=LEFT ===
v11 = [RIGHT]*9 + [DOWN]*3 + [DOWN] + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*5 + [ACT5]
print(f"\n=== V11: original path, idle=LEFT ({len(v11)} active) ===")
run_sequence("v11_idleLEFT", v11, idle_action=LEFT)

# === VARIANT 12: original path, idle=ACT5 ===
v12 = [RIGHT]*9 + [DOWN]*3 + [DOWN] + [ACT5] + [UP]*2 + [LEFT]*9 + [DOWN]*5 + [ACT5]
print(f"\n=== V12: original path, idle=ACT5 ({len(v12)} active) ===")
run_sequence("v12_idleACT5", v12, idle_action=ACT5)

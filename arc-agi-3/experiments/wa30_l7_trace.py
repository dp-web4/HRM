#!/usr/bin/env python3
"""Run current greedy solver on L7, trace what happens each step."""
import sys, os
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

src = open('arc-agi-3/experiments/wa30_solve_final.py').read()
cut = src.index('total_actions = 0')
exec(src[:cut], globals())

KNOWN = KNOWN_SOLUTIONS
all_solutions = []
for lv in range(7):
    fd = replay_from_solutions(all_solutions)
    blues = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    has_ai = len(blues)>0 or len(whites)>0
    if lv in KNOWN:
        sol = string_to_actions(KNOWN[lv])
    elif not has_ai:
        sol = solve_no_ai()
    else:
        sol = solve_with_ai(all_solutions)
    fd = replay_from_solutions(all_solutions)
    before = fd.levels_completed
    actual = []
    for m in sol:
        fd = env.step(m); actual.append(m)
        if fd.levels_completed > before or fd.state.name == 'WIN': break
    all_solutions.append(actual)

# Now at L7 start
fd = replay_from_solutions(all_solutions)
print("=== L7 ===")
slots = game.wyzquhjerd
slots2 = game.lqctaojiby
slot_aligned = set((sx, sy) for sx, sy in slots if sx % 4 == 0 and sy % 4 == 0)
print(f"Aligned correct slots ({len(slot_aligned)}): {sorted(slot_aligned)}")

# Trace
for step in range(150):
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    blues = game.current_level.get_sprites_by_tag("kdweefinfi")
    collidable = set(game.pkbufziase)
    blocked = set(game.qthdiggudy)
    pc = game.nsevyuople.get(player, None)
    placed = sum(1 for g in goals if (g.x,g.y) in slots and g not in game.zmqreragji)
    wrong = sum(1 for g in goals if (g.x,g.y) in slots2 and g not in game.zmqreragji)
    blue_carry = {(b.x,b.y): game.nsevyuople.get(b) for b in blues}
    white_carry = {(w.x,w.y): game.nsevyuople.get(w) for w in whites}
    action = plan_action(player, goals, whites, blues, collidable, blocked, slots, slots2, slot_aligned, pc)
    if action is None:
        action = UP
    fd = env.step(action)
    if step % 10 == 0 or fd.levels_completed > 7 or fd.state.name in ('LOSE','GAME_OVER'):
        bs = [(b.x,b.y, '+P' if game.nsevyuople.get(b) else '') for b in blues]
        ws = [(w.x,w.y, '+P' if game.nsevyuople.get(w) else '') for w in whites]
        print(f"s{step:3d} a={ACTION_NAMES[action]} p=({player.x},{player.y}){'+P' if pc else ''} placed={placed} wrong={wrong} B={bs} W={ws}")
    if fd.levels_completed > 7 or fd.state.name in ('LOSE','GAME_OVER'):
        print(f"End: {fd.state.name} step={step}")
        goals2 = game.current_level.get_sprites_by_tag("geezpjgiyd")
        for g in goals2:
            status = 'CORRECT' if (g.x,g.y) in slots else ('WRONG' if (g.x,g.y) in slots2 else 'LOOSE')
            claimed = game.zmqreragji.get(g)
            ctag = ''
            if claimed:
                if 'ysysltqlke' in claimed.tags: ctag = ' (WHITE-carrying)'
                elif 'kdweefinfi' in claimed.tags: ctag = ' (BLUE-carrying)'
                elif 'wbmdvjhthc' in claimed.tags: ctag = ' (PLAYER-carrying)'
            print(f"  piece ({g.x},{g.y}) {status}{ctag}")
        break

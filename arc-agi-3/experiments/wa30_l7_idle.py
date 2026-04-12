#!/usr/bin/env python3
"""L7: player does nothing (no-op bouncing), see how many blues deliver alone."""
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

# Player idles — repeatedly UP (and bounces back from wall)
fd = replay_from_solutions(all_solutions)
slots = game.wyzquhjerd
slots2 = game.lqctaojiby
print("=== L7 player idles ===")
for step in range(150):
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    placed = sum(1 for g in goals if (g.x,g.y) in slots and g not in game.zmqreragji)
    wrong = sum(1 for g in goals if (g.x,g.y) in slots2 and g not in game.zmqreragji)
    if step % 15 == 0:
        print(f"s{step:3d} placed={placed} wrong={wrong}")
    # Alternate U/D so player oscillates in place (hits wall eventually)
    fd = env.step(UP if step%2==0 else DOWN)
    if fd.levels_completed > 7 or fd.state.name in ('LOSE','GAME_OVER'):
        print(f"End: {fd.state.name} step={step} placed={placed}")
        break

goals2 = game.current_level.get_sprites_by_tag("geezpjgiyd")
placed = sum(1 for g in goals2 if (g.x,g.y) in slots and g not in game.zmqreragji)
wrong = sum(1 for g in goals2 if (g.x,g.y) in slots2 and g not in game.zmqreragji)
print(f"FINAL placed={placed} wrong={wrong}")
for g in goals2:
    st = 'C' if (g.x,g.y) in slots else ('X' if (g.x,g.y) in slots2 else 'L')
    claimed = game.zmqreragji.get(g)
    ct = ''
    if claimed:
        if 'ysysltqlke' in claimed.tags: ct = 'W'
        elif 'kdweefinfi' in claimed.tags: ct = 'B'
    print(f"  ({g.x},{g.y}) {st}{ct}")

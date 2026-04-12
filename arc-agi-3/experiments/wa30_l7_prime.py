#!/usr/bin/env python3
"""L7: Test 'prime' strategies where initial moves are hardcoded to kill a white."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

src = open('arc-agi-3/experiments/wa30_l7_search.py').read()
# strip the trace and random-search sections
cut = src.index('t0 = time.time()')
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

AM = {'U': UP, 'D': DOWN, 'L': LEFT, 'R': RIGHT, '5': ACT5}

def try_prime(prime_str, label):
    fd = replay_from_solutions(all_solutions)
    slots = game.wyzquhjerd
    slots2 = game.lqctaojiby
    slot_aligned = set((sx,sy) for sx,sy in slots if sx%4==0 and sy%4==0)
    prime = [AM[c] for c in prime_str]
    solution = []
    peak = 0
    final = 0
    for step in range(150):
        player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        whites = game.current_level.get_sprites_by_tag("ysysltqlke")
        blues = game.current_level.get_sprites_by_tag("kdweefinfi")
        collidable = set(game.pkbufziase)
        blocked = set(game.qthdiggudy)
        placed = sum(1 for g in goals if (g.x,g.y) in slots and g not in game.zmqreragji)
        peak = max(peak, placed)
        final = placed
        if step < len(prime):
            a = prime[step]
        else:
            a = smart_action(player,goals,whites,blues,collidable,blocked,slots,slots2,slot_aligned)
        fd = env.step(a)
        solution.append(a)
        if fd.levels_completed > 7 or fd.state.name == 'WIN':
            print(f"[{label}] WIN! {len(solution)} moves")
            return solution
        if fd.state.name in ('LOSE','GAME_OVER'):
            break
    print(f"[{label}] peak={peak} final={final}")
    return None

# Winner from last run: RRRUUURRRR5 (upper_march) -> 10
# Explore variations
primes = [
    'RRRUUURRRR5',       # baseline winner
    'RRRUUURRRRR5',
    'RRRUUURRR5',
    'RRRUUURRRR',
    'RRRUUURRR',
    'RRRUUURRRRRR5',
    'RRRUUURRRR5UUU',    # extra ups after
    'RRRUUURRRR55',
    'RRRUUU',
    'RRR',
    '',
    'RRRUUURRRRUURRRR',  # longer approach into upper-right
    'RRRUUURRRRUUU',
]
for pr in primes:
    try_prime(pr, pr if pr else 'none')


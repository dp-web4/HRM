#!/usr/bin/env python3
"""Run L4 beam search with widened params."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
from arc_agi import Arcade
from sk48_solve_final import KNOWN, solve_beam, ACTION_NAMES

def drive(env, game, act_or_pair):
    if isinstance(act_or_pair, tuple):
        act, data = act_or_pair
    else:
        act, data = act_or_pair, None
    env.step(act, data)
    while game.ljprkjlji or game.pzzwlsmdt:
        env.step(act, data)
    while game.lgdrixfno >= 0 and game.lgdrixfno < 35:
        env.step(act, data)

BEAM = int(os.environ.get('BEAM', '800'))
DEPTH = int(os.environ.get('DEPTH', '120'))
TIMEOUT = int(os.environ.get('TIMEOUT', '1800'))
TARGET = int(os.environ.get('TARGET', '4'))

arcade = Arcade()
env = arcade.make('sk48-41055498')
env.reset()
game = env._game

for lv in range(TARGET):
    for a in KNOWN[lv]:
        drive(env, game, a)
print(f"At L{game.level_index}, budget={game.qiercdohl}", flush=True)

t0 = time.time()
use_click = len(game.xpmcmtbcv) > 1
print(f"use_click={use_click}", flush=True)
sol = solve_beam(env, game, beam_width=BEAM, max_depth=DEPTH, timeout_secs=TIMEOUT, use_click=use_click)
dt = time.time() - t0

if sol is None:
    print(f"FAILED L{TARGET} after {dt:.0f}s", flush=True)
    sys.exit(1)

def mv_repr(a):
    act, data = a if isinstance(a,tuple) else (a, None)
    n = ACTION_NAMES.get(act, str(act))
    if data:
        return f"{n}@{data.get('x','?')},{data.get('y','?')}"
    return n
names = [mv_repr(a) for a in sol]
print(f"SOLVED L{TARGET} in {len(sol)} moves, {dt:.0f}s", flush=True)
print(f"MOVES: {names}", flush=True)

# Emit Python list suitable for KNOWN dict
py = []
for a in sol:
    act, data = a if isinstance(a,tuple) else (a, None)
    n = ACTION_NAMES.get(act, str(act))
    if data:
        py.append(f"({n}, {{'x': {data['x']}, 'y': {data['y']}}})")
    else:
        py.append(n)
print(f"PYLIST: [{', '.join(py)}]", flush=True)

with open(f'/tmp/sk48_runs/L{TARGET}_sol.json','w') as f:
    json.dump({'level':TARGET,'moves':names,'pylist':py,'duration':dt}, f)

import pickle
sol_dump = []
for a in sol:
    act, data = a if isinstance(a,tuple) else (a, None)
    sol_dump.append((int(act.value), data))
with open(f'/tmp/sk48_runs/L{TARGET}_sol.pkl','wb') as f:
    pickle.dump(sol_dump, f)

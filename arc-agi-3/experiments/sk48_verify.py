#!/usr/bin/env python3
"""Verify KNOWN[0..N] replay correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
from arc_agi import Arcade
from sk48_solve_final import KNOWN

N = int(os.environ.get('N', '5'))
arcade = Arcade()
env = arcade.make('sk48-41055498')
env.reset()
g = env._game
def drive(act_or_pair):
    if isinstance(act_or_pair, tuple):
        act, data = act_or_pair
    else:
        act, data = act_or_pair, None
    env.step(act, data)
    while g.ljprkjlji or g.pzzwlsmdt: env.step(act, data)
    while g.lgdrixfno>=0 and g.lgdrixfno<35: env.step(act, data)
for lv in range(N):
    if lv not in KNOWN:
        print(f"L{lv}: no KNOWN; at level_index={g.level_index}")
        break
    for a in KNOWN[lv]:
        drive(a)
    try:
        fd = env._last_frame
    except Exception:
        fd = None
    print(f"L{lv}: {len(KNOWN[lv])} moves -> level_index={g.level_index}, budget={g.qiercdohl}, won={getattr(g,'is_last_level',None)}, state={(fd.state.name if fd else '?')}")

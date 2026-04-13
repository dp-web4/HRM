#!/usr/bin/env python3
"""sk48 L4 probe: advance past L0-L3 using KNOWN, then inspect L4 state.

Prints: rails (head -> segments), active head, paired pairs (xpmcmtbcv),
targets above play area, walls/tracks, budget. Also emits L5/L6/L7 states
if we can reach them (using a dummy solve attempt? no, just print L4 first).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
from arc_agi import Arcade
from arcengine import GameAction
from sk48_solve_final import KNOWN, ACTION_NAMES

def drive(env, game, act):
    env.step(act, None)
    while game.ljprkjlji or game.pzzwlsmdt:
        env.step(act, None)
    while game.lgdrixfno >= 0 and game.lgdrixfno < 35:
        env.step(act, None)

def inspect(game, label):
    print(f"\n========== {label} ==========")
    print(f"level_index={game.level_index}  budget={game.qiercdohl}  init_budget={game.vhzjwcpmk}")
    active = game.vzvypfsnt
    print(f"active head: ({active.x},{active.y})")
    print(f"paired (xpmcmtbcv): {[((h.x,h.y),(r.x,r.y)) for h,r in game.xpmcmtbcv.items()]}")
    print(f"rails (mwfajkguqx):")
    for h in sorted(game.mwfajkguqx.keys(), key=lambda s:(s.x,s.y)):
        segs = game.mwfajkguqx[h]
        seg_pos = [(s.x,s.y) for s in segs]
        print(f"  ({h.x},{h.y}) -> {seg_pos}")
    game.gvtmoopqgy()
    for h, r in game.xpmcmtbcv.items():
        rt = game.vjfbwggsd.get(r, [])
        ut = game.vjfbwggsd.get(h, [])
        print(f"  pair ({h.x},{h.y})->({r.x},{r.y})")
        print(f"    ref colors = {[int(t.pixels[1,1]) for t in rt]}")
        print(f"    upper colors = {[int(t.pixels[1,1]) for t in ut]}")
    targets = sorted([(t.x,t.y,int(t.pixels[1,1])) for t in game.vbelzuaian if t.y < 53])
    print(f"upper targets ({len(targets)}): {targets}")
    # tracks
    tracks = sorted([(s.x,s.y) for s in game.pptqisyill if hasattr(s,'name') and 'track' in str(getattr(s,'name','')).lower()][:30])
    # look at sprite type tags instead
    try:
        from arcengine import IRKEOBNGYH  # probably not
    except Exception:
        pass
    tr = []
    for s in game.pptqisyill:
        cls = type(s).__name__
        if 'rk' in cls.lower() or 'track' in cls.lower():
            tr.append((s.x,s.y))
    if tr:
        print(f"tracks ({len(tr)}): {sorted(tr)[:40]}")
    # walls: iterate and look for wall sprites
    walls=[]
    for s in game.pptqisyill:
        cls = type(s).__name__
        if 'wall' in cls.lower():
            walls.append((s.x,s.y))
    if walls:
        print(f"walls ({len(walls)}): {sorted(walls)[:40]}")

def main():
    arcade = Arcade()
    env = arcade.make('sk48-41055498')
    fd = env.reset()
    game = env._game

    # Play L0..L3 from KNOWN
    for lv in range(4):
        if lv not in KNOWN:
            print("missing known for", lv); return
        for a in KNOWN[lv]:
            drive(env, game, a)
        print(f"After L{lv}: level_index={game.level_index}")

    inspect(game, f"L{game.level_index} (L4)")

if __name__ == '__main__':
    main()

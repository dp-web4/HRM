#!/usr/bin/env python3
"""Inspect L4..L7 by using set_level (inspection only, separate env)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
from arc_agi import Arcade

def inspect(game, lv):
    print(f"\n===== L{lv} =====")
    print(f"budget={game.qiercdohl} init={game.vhzjwcpmk}")
    active = game.vzvypfsnt
    print(f"active head: ({active.x},{active.y})")
    print(f"pairs: {[((h.x,h.y),(r.x,r.y)) for h,r in game.xpmcmtbcv.items()]}")
    print("rails:")
    for h in sorted(game.mwfajkguqx.keys(), key=lambda s:(s.x,s.y)):
        segs = game.mwfajkguqx[h]
        print(f"  ({h.x},{h.y}): {[(s.x,s.y) for s in segs]}")
    game.gvtmoopqgy()
    for h, r in game.xpmcmtbcv.items():
        rt = game.vjfbwggsd.get(r, [])
        ut = game.vjfbwggsd.get(h, [])
        print(f"  pair ({h.x},{h.y})->({r.x},{r.y}) ref={[int(t.pixels[1,1]) for t in rt]} upper={[int(t.pixels[1,1]) for t in ut]}")
    targets = sorted([(t.x,t.y,int(t.pixels[1,1])) for t in game.vbelzuaian if t.y<53])
    print(f"upper targets ({len(targets)}): {targets}")
    # Track sprites: find class names
    by_cls = {}
    for s in game.pptqisyill:
        cn = type(s).__name__
        by_cls.setdefault(cn, []).append((s.x, s.y))
    for cn, ps in sorted(by_cls.items()):
        if len(ps) <= 20:
            print(f"  {cn} ({len(ps)}): {sorted(ps)}")
        else:
            print(f"  {cn} ({len(ps)})")

arcade = Arcade()
env = arcade.make('sk48-41055498')
env.reset()
g = env._game
levels = g._levels if hasattr(g,'_levels') else None
print(f"num levels: {len(levels) if levels else '?'}")
for i in range(min(8, len(levels) if levels else 8)):
    try:
        g.set_level(i)
        inspect(g, i)
    except Exception as e:
        print(f"L{i} failed: {e}")

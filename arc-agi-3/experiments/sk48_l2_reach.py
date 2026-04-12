#!/usr/bin/env python3
"""BFS-count reachable states from L2 start with the 4 movement actions."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

UP,DN,LT,RT = GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4
N={UP:'U',DN:'D',LT:'L',RT:'R'}
L0 = [UP, UP, UP, RT, RT, RT, RT, LT, DN, DN, RT, LT, UP, RT]
L1 = [UP,UP,RT,RT,RT,RT,UP,LT,LT,UP,RT,RT,DN,DN,RT,UP,RT,LT,LT,UP,RT,RT,LT,LT,UP,RT,RT]
a=Arcade(); env=a.make('sk48-41055498'); env.reset(); g=env._game
def drive(x):
    env.step(x)
    while g.ljprkjlji or g.pzzwlsmdt: env.step(x)
    while 0 <= g.lgdrixfno < 35: env.step(x)
print("starting L0", flush=True)
for x in L0: drive(x)
print("L0 done, L1 start", flush=True)
for x in L1: drive(x)
print("L1 done, at L2", flush=True)

def key():
    h = g.vzvypfsnt
    segs = g.mwfajkguqx[h]
    tgts = tuple(sorted((t.x,t.y,int(t.pixels[1,1])) for t in g.vbelzuaian))
    return (h.x, h.y, len(segs), tgts)

cur = []
def goto(target):
    global cur
    cp = 0
    m = min(len(cur), len(target))
    while cp < m and cur[cp] == target[cp]: cp += 1
    while len(cur) > cp:
        g.uqclctlhyh(); cur.pop()
    for i in range(cp, len(target)):
        drive(target[i]); cur.append(target[i])
    g.qiercdohl = 196  # always restore full budget for search

import time
t0 = time.time()
goto([])
visited = {key(): ()}
frontier = [()]
depth = 0
# Record best heuristic progress
from collections import Counter
ref_colors = [int(t.pixels[1,1]) for t in g.vjfbwggsd[g.xpmcmtbcv[g.vzvypfsnt]]]
print(f"ref: {ref_colors}")
wins = []
while frontier and depth < 40:
    if time.time() - t0 > 200:
        print("TIMEOUT"); break
    new_frontier = []
    for path in frontier:
        goto(list(path))
        for act in [UP,DN,LT,RT]:
            hist = len(g.seghobzez)
            bud_save = g.qiercdohl
            drive(act); cur.append(act)
            if len(g.seghobzez) > hist:
                g.gvtmoopqgy()
                upper = [int(v.pixels[1,1]) for v in g.vjfbwggsd[g.vzvypfsnt]]
                if upper[:len(ref_colors)] == ref_colors:
                    print(f"WIN at depth {depth+1}: {[N[a] for a in cur]}")
                    wins.append(tuple(cur))
                k = key()
                if k not in visited:
                    visited[k] = tuple(cur)
                    new_frontier.append(tuple(cur))
                g.uqclctlhyh(); cur.pop()
                g.qiercdohl = bud_save
            else:
                cur.pop()
                g.qiercdohl = bud_save
    frontier = new_frontier
    depth += 1
    print(f"depth {depth}: {len(frontier)} new, {len(visited)} total, t={time.time()-t0:.1f}s", flush=True)
    if depth >= 30: break

print(f"\nTotal reachable: {len(visited)}")
print(f"Wins found: {len(wins)}")

# Dump all visited states — are any of them wins we missed?
goto([])
won_states = 0
for k, path in visited.items():
    goto(list(path))
    g.gvtmoopqgy()
    upper = [int(v.pixels[1,1]) for v in g.vjfbwggsd[g.vzvypfsnt]]
    if upper[:4] == ref_colors:
        won_states += 1
print(f"States matching ref: {won_states}")

# Show unique upper visit lists seen
uppers = set()
for k, path in list(visited.items())[:50]:
    goto(list(path))
    g.gvtmoopqgy()
    u = tuple(int(v.pixels[1,1]) for v in g.vjfbwggsd[g.vzvypfsnt])
    uppers.add(u)
print(f"Unique upper visit lists: {sorted(uppers)}")
# Check if any win state
for k, path in visited.items():
    # Would need to check win from this state
    pass

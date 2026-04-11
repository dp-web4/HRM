#!/usr/bin/env python3
"""g50t L5 solver — replay-based exploration."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
PHASE_ACT = GameAction.ACTION5

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
DIR = {'U': UP, 'D': DOWN, 'L': LEFT, 'R': RIGHT}

L0_L4 = {
    0: 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT',
    1: 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT',
    2: 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP',
    3: 'DOWN DOWN RIGHT DOWN UNDO DOWN DOWN RIGHT RIGHT UP UP RIGHT RIGHT DOWN DOWN DOWN UNDO LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT LEFT LEFT LEFT',
    4: 'UP DOWN DOWN RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT LEFT DOWN LEFT LEFT LEFT LEFT LEFT UP UP',
}

def make_l5():
    arc = Arcade()
    env = arc.make('g50t-5849a774')
    env.reset()
    for lv in range(5):
        for name in L0_L4[lv].split():
            env.step(INT_TO_GA[NAME_TO_INT[name]])
    return env

def pos(env):
    p = env._game.vgwycxsxjz.dzxunlkwxt
    return (p.x, p.y)

def goal(env):
    g = env._game.vgwycxsxjz.whftgckbcu
    return (g.x+1, g.y+1)

def won(env):
    return env._game.level_index > 5

def lost(env):
    return env._game.vgwycxsxjz.zvuxrhnlcb

def run_path(env, path_str):
    """Run a path string like 'LLLDDD'. Return (final_pos, won, lost)."""
    for d in path_str:
        env.step(DIR[d])
        if lost(env): return pos(env), False, True
        if won(env): return pos(env), True, False
    return pos(env), won(env), lost(env)

print("=== g50t L5 Solver ===\n")

# Map reachable from (55,31) in phase 1
print("Phase 1 reachable (from (55,31)):")
paths_to_try = [
    '', 'L', 'LL', 'LLL', 'D', 'DD', 'DDD',
    'U', 'UU', 'UUU', 'UUUU',
    'LD', 'LLD', 'LLLD', 'LDD', 'LLDD',
    'LU', 'LLU', 'LLLU', 'LUU', 'LLUU',
    'LLLDD', 'LLLDDD', 'LLLDDDD',
    'LLLUU', 'LLLUUU', 'LLLUUUU',
    'LLLLUUU', 'LLLLLUUU', 'LLLLLLUUU', 'LLLLLLLUUU',
    'LLLD', 'LLLLD', 'LLLLLD', 'LLLLLLD',
    'LLUUUUU', 'LLUUUU', 'LLUUU',
]

seen = {}
for seq in paths_to_try:
    e = make_l5()
    p, w, l = run_path(e, seq)
    if p not in seen or len(seq) < len(seen[p]):
        seen[p] = seq
        tag = ' DEAD' if l else (' WIN' if w else '')
        print(f"  {seq:20s} → {p}{tag}")

print(f"\nAll reachable: {sorted(seen.keys())}")
print(f"Goal: {goal(make_l5())}")

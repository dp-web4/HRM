#!/usr/bin/env python3
"""Explore g50t levels interactively via SDK."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
STEP = 6

def play_sequence(env, seq_str, fd=None):
    """Play a sequence of moves, return fd after each."""
    if fd is None:
        fd = env.reset()
    actions = seq_str.strip().split()
    for i, name in enumerate(actions):
        a = NAME_TO_INT[name]
        prev_lc = fd.levels_completed
        fd = env.step(INT_TO_GA[a])
        level_up = fd.levels_completed > prev_lc
        sym = '★' if level_up else '·'
        if level_up or i == len(actions) - 1:
            print(f"  {sym} step {i+1}: {name:5s} | L={fd.levels_completed} state={fd.state.name}")
    return fd

def extract_state(env):
    """Extract game state details."""
    game = env._game
    lc = game.vgwycxsxjz
    player = lc.dzxunlkwxt
    goal = lc.whftgckbcu
    arena = lc.afbbgvkpip

    print(f"\n  Player: ({player.x}, {player.y})")
    print(f"  Goal: ({goal.x}, {goal.y})")
    print(f"  Start: ({lc.yugzlzepkr}, {lc.vgpdqizwwm})")
    print(f"  areahjypvy: {list(lc.areahjypvy)}")
    print(f"  undo_count: {lc.rlazdofsxb}")
    print(f"  indicators: {len(lc.drofvwhbxb)}")
    print(f"  ghosts: {len(lc.rloltuowth)}")

    obstacles = lc.uwxkstolmf
    modifiers = lc.hamayflsib
    print(f"  obstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"    obs[{i}]: ({obs.x},{obs.y}) vis={obs.is_visible}")
    print(f"  modifiers: {len(modifiers)}")
    for i, mod in enumerate(modifiers):
        pc = mod.nexhtmlmxh
        obs_targets = [(o.x,o.y) for o in pc.ytztewxdin] if pc else []
        print(f"    mod[{i}]: ({mod.x},{mod.y}) rot={mod.rotation} → {obs_targets}")

    # Map walkable grid
    walkable = set()
    obs_cells = set()
    orig_x, orig_y = player.x, player.y
    for px in range(arena.x - 6, arena.x + arena.width + 6, STEP):
        for py in range(arena.y - 6, arena.y + arena.height + 6, STEP):
            player.set_position(px, py)
            if lc.xvkyljflji(player, arena):
                if lc.vjpujwqrto(player):
                    obs_cells.add((px, py))
                else:
                    walkable.add((px, py))
    player.set_position(orig_x, orig_y)
    print(f"  walkable: {len(walkable)}, obs_blocked: {len(obs_cells)}")

    # Show grid
    all_cells = walkable | obs_cells
    if all_cells:
        xs = sorted(set(p[0] for p in all_cells))
        ys = sorted(set(p[1] for p in all_cells))
        print(f"\n  Grid ({len(xs)}x{len(ys)}):")
        header = "     " + "".join(f"{x:3d}" for x in xs)
        print(header)
        for y in ys:
            row = f"  {y:3d} "
            for x in xs:
                if (x, y) == (player.x, player.y):
                    row += " P "
                elif (x, y) == (goal.x + 1, goal.y + 1):
                    row += " G "
                elif (x, y) in obs_cells:
                    row += " # "
                elif any((x, y) == (mod.x, mod.y) for mod in modifiers):
                    row += " M "
                elif (x, y) in walkable:
                    row += " . "
                else:
                    row += "   "
            print(row)

    return lc

# Solutions from previous session
L1 = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT LEFT LEFT UP'
L2 = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT'
L3 = 'UP UP RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN UP DOWN UP DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT'

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()

print("=== L1 ===")
fd = play_sequence(env, L1, fd)
print(f"After L1: levels_completed={fd.levels_completed}, state={fd.state.name}")
if fd.levels_completed >= 1:
    print("\n=== L1 → L2 state ===")
    extract_state(env)

print("\n=== L2 ===")
fd = play_sequence(env, L2, fd)
print(f"After L2: levels_completed={fd.levels_completed}, state={fd.state.name}")
if fd.levels_completed >= 2:
    print("\n=== L2 → L3 state ===")
    extract_state(env)

print("\n=== L3 ===")
fd = play_sequence(env, L3, fd)
print(f"After L3: levels_completed={fd.levels_completed}, state={fd.state.name}")
if fd.levels_completed >= 3:
    print("\n=== L3 → L4 state ===")
    lc4 = extract_state(env)

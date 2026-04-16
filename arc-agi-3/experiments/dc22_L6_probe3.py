#!/usr/bin/env python3
"""L6: test clicking jpug-bjuk (c) and sprite-6 (f) — the two visible buttons."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells, save_frame

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

def summarize(game, label):
    p = game.fdvakicpimr
    parents = player_reachable_cells(game)
    visible_clicks = []
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags and s.is_visible and s.interaction != InteractionMode.REMOVED:
            visible_clicks.append((s.name, s.x, s.y))
    print(f"\n-- {label} --")
    print(f"  player=({p.x},{p.y}) reach={len(parents)}")
    print(f"  visible clicks: {visible_clicks}")
    # itki sprites
    itkis = [(s.name, s.x, s.y) for s in game.current_level.get_sprites()
             if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED]
    print(f"  itki: {itkis}")
    # reach bboxes
    if parents:
        xs = sorted({x for x,_ in parents})
        ys = sorted({y for _,y in parents})
        print(f"  reach bbox: x {xs[0]}..{xs[-1]} y {ys[0]}..{ys[-1]}")

def setup(arcade):
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    with open(f"{VIS}/solutions.json") as f:
        raw = json.load(f)
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4,
          6: GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    return env, am

def main():
    arcade = Arcade(operation_mode='offline')

    # Test 1: click jpug-bjuk (c) at center (51, 25)
    env, am = setup(arcade)
    game = env._game
    summarize(game, "start")
    # jpug-bjuk at (45,23) size 13x5 → center (51,25)
    r = env.step(am[6], data={'x': 51, 'y': 25})
    summarize(game, "after click c(51,25)")
    save_frame(env.observation_space.frame, f"{VIS}/L6_after_c.png")

    # Test 2: click sprite-6 (f) at center (56, 8)
    env, am = setup(arcade)
    game = env._game
    r = env.step(am[6], data={'x': 56, 'y': 8})
    summarize(game, "after click f(56,8)")
    save_frame(env.observation_space.frame, f"{VIS}/L6_after_f.png")

    # Test 3: click both
    env, am = setup(arcade)
    game = env._game
    env.step(am[6], data={'x': 51, 'y': 25})
    env.step(am[6], data={'x': 56, 'y': 8})
    summarize(game, "after c then f")

    # Test 4: click c multiple times
    env, am = setup(arcade)
    game = env._game
    for i in range(4):
        env.step(am[6], data={'x': 51, 'y': 25})
        summarize(game, f"after c x{i+1}")

if __name__ == "__main__":
    main()

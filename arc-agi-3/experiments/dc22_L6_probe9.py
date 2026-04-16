#!/usr/bin/env python3
"""Grid-scan: what supports (30, 52)? Cycle sprites and see."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

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

def check_support(game, x, y):
    p = game.fdvakicpimr
    ox, oy = p.x, p.y
    p.set_position(x, y)
    r = game.uxwpppoljm(x, y, p)
    p.set_position(ox, oy)
    return r

def main():
    arcade = Arcade(operation_mode='offline')
    am = {1:GameAction.ACTION1, 2:GameAction.ACTION2, 3:GameAction.ACTION3, 4:GameAction.ACTION4, 6:GameAction.ACTION6}

    # Under each c/f combo, test support at each (x,y) in the critical region
    for nc in range(2):
        for nf in range(6):
            env, _ = setup(arcade)
            game = env._game
            for _ in range(nc):
                env.step(am[6], data={'x':51,'y':25})
            for _ in range(nf):
                env.step(am[6], data={'x':56,'y':8})
            # Check x in 26..40 y in 48..62 — bridge zone
            print(f"\nc={nc} f={nf} support map (x=0..40 y=44..62):")
            print("     " + ''.join(f'{x:2d}'[-1] for x in range(0,41,2)))
            for y in range(44, 63, 2):
                row = f"y{y:2d}: "
                for x in range(0, 41, 2):
                    s = check_support(game, x, y)
                    if s is None:
                        row += ' '
                    else:
                        # one-char tag identifier
                        name = s.name
                        row += '.'
                print(row)

if __name__ == "__main__":
    main()

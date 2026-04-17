#!/usr/bin/env python3
"""Try L6 solve: walk right, flip gravity, try various paths."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.setrecursionlimit(50000)

ARCSAGE = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/arc-agi-3/experiments")
if (ARCSAGE / "environment_files" / "bp35").exists():
    os.chdir(ARCSAGE)

from arc_agi import Arcade
from arcengine import GameAction

TRACE_IN = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/bp35/run_legitimate_chain/run.json")
ACT_MAP = {"UP": GameAction.ACTION1, "DOWN": GameAction.ACTION2,
           "LEFT": GameAction.ACTION3, "RIGHT": GameAction.ACTION4,
           "CLICK": GameAction.ACTION6, "UNDO": GameAction.ACTION7}

def status(env, game, label):
    scene = game.oztjzzyqoek
    try:
        px, py = scene.twdpowducb.qumspquyus
    except Exception:
        px, py = (-1, -1)
    print(f"[{label}] L{scene.qswcochjodb} P=({px},{py}) gravUp={scene.vivnprldht}")
    return scene.qswcochjodb, (px, py), scene.vivnprldht

def vp_click(player_y, grid_x, grid_y):
    """Compute viewport coords for clicking grid (grid_x, grid_y) given current player_y.
       camera_y = player.grid_y * 6 - 36; vp_x = grid_x*6; vp_y = grid_y*6 - camera_y.
    """
    cam_y = player_y * 6 - 36
    return grid_x * 6, grid_y * 6 - cam_y

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    for s in json.load(open(TRACE_IN))["steps"]:
        if s["step"] > 143: break
        a = s["action"]
        if a == "CLICK": env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif a != "CLICK_OOB_SKIPPED": env.step(ACT_MAP[a])
    env.step(ACT_MAP["LEFT"])  # L5 win → on L6
    status(env, game, "L6 start")

    scene = game.oztjzzyqoek
    def step_and_report(label, fn):
        fn()
        status(env, game, label)

    # 5 RIGHTs to (8,23)
    for i in range(5):
        step_and_report(f"RIGHT #{i+1}", lambda: env.step(ACT_MAP["RIGHT"]))

    # Click (6,22) viewport coords
    _, (px, py), _ = status(env, game, "before click (6,22)")
    vx, vy = vp_click(py, 6, 22)
    print(f"click vp=({vx},{vy})")
    step_and_report("CLICK(6,22)->flip grav+destroy g", lambda: env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}))

    # Now grav DOWN, player ??? Let's see
    # Try walk LEFT 3x to reach (5,31)
    for i in range(3):
        step_and_report(f"LEFT #{i+1}", lambda: env.step(ACT_MAP["LEFT"]))

    # Click (4,31)
    _, (px, py), _ = status(env, game, "before click (4,31)")
    vx, vy = vp_click(py, 4, 31)
    print(f"click vp=({vx},{vy})")
    step_and_report("CLICK(4,31)->flip grav+destroy g", lambda: env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}))

    # Click (6,22) might not exist, skip. Try click (8,1) to flip gravity again
    # Actually better: plan depends on current pos. Just dump everything.

    # Dump area around player
    _, (px, py), _ = status(env, game, "after 4,31")
    print("nearby tiles:")
    for y in range(max(0,py-4), min(45,py+5)):
        row = []
        for x in range(max(0,px-5), min(12,px+5)):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            names = {e.name for e in ents}
            if (x,y)==(px,py): ch = "P"
            elif "fjlzdjxhant" in names: ch = "+"
            elif "lrpkmzabbfa" in names: ch = "g"
            elif "ubhhgljbnpu" in names or "hzusueifitk" in names: ch = "v"
            elif "xcjjwqfzjfe" in names: ch = "#"
            elif "oonshderxef" in names: ch = "2"
            elif not names: ch = "."
            else: ch = "?"
            row.append(ch)
        print(f" y={y:2}: {''.join(row)}")

if __name__ == "__main__":
    main()

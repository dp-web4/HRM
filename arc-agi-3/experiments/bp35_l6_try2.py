#!/usr/bin/env python3
"""Try L6 solve v2: destroy all 3 gravity gems, then walk through chamber."""
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

def status(scene, label):
    try: px, py = scene.twdpowducb.qumspquyus
    except: px, py = (-1,-1)
    cam_y = scene.camera.rczgvgfsfb[1]
    print(f"[{label}] L{scene.qswcochjodb} P=({px},{py}) gravUp={scene.vivnprldht} camY={cam_y}")
    return px, py

def vp_for(px, py, scene, target):
    cam_y = scene.camera.rczgvgfsfb[1]
    return target[0]*6, target[1]*6 - cam_y

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
    env.step(ACT_MAP["LEFT"])  # L5 → L6
    scene = game.oztjzzyqoek
    status(scene, "L6 start")

    # 5 RIGHTs
    for i in range(5):
        env.step(ACT_MAP["RIGHT"])
        status(scene, f"R{i+1}")

    # Click (6,22)
    px, py = status(scene, "pre CLICK (6,22)")
    vx, vy = vp_for(px, py, scene, (6, 22))
    print(f"  vp=({vx},{vy})")
    env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    status(scene, "post CLICK (6,22)")

    # Click (4,31)
    px, py = status(scene, "pre CLICK (4,31)")
    vx, vy = vp_for(px, py, scene, (4, 31))
    print(f"  vp=({vx},{vy})")
    env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    status(scene, "post CLICK (4,31)")

    # Click (8,1)
    px, py = status(scene, "pre CLICK (8,1)")
    vx, vy = vp_for(px, py, scene, (8, 1))
    print(f"  vp=({vx},{vy})")
    env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    status(scene, "post CLICK (8,1)")

    # LEFT repeatedly
    for i in range(10):
        fd = env.step(ACT_MAP["LEFT"])
        scene = game.oztjzzyqoek  # re-fetch in case level advanced
        lvl_now = scene.qswcochjodb
        px, py = status(scene, f"L{i+1}")
        print(f"    fd.levels_completed = {getattr(fd, 'levels_completed', None)}, "
              f"fd.state = {getattr(fd, 'state', None)}, won={scene.nkuphphdgrp}")
        if lvl_now > 6:
            print(f"*** ADVANCED to L{lvl_now}! ***")
            break
        if scene.nkuphphdgrp:
            print(f"    WIN flag set; try a RIGHT to drain animations...")
            fd2 = env.step(ACT_MAP["RIGHT"])
            print(f"    after RIGHT: lvl={scene.qswcochjodb}, fd.state={fd2.state}, fd.levels_completed={fd2.levels_completed}")
            if scene.qswcochjodb > 6:
                print(f"*** ADVANCED to L{scene.qswcochjodb}! ***")
                break
        # Also check if at gem location, what tiles
        ents = scene.hdnrlfmyrj.jhzcxkveiw(px, py)
        print(f"    tiles at P=({px},{py}): {[e.name for e in ents]}")
        ents231 = scene.hdnrlfmyrj.jhzcxkveiw(2, 31)
        print(f"    tiles at (2,31): {[e.name for e in ents231]}")

if __name__ == "__main__":
    main()

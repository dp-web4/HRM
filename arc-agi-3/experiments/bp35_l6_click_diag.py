#!/usr/bin/env python3
"""Diagnose why second click didn't flip gravity."""
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
    for _ in range(5): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    px, py = scene.twdpowducb.qumspquyus
    print(f"Before click (6,22). P=({px},{py}) grav_up={scene.vivnprldht}")
    cam_y = scene.camera.rczgvgfsfb[1]
    print(f"camera_y = {cam_y}")

    # Check what's at (6,22) right now
    ents = scene.hdnrlfmyrj.jhzcxkveiw(6, 22)
    print(f"  tiles at (6,22): {[e.name for e in ents]}")
    vp = (6*6, 22*6 - cam_y)
    print(f"  vp = {vp}; maps to grid {scene.hdnrlfmyrj.hyntnfvpgl(vp[0], vp[1] + cam_y)}")

    env.step(ACT_MAP["CLICK"], data={"x": vp[0], "y": vp[1]})
    px, py = scene.twdpowducb.qumspquyus
    print(f"\nAfter click (6,22). P=({px},{py}) grav_up={scene.vivnprldht}")
    cam_y = scene.camera.rczgvgfsfb[1]
    print(f"camera_y = {cam_y}")
    ents = scene.hdnrlfmyrj.jhzcxkveiw(6, 22)
    print(f"  tiles at (6,22): {[e.name for e in ents]}")

    # 3 LEFT
    for _ in range(3): env.step(ACT_MAP["LEFT"])
    px, py = scene.twdpowducb.qumspquyus
    print(f"\nAfter 3 LEFT. P=({px},{py}) grav_up={scene.vivnprldht}")
    cam_y = scene.camera.rczgvgfsfb[1]
    print(f"camera_y = {cam_y}")

    ents = scene.hdnrlfmyrj.jhzcxkveiw(4, 31)
    print(f"  tiles at (4,31): {[e.name for e in ents]}")

    vp = (4*6, 31*6 - cam_y)
    print(f"  vp for (4,31) = {vp}; hyntnfvpgl = {scene.hdnrlfmyrj.hyntnfvpgl(vp[0], vp[1] + cam_y)}")
    env.step(ACT_MAP["CLICK"], data={"x": vp[0], "y": vp[1]})
    px, py = scene.twdpowducb.qumspquyus
    print(f"\nAfter click (4,31). P=({px},{py}) grav_up={scene.vivnprldht}")
    ents = scene.hdnrlfmyrj.jhzcxkveiw(4, 31)
    print(f"  tiles at (4,31): {[e.name for e in ents]}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deep probe of L7 col 2 and col 8,9."""
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

def vp_for(scene, target):
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
    env.step(ACT_MAP["LEFT"])
    for _ in range(5): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    for tgt in [(6,22), (4,31), (8,1)]:
        vx, vy = vp_for(scene, tgt)
        env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    for _ in range(6): env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    assert scene.qswcochjodb == 7
    short = {
        "xcjjwqfzjfe": "#", "qclfkhjnaac": "X", "fjlzdjxhant": "+",
        "lrpkmzabbfa": "g", "yuuqpmlxorv": "1", "oonshderxef": "2",
        "ubhhgljbnpu": "v", "hzusueifitk": "u", "etlsaqqtjvn": "y",
        "player_right": "P",
    }
    for x in [2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"\n=== L7 col x={x} ===")
        for y in range(0, 30):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            names = [e.name for e in ents]
            tag = "".join(short.get(n, "?") for n in names) or "."
            print(f"  y={y:2}: {tag}  {names}")

if __name__ == "__main__":
    main()

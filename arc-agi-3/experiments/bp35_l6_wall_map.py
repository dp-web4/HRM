#!/usr/bin/env python3
"""Render L6 as a walkable-vs-blocking map."""
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
    env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    px, py = scene.twdpowducb.qumspquyus

    W, H = 11, 45
    print(f"L{scene.qswcochjodb} player=({px},{py}) gravUp={scene.vivnprldht}\n")
    print("    " + "".join(f"{x%10}" for x in range(W)))
    for y in range(H):
        row = []
        for x in range(W):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            names = {e.name for e in ents}
            if (x,y) == (px,py):
                ch = "P"
            elif "fjlzdjxhant" in names:
                ch = "+"  # WIN GEM
            elif "lrpkmzabbfa" in names:
                ch = "g"
            elif "ubhhgljbnpu" in names or "hzusueifitk" in names:
                ch = "v"  # spike (death)
            elif "qclfkhjnaac" in names:
                ch = "X"  # destructible ground
            elif "xcjjwqfzjfe" in names:
                ch = "#"  # wall (background, blocks movement)
            elif "oonshderxef" in names:
                ch = "2"  # passthrough
            elif "etlsaqqtjvn" in names:
                ch = "y"
            elif not names:
                ch = "."  # AIR / empty — walkable
            else:
                ch = "?"
            row.append(ch)
        print(f"{y:2}  {''.join(row)}")

if __name__ == "__main__":
    main()

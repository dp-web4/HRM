#!/usr/bin/env python3
"""Probe L6 columns 0 and 1 and 4 for connectivity to the + gem chamber."""
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

    cols_to_probe = [0, 1, 2, 3, 4, 6, 8]
    print("(col, y) -> names")
    for x in cols_to_probe:
        print(f"\n--- column x={x} ---")
        for y in range(45):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            names = [e.name for e in ents]
            if names:
                short = {
                    "xcjjwqfzjfe": "#",
                    "qclfkhjnaac": "X",
                    "fjlzdjxhant": "+",
                    "lrpkmzabbfa": "g",
                    "yuuqpmlxorv": "1",
                    "oonshderxef": "2",
                    "ubhhgljbnpu": "v",
                    "hzusueifitk": "u",
                    "etlsaqqtjvn": "y",
                    "player_right": "P",
                    "player_left": "P",
                }
                tags = "".join(short.get(n, "?") for n in names)
                print(f"  y={y:2}: {tags}  {names}")

if __name__ == "__main__":
    main()

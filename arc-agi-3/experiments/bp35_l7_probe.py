#!/usr/bin/env python3
"""Probe L7 after full L1-L6 winning chain."""
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

def play_through_l6(env, game):
    """Replay L1-L4 legacy, L5 LEFT, then L6 solve."""
    for s in json.load(open(TRACE_IN))["steps"]:
        if s["step"] > 143: break
        a = s["action"]
        if a == "CLICK": env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif a != "CLICK_OOB_SKIPPED": env.step(ACT_MAP[a])
    env.step(ACT_MAP["LEFT"])  # L5 win

    # L6 solve: R×5, C(6,22), C(4,31), C(8,1), L×6
    for _ in range(5): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    for tgt in [(6,22), (4,31), (8,1)]:
        vx, vy = vp_for(scene, tgt)
        env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    for _ in range(6): env.step(ACT_MAP["LEFT"])

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    play_through_l6(env, game)
    scene = game.oztjzzyqoek
    px, py = scene.twdpowducb.qumspquyus
    print(f"L{scene.qswcochjodb} P=({px},{py}) gravUp={scene.vivnprldht} camY={scene.camera.rczgvgfsfb[1]}")
    assert scene.qswcochjodb == 7, f"Expected L7, got L{scene.qswcochjodb}"

    # Render L7 full map
    print("\n=== L7 FULL GRID ===")
    print("    " + "".join(f"{x%10}" for x in range(11)))
    for y in range(45):
        row = []
        for x in range(11):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            names = {e.name for e in ents}
            if (x,y) == (px,py): ch = "P"
            elif "fjlzdjxhant" in names: ch = "+"
            elif "lrpkmzabbfa" in names: ch = "g"
            elif "ubhhgljbnpu" in names or "hzusueifitk" in names: ch = "v"
            elif "qclfkhjnaac" in names: ch = "X"
            elif "xcjjwqfzjfe" in names: ch = "#"
            elif "yuuqpmlxorv" in names: ch = "1"
            elif "oonshderxef" in names: ch = "2"
            elif "etlsaqqtjvn" in names: ch = "y"
            elif not names: ch = "."
            else: ch = "?"
            row.append(ch)
        print(f"{y:2}  {''.join(row)}")

    # List special tiles
    print("\n=== L7 special tiles ===")
    cat = {}
    for y in range(45):
        for x in range(11):
            for e in scene.hdnrlfmyrj.jhzcxkveiw(x, y):
                cat.setdefault(e.name, []).append((x,y))
    for n, locs in sorted(cat.items()):
        if n in ("xcjjwqfzjfe", "player_right"): continue
        print(f"  {n}: {locs}")

if __name__ == "__main__":
    main()

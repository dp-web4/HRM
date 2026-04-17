#!/usr/bin/env python3
"""Probe L8 after L1-L7 winning chain."""
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

def reach_l7(env, game):
    for s in json.load(open(TRACE_IN))["steps"]:
        if s["step"] > 143: break
        a = s["action"]
        if a == "CLICK": env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif a != "CLICK_OOB_SKIPPED": env.step(ACT_MAP[a])
    env.step(ACT_MAP["LEFT"])
    for _ in range(5): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    for tgt in [(6,22),(4,31),(8,1)]:
        vx, vy = vp_for(scene, tgt)
        env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    for _ in range(6): env.step(ACT_MAP["LEFT"])

def solve_l7(env, game):
    scene = game.oztjzzyqoek
    targets_initial = [(4,14),(5,14),(4,16),(5,16),(6,16),(6,19),(4,9)]
    for t in targets_initial:
        vx, vy = vp_for(scene, t); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    # flip (0,19)
    vx, vy = vp_for(scene, (0,19)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    for _ in range(3): env.step(ACT_MAP["RIGHT"])
    vx, vy = vp_for(scene, (0,21)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    for _ in range(2): env.step(ACT_MAP["RIGHT"])
    for _ in range(4): env.step(ACT_MAP["LEFT"])
    vx, vy = vp_for(scene, (0,18)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    vx, vy = vp_for(scene, (0,17)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["RIGHT"])
    vx, vy = vp_for(scene, (0,16)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["RIGHT"])
    vx, vy = vp_for(scene, (0,15)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (5,10)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["RIGHT"])
    vx, vy = vp_for(scene, (5,10)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,14)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (6,9)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["RIGHT"])
    vx, vy = vp_for(scene, (6,9)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,13)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["RIGHT"])
    vx, vy = vp_for(scene, (7,8)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,12)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    vx, vy = vp_for(scene, (0,11)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    vx, vy = vp_for(scene, (0,10)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy}); scene = game.oztjzzyqoek

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    reach_l7(env, game)
    solve_l7(env, game)
    scene = game.oztjzzyqoek
    assert scene.qswcochjodb == 8, f"Expected L8, got {scene.qswcochjodb}"
    print(f"On L8: P={scene.twdpowducb.qumspquyus} grav_up={scene.vivnprldht} camY={scene.camera.rczgvgfsfb[1]}")

    # Dump L8
    print("\n=== L8 FULL GRID (x: 0-10, y: 0-44) ===")
    print("    " + "".join(f"{x%10}" for x in range(11)))
    for y in range(45):
        row = []
        for x in range(11):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            names = {e.name for e in ents}
            if (x,y) == scene.twdpowducb.qumspquyus: ch = "P"
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

    # Special tiles
    print("\n=== Special tiles ===")
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

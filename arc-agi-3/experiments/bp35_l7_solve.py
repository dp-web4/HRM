#!/usr/bin/env python3
"""L7 solve attempt: full 30+ action sequence to reach gem."""
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

def status(scene, lbl=""):
    try:
        p = scene.twdpowducb.qumspquyus
        g = scene.vivnprldht
    except Exception:
        p, g = ("?", "?")
    print(f"  [{lbl}] L{scene.qswcochjodb} P={p} grav_up={g}")

def do_click(env, game, target, lbl=""):
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, target)
    env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    scene = game.oztjzzyqoek
    status(scene, f"CLICK{target} {lbl}")
    return scene

def do_act(env, game, action, lbl=""):
    env.step(ACT_MAP[action])
    scene = game.oztjzzyqoek
    status(scene, f"{action} {lbl}")
    return scene

def reach_l7(env, game):
    for s in json.load(open(TRACE_IN))["steps"]:
        if s["step"] > 143: break
        a = s["action"]
        if a == "CLICK":
            env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif a != "CLICK_OOB_SKIPPED":
            env.step(ACT_MAP[a])
    env.step(ACT_MAP["LEFT"])
    for _ in range(5):
        env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    for tgt in [(6, 22), (4, 31), (8, 1)]:
        vx, vy = vp_for(scene, tgt)
        env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    for _ in range(6):
        env.step(ACT_MAP["LEFT"])

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    reach_l7(env, game)
    scene = game.oztjzzyqoek
    print(f"Start L7: P={scene.twdpowducb.qumspquyus} grav_up={scene.vivnprldht}")

    # Phase A: setup conversions from spawn
    do_click(env, game, (4, 14), "4,14 2->1")
    do_click(env, game, (5, 14), "5,14 2->1")
    do_click(env, game, (4, 16), "4,16 2->1")
    do_click(env, game, (5, 16), "5,16 2->1")
    do_click(env, game, (6, 16), "6,16 2->1")
    do_click(env, game, (6, 19), "6,19 2->1 (for (6,20) floor)")
    do_click(env, game, (4, 9), "4,9 2->1")
    # DON'T convert (7,8) yet — we need col 7 all-passthrough for fall to (7,4)

    # Phase B: spawn to (8,15) then (2,8) → (3,8) → (3,11) → (4,11) → (4,10)
    do_click(env, game, (0, 19), "flip UP->DOWN → (3,21)")
    do_act(env, game, "RIGHT")
    do_act(env, game, "RIGHT")
    do_act(env, game, "RIGHT")  # (6,21)
    do_click(env, game, (0, 21), "flip DOWN->UP → (6,20)")
    do_act(env, game, "RIGHT")  # (7,20)
    do_act(env, game, "RIGHT")  # (8,15)
    do_act(env, game, "LEFT")   # (7,13)
    do_act(env, game, "LEFT")   # (6,13)
    do_act(env, game, "LEFT")   # (5,13)
    do_act(env, game, "LEFT")   # (4,13)
    do_click(env, game, (0, 18), "flip UP->DOWN @ (4,13) stays (4,14)=1")
    do_act(env, game, "RIGHT")  # (5,13)
    do_act(env, game, "RIGHT")  # (6,15) via (6,14)=2,(6,15)=2,(6,16)=1
    do_act(env, game, "LEFT")   # (5,15) via (5,16)=1
    do_act(env, game, "LEFT")   # (4,15) via (4,16)=1
    do_click(env, game, (0, 17), "flip DOWN->UP @ (4,15) stays (4,14)=1")
    do_act(env, game, "LEFT")   # (3,15) via (3,14)#
    do_act(env, game, "LEFT")   # (2,8) via col 2 fall up
    do_act(env, game, "RIGHT")  # (3,8)

    # Phase C: (3,8) → col 7 → col 8 → col 9
    do_click(env, game, (0, 16), "flip UP->DOWN @ (3,8) → (3,11)")
    do_act(env, game, "RIGHT")  # (4,11)
    do_click(env, game, (0, 15), "flip DOWN->UP @ (4,11) → (4,10)")
    do_click(env, game, (5, 10), "5,10 1->2")
    do_act(env, game, "RIGHT")  # (5,8) via col 5 fall up
    do_click(env, game, (5, 10), "5,10 2->1 (stop fall DOWN at (5,9))")
    do_click(env, game, (0, 14), "flip UP->DOWN @ (5,8) → (5,9)")
    do_click(env, game, (6, 9), "6,9 1->2 (walk thru)")
    do_act(env, game, "RIGHT")  # (5,9) RIGHT (6,9)=2 fall → (6,11)
    do_click(env, game, (6, 9), "6,9 2->1 (stop col 6 fall UP)")
    do_click(env, game, (0, 13), "flip DOWN->UP @ (6,11) → (6,10)")
    do_act(env, game, "RIGHT")  # (6,10) RIGHT (7,10)=2 fall up col 7 → (7,4)
    # NOW convert (7,8) → 1 so flip DOWN stops at (7,7)
    do_click(env, game, (7, 8), "7,8 2->1 (col7 fall stop at (7,7))")
    do_click(env, game, (0, 12), "flip UP->DOWN @ (7,4) → (7,7) via (7,8)=1")
    do_act(env, game, "RIGHT")  # (7,7) → (8,7) via (8,8)#
    do_act(env, game, "RIGHT")  # (8,7) → (9,26) via col 9 fall

    # Phase D: navigate to gem
    do_act(env, game, "LEFT")   # (9,26) → (8,26)
    do_act(env, game, "LEFT")   # → (7,26)
    do_click(env, game, (0, 11), "flip DOWN->UP @ (7,26) → (7,23)")
    do_act(env, game, "LEFT")   # → (6,23)
    do_act(env, game, "LEFT")   # → (5,23)
    do_act(env, game, "LEFT")   # → (4,23)
    do_act(env, game, "LEFT")   # → (3,23)
    do_click(env, game, (0, 10), "flip UP->DOWN @ (3,23) → WIN!")

    scene = game.oztjzzyqoek
    print(f"\nFinal state: L{scene.qswcochjodb}, P={scene.twdpowducb.qumspquyus}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""L8 solve: y-spread chain to create platform at row 16, walk to (8,17), flip, win."""
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
    except Exception: p, g = ("?", "?")
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

def reach_l8(env, game):
    # L1-L4
    for s in json.load(open(TRACE_IN))["steps"]:
        if s["step"] > 143: break
        a = s["action"]
        if a == "CLICK": env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif a != "CLICK_OOB_SKIPPED": env.step(ACT_MAP[a])
    env.step(ACT_MAP["LEFT"])  # L5 win
    # L6 solve
    for _ in range(5): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    for tgt in [(6,22),(4,31),(8,1)]:
        vx, vy = vp_for(scene, tgt)
        env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    for _ in range(6): env.step(ACT_MAP["LEFT"])
    # L7 solve (from bp35_l7_solve.py logic)
    scene = game.oztjzzyqoek
    targets_initial = [(4,14),(5,14),(4,16),(5,16),(6,16),(6,19),(4,9)]
    for t in targets_initial:
        vx, vy = vp_for(scene, t); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,19)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    for _ in range(3): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,21)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    for _ in range(2): env.step(ACT_MAP["RIGHT"])
    for _ in range(4): env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,18)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,17)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,16)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,15)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (5,10)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (5,10)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,14)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (6,9)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (6,9)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,13)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (7,8)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,12)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,11)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,10)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    reach_l8(env, game)
    scene = game.oztjzzyqoek
    assert scene.qswcochjodb == 8, f"Expected L8, got {scene.qswcochjodb}"
    print(f"Start L8: P={scene.twdpowducb.qumspquyus} grav_up={scene.vivnprldht}")

    # L8 solve
    # Step A: y-spread chain to create platform at row 16.
    # C(3,18) → C(3,17) → C(3,16) → C(3,15) → C(4,15) → C(5,15) → C(6,15) → C(7,15) → C(8,15)
    do_click(env, game, (3,18), "y-spread start")
    do_click(env, game, (3,17), "spread up col 3")
    do_click(env, game, (3,16), "spread to (3,15)")
    do_click(env, game, (3,15), "chain to col 3 y=14, spawn (4,15)")
    do_click(env, game, (4,15), "spread (5,15)")
    do_click(env, game, (5,15), "spread (6,15)")
    do_click(env, game, (6,15), "spread (7,15)")
    do_click(env, game, (7,15), "spread (8,15) and (7,16)")
    do_click(env, game, (8,15), "spread (8,16)!")

    # Step B: walk RIGHT to trigger fall up col 5 to (5,17)
    do_act(env, game, "RIGHT")  # (3,32) → (4,32)
    do_act(env, game, "RIGHT")  # (4,32) → (5,32) → fall up col 5 to (5,17)

    # Step C: walk RIGHT chain to (8,17)
    do_act(env, game, "RIGHT")  # (5,17) → (6,17)
    do_act(env, game, "RIGHT")  # (6,17) → (7,17)
    do_act(env, game, "RIGHT")  # (7,17) → (8,17)

    # Step D: flip gravity and convert (8,18)
    do_click(env, game, (5,2), "flip g → DOWN")
    do_click(env, game, (8,18), "click 1 match → fall to (8,19)")

    # Step E: walk RIGHT into gem
    do_act(env, game, "RIGHT")  # (8,19) → (9,19) WIN

    scene = game.oztjzzyqoek
    print(f"\nFinal: L{scene.qswcochjodb}, P={scene.twdpowducb.qumspquyus if scene.qswcochjodb == 8 else 'ADV'}")
    return 0 if scene.qswcochjodb > 8 else 1

if __name__ == "__main__":
    sys.exit(main())

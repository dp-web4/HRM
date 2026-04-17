#!/usr/bin/env python3
"""L8 attempt 2: extensive y-spread from BOTH (3,18) and (2,29). See where this lands us."""
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
        died = scene.jrhqdvdwpsb
    except Exception: p, g, died = ("?","?",False)
    print(f"  [{lbl}] L{scene.qswcochjodb} P={p} grav={g} died={died}")

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
    # Reach L8 via the established chain.
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
    # L7
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
    env.step(ACT_MAP["RIGHT"]); env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["LEFT"]); env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,17)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    env.step(ACT_MAP["LEFT"]); env.step(ACT_MAP["LEFT"]); env.step(ACT_MAP["RIGHT"])
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
    env.step(ACT_MAP["RIGHT"]); env.step(ACT_MAP["RIGHT"])
    env.step(ACT_MAP["LEFT"]); env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,11)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    for _ in range(4): env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, (0,10)); env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    reach_l8(env, game)
    scene = game.oztjzzyqoek
    assert scene.qswcochjodb == 8
    print(f"L8 start: P={scene.twdpowducb.qumspquyus} grav_up={scene.vivnprldht}")

    # Y-spread chain 1: from (2,29) along row 29 to (9,29), then up via cols 8,9.
    # Goal: have y at (5,30),(6,30),(7,30) so falling from (5,32),(6,32),(7,32) lands at (5,31),(6,31),(7,31).
    # Also (8,30) y so col 8 stable too.

    do_click(env, game, (2,29), "y(2,29) -> spread row 29 + col 2")
    do_click(env, game, (3,29), "y(3,29) -> (4,29),(3,28),(3,30)")
    do_click(env, game, (4,29), "y(4,29) -> (5,29)")
    do_click(env, game, (5,29), "y(5,29) -> (6,29),(5,28),(5,30)")
    do_click(env, game, (6,29), "y(6,29) -> (7,29),(6,28),(6,30)")
    do_click(env, game, (7,29), "y(7,29) -> (8,29),(7,28),(7,30)")
    do_click(env, game, (8,29), "y(8,29) -> (9,29),(8,28),(8,30)")
    do_click(env, game, (9,29), "y(9,29) -> (9,28),(9,30)")

    # At this point, row 29-30 should have many y's. Walking on row 32 should now fall to row 31.
    # Walk RIGHT from (3,32):
    do_act(env, game, "RIGHT")  # (4,32)
    do_act(env, game, "RIGHT")  # (5,32) → falls to (5,31) (5,30)y stops
    do_act(env, game, "RIGHT")  # (6,31) (6,30)y stops
    do_act(env, game, "RIGHT")  # (7,31) (7,30)y stops
    do_act(env, game, "RIGHT")  # (8,31)# bounce or land?

    # Hmm (8,31)# wall. Try walk to (8,32) via different path.
    # If walking (7,31) → (8,31)# bounces, can't reach (8,32).

    # Instead try: don't walk on row 31. Use y-spread to climb higher.
    # From (7,31), click some y to spread further up.

    # Actually let me try clicking (7,30)y to spread to (7,31)y... player is at (7,31).
    do_click(env, game, (7,30), "click y(7,30) - removes (7,30), spawns (6,30)re,(8,30)re,(7,29)re,(7,31)y")
    # Now (7,31)y (player coexists). (7,30) empty.

    # Walk LEFT from (7,31)y to (6,31). What happens?
    do_act(env, game, "LEFT", "from (7,31)y LEFT to (6,31)?")

    # Try clicking more.
    do_click(env, game, (6,30), "click y(6,30) - spawns (6,31)y")
    do_act(env, game, "LEFT", "LEFT")
    do_act(env, game, "RIGHT", "RIGHT")

    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Manual try: reach upper chamber then gem via the discovered conversions."""
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

def do_click(env, game, target, label=""):
    scene = game.oztjzzyqoek
    vx, vy = vp_for(scene, target)
    before_p = scene.twdpowducb.qumspquyus
    before_g = scene.vivnprldht
    ents_before = [e.name for e in scene.hdnrlfmyrj.jhzcxkveiw(*target)]
    env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
    scene = game.oztjzzyqoek
    after_p = scene.twdpowducb.qumspquyus if scene.qswcochjodb == 7 else "LVL-ADVANCED"
    after_g = scene.vivnprldht
    ents_after = [e.name for e in scene.hdnrlfmyrj.jhzcxkveiw(*target)]
    print(f"  CLICK{target} vp=({vx},{vy}) {label}: P{before_p}→{after_p} grav{before_g}→{after_g} tile {ents_before}→{ents_after}")

def do_act(env, game, action, label=""):
    scene = game.oztjzzyqoek
    before_p = scene.twdpowducb.qumspquyus
    before_g = scene.vivnprldht
    env.step(ACT_MAP[action])
    scene = game.oztjzzyqoek
    if scene.qswcochjodb != 7:
        print(f"  {action} {label}: LEVEL ADVANCED to L{scene.qswcochjodb} !")
        return True
    after_p = scene.twdpowducb.qumspquyus
    after_g = scene.vivnprldht
    print(f"  {action} {label}: P{before_p}→{after_p} grav{before_g}→{after_g}")
    return False

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

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    reach_l7(env, game)
    scene = game.oztjzzyqoek
    print(f"On L7. P={scene.twdpowducb.qumspquyus} grav={scene.vivnprldht}")

    # Step 1: conversions at start
    do_click(env, game, (4,14), "4,14 2->1")
    do_click(env, game, (5,14), "5,14 2->1")
    do_click(env, game, (4,16), "4,16 2->1")
    do_click(env, game, (6,19), "6,19 2->1")
    do_click(env, game, (4,9), "4,9 2->1 (for later fall stop)")

    # Step 2: flip DOWN - need a g click. Use (0,19) (closest)
    do_click(env, game, (0,19), "flip gravity UP->DOWN")

    # Step 3: walk RIGHT x3 to (6,21)
    for i in range(3):
        if do_act(env, game, "RIGHT", f"#{i+1}"):
            return 0

    # Step 4: flip UP via click g. Use (0,21)
    do_click(env, game, (0,21), "flip DOWN->UP")

    # Step 5: walk RIGHT to (7,20) to (8,15) via col 8 fall up
    for i in range(2):
        if do_act(env, game, "RIGHT", f"#{i+1}"):
            return 0

    # Step 6: LEFT x4 to (4,13)
    for i in range(4):
        if do_act(env, game, "LEFT", f"#{i+1}"):
            return 0

    # Step 7: At (4,13) grav UP. Flip DOWN via click (0,18) g.
    do_click(env, game, (0,18), "flip UP->DOWN @ (4,13)")
    # With (4,14)=1, stays (4,13)

    # Step 8: walk RIGHT to (5,13). (5,14)=1 stops.
    do_act(env, game, "RIGHT", "5,13?")

    # Step 9: walk RIGHT to (6,13). Fall to (6,17) via (6,14-17).
    # Actually we need (6,16) to be 1 to stop at (6,15).
    do_click(env, game, (6,16), "6,16 2->1 (stop at 6,15)")
    do_click(env, game, (5,16), "5,16 2->1 (stop at 5,15)")

    # Step 10: walk RIGHT (6,13). fsvnqdbzrp falls grav DOWN: (6,14)=2,(6,15)=2,(6,16)=1 stop → lands (6,15)
    do_act(env, game, "RIGHT", "should reach 6,15")

    # Step 11: walk LEFT (5,15)=2. (5,16)=1 stop → lands (5,15)
    do_act(env, game, "LEFT", "should reach 5,15")

    # Step 12: walk LEFT (4,15)=2. (4,16)=1 stop → lands (4,15)
    do_act(env, game, "LEFT", "should reach 4,15")

    # Step 13: flip UP via click g. Player (4,15). (4,14)=1 solid. qssroarxob=True + yuuqpmlxorv → camera only. stays.
    do_click(env, game, (0,17), "flip DOWN->UP @ (4,15)")

    # Step 14: walk LEFT (3,15)`.`. (3,14)# stop.
    do_act(env, game, "LEFT", "should reach 3,15")

    # Step 15: walk LEFT (2,15)`.` → falls up col 2 to (2,8).
    do_act(env, game, "LEFT", "should reach 2,8 via col2 fall up")

    # Step 16: walk RIGHT (3,8)=2. (3,7)# stop.
    do_act(env, game, "RIGHT", "should reach 3,8")

    # Step 17: flip DOWN — player (3,8). (3,9)=2→fall grav DOWN: (3,10)=2,(3,11)=2,(3,12)#→lands (3,11).
    do_click(env, game, (0,16), "flip UP->DOWN @ (3,8)")

    # Step 18: RIGHT to (4,11).
    do_act(env, game, "RIGHT", "should reach 4,11")

    # Step 19: flip UP @ (4,11). (4,10)`.`. fsvnqdbzrp grav UP: (4,9)=1 stop. Lands (4,10).
    do_click(env, game, (0,15), "flip DOWN->UP @ (4,11)")

    # Step 20: convert (5,10)→`2` to allow walk/fall through.
    do_click(env, game, (5,10), "5,10 1->2")

    # Step 21: RIGHT (5,10)=2. fsvnqdbzrp grav UP: (5,9),(5,8),(5,7)# → lands (5,8).
    do_act(env, game, "RIGHT", "should reach 5,8")

    # Step 22: stuck (6,8)v, (4,8)v bounces. Flip DOWN (but need (5,10)=1 or dies on (5,11)u)
    # Actually we want to reach col 8 or 9. Let's try clicking (6,9)→`2` (breaks stop) — risky.
    # Or better: reconvert (5,10)→`1` (click (5,10)=2→`1`), then flip DOWN to lands (5,9).
    do_click(env, game, (5,10), "5,10 2->1 again")
    do_click(env, game, (0,14), "flip UP->DOWN @ (5,8) — should land (5,9)")

    # Step 23: walk LEFT from (5,9). (4,9)=1 bounces (we set earlier).
    #   Or RIGHT (6,9)=1 bounces.
    # Convert (4,9)`1`→`2` again. (Break stop for later.)
    do_click(env, game, (4,9), "4,9 1->2")

    do_act(env, game, "LEFT", "walk LEFT from 5,9")

    return 0

if __name__ == "__main__":
    sys.exit(main())

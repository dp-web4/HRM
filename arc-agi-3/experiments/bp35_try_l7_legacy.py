#!/usr/bin/env python3
"""Try the legacy L7 trace after real L6 win. Skip OOB clicks."""
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
L7_TRACE = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/bp35/run_20260413_111605/run_fixed.json")
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

    # L1-L4
    for s in json.load(open(TRACE_IN))["steps"]:
        if s["step"] > 143: break
        a = s["action"]
        if a == "CLICK": env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif a != "CLICK_OOB_SKIPPED": env.step(ACT_MAP[a])
    env.step(ACT_MAP["LEFT"])  # L5
    # L6 solve
    for _ in range(5): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    for tgt in [(6,22), (4,31), (8,1)]:
        vx, vy = vp_for(scene, tgt)
        env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    for _ in range(6): env.step(ACT_MAP["LEFT"])

    scene = game.oztjzzyqoek
    assert scene.qswcochjodb == 7, f"Expected L7, got {scene.qswcochjodb}"
    print(f"On L7 at {scene.twdpowducb.qumspquyus}, grav_up={scene.vivnprldht}")

    # Replay legacy L7 steps — clamp OOB clicks
    l7_steps = [s for s in json.load(open(L7_TRACE))["steps"] if s.get("level") == 7]
    for i, s in enumerate(l7_steps):
        act = s["action"]
        lvl_before = scene.qswcochjodb
        player_before = scene.twdpowducb.qumspquyus
        grav_before = scene.vivnprldht
        if act == "CLICK":
            x, y = s["x"], s["y"]
            if not (0 <= x <= 63 and 0 <= y <= 63):
                print(f"  {i}: OOB click ({x},{y}) — skipping")
                continue
            env.step(ACT_MAP["CLICK"], data={"x": x, "y": y})
        else:
            env.step(ACT_MAP[act])
        scene = game.oztjzzyqoek
        lvl = scene.qswcochjodb
        try:
            player = scene.twdpowducb.qumspquyus
            grav = scene.vivnprldht
        except Exception:
            player = grav = "?"
        extra = f" ({s.get('x')},{s.get('y')})" if act=='CLICK' else ''
        print(f"  L7-step{i:3}: {act}{extra}  before L{lvl_before} P={player_before} grav_up={grav_before} → L{lvl} P={player} grav_up={grav}")
        if lvl > 7:
            print(f"*** L7 WON — advanced to L{lvl} ***")
            return 0
        if getattr(scene, "jrhqdvdwpsb", False):
            print(f"*** DIED at L7 step {i} ***")
            return 1
    print("L7 trace exhausted without win/die")
    return 1

if __name__ == "__main__":
    sys.exit(main())

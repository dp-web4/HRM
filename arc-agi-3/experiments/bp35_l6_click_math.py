#!/usr/bin/env python3
"""Figure out exact click coord math by calling the engine's hyntnfvpgl."""
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
    env.step(ACT_MAP["LEFT"])  # L5 → L6
    # Walk right 5x
    for _ in range(5):
        env.step(ACT_MAP["RIGHT"])

    scene = game.oztjzzyqoek
    px, py = scene.twdpowducb.qumspquyus
    print(f"P=({px},{py}) gravUp={scene.vivnprldht}")
    cam = scene.camera
    print(f"camera rczgvgfsfb = {cam.rczgvgfsfb}")
    print(f"tile_w={scene.hdnrlfmyrj.unxmkbpkzwj}, tile_h={scene.hdnrlfmyrj.ltlyhlyvapv}")
    print(f"grid origin (knpqzpefyn) = {scene.hdnrlfmyrj.knpqzpefyn()}")

    # Test what click viewport coords correspond to grid (6, 22)
    # The bp35 code: kojxiszwpx = self.hdnrlfmyrj.hyntnfvpgl(x, y + self.camera.rczgvgfsfb[1])
    # where x,y are viewport coords.
    target = (6, 22)
    cam_y_off = cam.rczgvgfsfb[1]
    ox, oy = scene.hdnrlfmyrj.knpqzpefyn()
    tw = scene.hdnrlfmyrj.unxmkbpkzwj
    th = scene.hdnrlfmyrj.ltlyhlyvapv

    # Want: hyntnfvpgl(vp_x, vp_y + cam_y) = (target_gx, target_gy)
    # => (vp_x - ox) // tw = target_gx  → vp_x = ox + target_gx * tw
    # => (vp_y + cam_y - oy) // th = target_gy → vp_y = oy + target_gy * th - cam_y
    vp_x = ox + target[0] * tw
    vp_y = oy + target[1] * th - cam_y_off
    print(f"correct click for grid {target}: vp=({vp_x},{vp_y})")

    # Verify by calling hyntnfvpgl
    result = scene.hdnrlfmyrj.hyntnfvpgl(vp_x, vp_y + cam_y_off)
    print(f"hyntnfvpgl({vp_x}, {vp_y + cam_y_off}) = {result}")

if __name__ == "__main__":
    main()

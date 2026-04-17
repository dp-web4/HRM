#!/usr/bin/env python3
"""Find the WIN gem (fjlzdjxhant) on L6 live grid."""
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
    scene = game.oztjzzyqoek
    px, py = scene.twdpowducb.qumspquyus
    print(f"L{scene.qswcochjodb} P=({px},{py}) grav_up={scene.vivnprldht}\n")

    WIN_GEM = "fjlzdjxhant"
    BOUNDARY_GEM = "lrpkmzabbfa"
    SPIKE_A = "ubhhgljbnpu"
    SPIKE_B = "hzusueifitk"
    GROUND = "qclfkhjnaac"  # destructible ground "x"
    BG = "xcjjwqfzjfe"  # background wall (indestructible "o")

    win_gems = []
    boundary_gems = []
    grounds = []
    spikes_a = []
    spikes_b = []
    for y in range(45):
        for x in range(12):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            for e in ents:
                if e.name == WIN_GEM: win_gems.append((x,y))
                elif e.name == BOUNDARY_GEM: boundary_gems.append((x,y))
                elif e.name == GROUND: grounds.append((x,y))
                elif e.name == SPIKE_A: spikes_a.append((x,y))
                elif e.name == SPIKE_B: spikes_b.append((x,y))
    print(f"WIN gems (+/fjlzdjxhant): {win_gems}")
    print(f"Boundary gems (g/lrpkmzabbfa): {boundary_gems}")
    print(f"Destructible ground (x/qclfkhjnaac): {grounds}")
    print(f"Spikes A (v/ubhhgljbnpu): {spikes_a}")
    print(f"Spikes B (u/hzusueifitk): {spikes_b}")

if __name__ == "__main__":
    main()

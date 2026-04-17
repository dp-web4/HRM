#!/usr/bin/env python3
"""BP35 L6 live-grid probe.

Replay L1-L5 (via the L5-winning chain: 143 steps for L1-L4 + 1 LEFT for L5)
then walk the live L6 grid and dump ASCII + current player / gravity state.
This is pure exploration — no moves on L6 yet. The goal: figure out what L6
actually looks like in the engine, not what anyone reconstructed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.setrecursionlimit(50000)

ARCSAGE_EXPERIMENTS = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/arc-agi-3/experiments")
if (ARCSAGE_EXPERIMENTS / "environment_files" / "bp35").exists():
    os.chdir(ARCSAGE_EXPERIMENTS)

from arc_agi import Arcade
from arcengine import GameAction

TRACE_IN = Path(
    "/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/bp35/"
    "run_legitimate_chain/run.json"
)

ACT_MAP = {
    "UP": GameAction.ACTION1,
    "DOWN": GameAction.ACTION2,
    "LEFT": GameAction.ACTION3,
    "RIGHT": GameAction.ACTION4,
    "CLICK": GameAction.ACTION6,
    "UNDO": GameAction.ACTION7,
}


def dump_full_map(scene, width: int = 12, height: int = 45) -> str:
    lines = []
    header = "    " + "".join(f"{x % 10}" for x in range(width))
    lines.append(header)
    try:
        px, py = scene.twdpowducb.qumspquyus
    except Exception:
        px, py = (-1, -1)
    for y in range(height):
        row_chars = []
        for x in range(width):
            try:
                ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            except Exception:
                row_chars.append("?")
                continue
            names = [e.name for e in ents]
            if not names:
                ch = " "
            elif (x, y) == (px, py):
                ch = "P"
            elif "fjlzdjxhant" in names:
                ch = "+"
            elif "lrpkmzabbfa" in names:
                ch = "g"
            elif "qclfkhjnaac" in names:
                ch = "x"
            elif "yuuqpmlxorv" in names:
                ch = "1"
            elif "oonshderxef" in names:
                ch = "2"
            elif "xcjjwqfzjfe" in names:
                ch = "."
            elif "ubhhgljbnpu" in names or "hzusueifitk" in names:
                ch = "v"
            elif "etlsaqqtjvn" in names:
                ch = "y"
            else:
                ch = "?"
            row_chars.append(ch)
        lines.append(f"{y:2}  {''.join(row_chars)}")
    return "\n".join(lines)


def main() -> int:
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game

    trace_in = json.load(open(TRACE_IN))
    src_steps = trace_in["steps"]
    l1_l4_steps = [s for s in src_steps if s["step"] <= 143]

    for s in l1_l4_steps:
        act = s["action"]
        if act == "CLICK":
            env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif act == "CLICK_OOB_SKIPPED":
            continue
        else:
            env.step(ACT_MAP[act])

    scene = game.oztjzzyqoek
    print(f"After L1-L4 replay: level={scene.qswcochjodb}, "
          f"player={scene.twdpowducb.qumspquyus}, grav_up={scene.vivnprldht}")

    # Win L5
    env.step(ACT_MAP["LEFT"])
    scene = game.oztjzzyqoek  # may rebind after level transition

    print("\n=== After L5 WIN (should be on L6 spawn) ===")
    print(f"level={scene.qswcochjodb}")
    print(f"player={scene.twdpowducb.qumspquyus}")
    print(f"grav_up={scene.vivnprldht}")
    print(f"camera={scene.kyhwokmebx.rczgvgfsfb if hasattr(scene, 'kyhwokmebx') else 'n/a'}")

    print("\n=== L6 FULL GRID ===")
    print(dump_full_map(scene, width=12, height=45))

    return 0


if __name__ == "__main__":
    sys.exit(main())

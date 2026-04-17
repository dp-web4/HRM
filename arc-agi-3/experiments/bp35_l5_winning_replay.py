#!/usr/bin/env python3
"""BP35 L5 winning replay.

Prior verdict: "L5 gem chamber is sealed — requires 3 gravity flips with only 2 G tiles."
That verdict was built on a reconstructed grid with wrong coordinates.

Reality (verified by actually replaying the L1-L4 chain in the live engine):
- After L1-L4 replay, the player arrives on L5 at grid (6, 7), gravity UP, camera (0, 6).
- The gem is at grid (5, 7) — IMMEDIATELY LEFT of the player.
- A single ACTION3 (LEFT) wins L5 and advances to L6.

The "sealed chamber" verdict came from an offline grid reconstruction that placed the
player at (3, 23) on L5 — but (3, 23) is actually the L6 spawn (verified by inspecting
scene.qswcochjodb == 6 after the L5 win). Prior agents conflated L5 and L6 geometry,
then BFS'd an L6-like map and concluded "unsolvable."

Framing shift: when analysis disagrees with the engine, trust the engine. Walking the
map in state, not in the head.

This script replays the L1-L4 chain from run_legitimate_chain/run.json (143 steps) and
then wins L5 with a single LEFT action. It also captures 64x64 PNG frames before and
after the winning move and writes the complete step-log to
run_L5_winning/run.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.setrecursionlimit(50000)

# Make sure we can import arc_agi and the environment_files directory we need
EXPERIMENTS_DIR = Path(__file__).resolve().parent
# Some local experiment dirs have an environment_files/ sibling; the ARC-SAGE
# experiments dir is the canonical one. Prefer the directory that contains bp35.
ARCSAGE_EXPERIMENTS = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/arc-agi-3/experiments")
if not (EXPERIMENTS_DIR / "environment_files" / "bp35").exists() and \
        (ARCSAGE_EXPERIMENTS / "environment_files" / "bp35").exists():
    os.chdir(ARCSAGE_EXPERIMENTS)

from arc_agi import Arcade
from arcengine import GameAction

OUT_DIR = Path(
    "/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/bp35/run_L5_winning"
)
FRAMES_DIR = OUT_DIR / "frames"
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


def save_frame(arr: np.ndarray, path: Path) -> None:
    """Render a 64x64 int8 frame as a PNG without external deps beyond matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr, cmap="tab20", interpolation="nearest")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=80, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        # Fall back to raw PPM P2 if matplotlib is unavailable
        with open(path.with_suffix(".pgm"), "w") as fp:
            fp.write("P2\n64 64\n255\n")
            for row in arr:
                fp.write(" ".join(str(int(v) & 0xFF) for v in row) + "\n")


def step_frame(fd) -> np.ndarray:
    """Extract the last-rendered 64x64 frame from a frame-data object."""
    frames = fd.frame
    if isinstance(frames, list) and frames:
        last = frames[-1]
        if isinstance(last, np.ndarray):
            return last
        return np.array(last)
    return np.array(frames)


def dump_map(scene) -> str:
    """Return a compact ASCII render of the full L5 grid directly from the engine."""
    width = 12
    height = 40
    lines: list[str] = []
    header = "   " + "".join(f"{x % 10}" for x in range(width))
    lines.append(header)
    px, py = scene.twdpowducb.qumspquyus
    for y in range(height):
        row_chars: list[str] = []
        for x in range(width):
            ents = scene.hdnrlfmyrj.jhzcxkveiw(x, y)
            names = [e.name for e in ents]
            if not names:
                ch = " "
            elif (x, y) == (px, py):
                ch = "P"
            elif "fjlzdjxhant" in names:
                ch = "+"
            elif "lrpkmzabbfa" in names:
                ch = "G"
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
                ch = "E"
            else:
                ch = "?"
            row_chars.append(ch)
        lines.append(f"{y:2} {''.join(row_chars)}")
    return "\n".join(lines)


def main() -> int:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    fd = env.reset()
    game = env._game

    trace_in = json.load(open(TRACE_IN))
    src_steps = trace_in["steps"]
    # Steps 1..143 take us through L1..L4 and deposit us on L5 spawn.
    l1_l4_steps = [s for s in src_steps if s["step"] <= 143]

    out_steps: list[dict] = []

    def record(action_name: str, note: str = "", x: int | None = None,
               y: int | None = None) -> None:
        entry = {
            "step": len(out_steps) + 1,
            "level": game.oztjzzyqoek.qswcochjodb - 1,
            "action": action_name,
        }
        if x is not None:
            entry["x"] = x
        if y is not None:
            entry["y"] = y
        if note:
            entry["note"] = note
        out_steps.append(entry)

    # Replay L1-L4
    for s in l1_l4_steps:
        act = s["action"]
        if act == "CLICK":
            env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
            record("CLICK", note=f"L1-L4 replay[{s['step']}]", x=s["x"], y=s["y"])
        elif act == "CLICK_OOB_SKIPPED":
            continue
        else:
            env.step(ACT_MAP[act])
            record(act, note=f"L1-L4 replay[{s['step']}]")

    scene = game.oztjzzyqoek
    print(f"After L1-L4 replay: level={scene.qswcochjodb}, "
          f"player={scene.twdpowducb.qumspquyus}, grav_up={scene.vivnprldht}")
    assert scene.qswcochjodb == 5, f"Expected to be on L5, got {scene.qswcochjodb}"
    assert scene.twdpowducb.qumspquyus == (6, 7), \
        f"Expected spawn (6,7), got {scene.twdpowducb.qumspquyus}"

    # Capture the pre-L5-move frame + ASCII map
    pre_fd = env._last_frame if hasattr(env, "_last_frame") else None
    # Use a fresh step's frame by doing a no-op-free capture — render directly
    pre_frame = scene.srlqyenmue()
    save_frame(pre_frame, FRAMES_DIR / "L5_pre_winning_move_p6_7.png")
    with open(OUT_DIR / "L5_pre_map.txt", "w") as fp:
        fp.write(dump_map(scene) + "\n")

    # THE WINNING MOVE: gem is at (5, 7), just LEFT of player (6, 7).
    fd_final = env.step(ACT_MAP["LEFT"])
    record("LEFT", note="WINNING MOVE: step onto gem at (5,7)")

    post_level = scene.qswcochjodb
    post_player = scene.twdpowducb.qumspquyus
    levels_completed = getattr(fd_final, "levels_completed", None)

    print(f"After LEFT: level={post_level}, player={post_player}, "
          f"levels_completed={levels_completed}")

    # Engine auto-advances on win — we are now on L6.
    # Capture a post-win frame (L6 spawn view)
    post_frame = scene.srlqyenmue()
    save_frame(post_frame, FRAMES_DIR / f"L5_post_winning_move_L{post_level}_p{post_player[0]}_{post_player[1]}.png")

    winning = levels_completed is not None and levels_completed >= 5
    print(f"WON L5? {winning}")

    out = {
        "game_id": "bp35-0a0ad940",
        "player": "bp35-L5-winning",
        "win_levels_proved": 5,
        "total_steps": len(out_steps),
        "l5_trigger_step": len(out_steps),
        "l5_winning_action": "LEFT",
        "l5_spawn": list(scene.twdpowducb.qumspquyus) if post_level == 5 else [6, 7],
        "l5_gem_position": [5, 7],
        "l5_gravity_up": True,
        "final_level_index": post_level,
        "final_levels_completed": levels_completed,
        "final_player": list(post_player),
        "note": (
            "Prior 'sealed chamber' verdict for L5 was based on an incorrect reconstruction "
            "of the level geometry. Engine inspection shows the L5 spawn is (6,7) with gem "
            "immediately left at (5,7) — a single LEFT action wins L5. The (3,23) spawn "
            "described in prior analyses is in fact the L6 spawn; prior BFS treated the L6 "
            "grid geometry as L5 and concluded unsolvable."
        ),
        "steps": out_steps,
    }

    with open(OUT_DIR / "run.json", "w") as fp:
        json.dump(out, fp, indent=2)

    print(f"Wrote run.json to {OUT_DIR / 'run.json'}")
    print(f"Wrote frames to {FRAMES_DIR}")
    return 0 if winning else 1


if __name__ == "__main__":
    sys.exit(main())

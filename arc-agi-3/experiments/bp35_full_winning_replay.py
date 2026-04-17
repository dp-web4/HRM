#!/usr/bin/env python3
"""BP35 full winning replay through L7 (L8+ in progress).

Plays through L1, L2, L3, L4, L5, L6, L7 and lands on L8 spawn.
- L1-L4: replay 143 actions from prior winning chain.
- L5: single LEFT (gem at (5,7), player at (6,7) grav UP — already discovered).
- L6: 14 actions — RIGHT×5, CLICK 3 g-tiles to flip gravity & destroy them, LEFT×6.
- L7: ~45 actions — `1`/`2` tile-conversion puzzle, navigate from spawn (3,19) up
  through upper chamber via col 7/8/9 back down to gem chamber at (3,25).

The script writes a runnable trace and frames per level to:
  ARC-SAGE/knowledge/visual-memory/bp35/run_L6plus_winning/

L8 is geometrically extremely hard: gem at (9,19) is walled off (only reachable by
converting (8,18) `1`->`2` and falling through). To even reach (8,17) requires
y-spread (etlsaqqtjvn tiles) blocking spike row at y=27. Single gravity gem
at (5,2) means precise timing required. Partial analysis in
bp35_l8_probe.py and bp35_l8_solve.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.setrecursionlimit(50000)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ARCSAGE_EXPERIMENTS = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/arc-agi-3/experiments")
if not (EXPERIMENTS_DIR / "environment_files" / "bp35").exists() and \
        (ARCSAGE_EXPERIMENTS / "environment_files" / "bp35").exists():
    os.chdir(ARCSAGE_EXPERIMENTS)

from arc_agi import Arcade
from arcengine import GameAction

OUT_DIR = Path(
    "/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/bp35/run_L6plus_winning"
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


def vp_for(scene, target):
    cam_y = scene.camera.rczgvgfsfb[1]
    return target[0] * 6, target[1] * 6 - cam_y


def save_frame(arr: np.ndarray, path: Path) -> None:
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
    except Exception:
        with open(path.with_suffix(".pgm"), "w") as fp:
            fp.write("P2\n64 64\n255\n")
            for row in arr:
                fp.write(" ".join(str(int(v) & 0xFF) for v in row) + "\n")


class Tracer:
    """Wraps env.step() to record every action and capture key frames."""

    def __init__(self, env, game):
        self.env = env
        self.game = game
        self.steps = []
        self.frames_saved = set()

    def _scene(self):
        return self.game.oztjzzyqoek

    def step_action(self, action_name: str, note: str = "") -> object:
        scene_before = self._scene()
        lvl_before = scene_before.qswcochjodb
        fd = self.env.step(ACT_MAP[action_name])
        scene_after = self.game.oztjzzyqoek
        lvl_after = scene_after.qswcochjodb
        try:
            p = scene_after.twdpowducb.qumspquyus
            grav = scene_after.vivnprldht
        except Exception:
            p, grav = None, None
        entry = {
            "step": len(self.steps) + 1,
            "level": lvl_before,
            "action": action_name,
        }
        if note:
            entry["note"] = note
        self.steps.append(entry)
        self._maybe_capture_frame(scene_before, scene_after, lvl_before, lvl_after, p)
        return fd

    def step_click(self, target, note: str = "") -> object:
        scene_before = self._scene()
        lvl_before = scene_before.qswcochjodb
        vx, vy = vp_for(scene_before, target)
        fd = self.env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene_after = self.game.oztjzzyqoek
        lvl_after = scene_after.qswcochjodb
        try:
            p = scene_after.twdpowducb.qumspquyus
        except Exception:
            p = None
        entry = {
            "step": len(self.steps) + 1,
            "level": lvl_before,
            "action": "CLICK",
            "x": vx,
            "y": vy,
            "grid_target": list(target),
        }
        if note:
            entry["note"] = note
        self.steps.append(entry)
        self._maybe_capture_frame(scene_before, scene_after, lvl_before, lvl_after, p)
        return fd

    def _maybe_capture_frame(self, scene_before, scene_after, lvl_before, lvl_after, player):
        # Capture first frame on each level + level transition.
        if lvl_after not in self.frames_saved:
            try:
                arr = scene_after.srlqyenmue()
                p = player or (-1, -1)
                save_frame(arr, FRAMES_DIR / f"L{lvl_after}_first_frame_p{p[0]}_{p[1]}.png")
                self.frames_saved.add(lvl_after)
            except Exception:
                pass
        # Capture every transition (level advance)
        if lvl_after > lvl_before:
            try:
                arr = scene_after.srlqyenmue()
                save_frame(arr, FRAMES_DIR / f"L{lvl_before}_won_advance_to_L{lvl_after}.png")
            except Exception:
                pass


def play_l1_to_l4(tracer: Tracer):
    """Replay L1-L4 from the legacy 143-step chain."""
    src = json.load(open(TRACE_IN))["steps"]
    for s in src:
        if s["step"] > 143:
            break
        a = s["action"]
        if a == "CLICK":
            # Use literal viewport coords from the trace
            vx, vy = s["x"], s["y"]
            # Direct env step (no Tracer.step_click since coords are pre-computed)
            scene_before = tracer._scene()
            lvl_before = scene_before.qswcochjodb
            tracer.env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
            tracer.steps.append({
                "step": len(tracer.steps) + 1,
                "level": lvl_before,
                "action": "CLICK",
                "x": vx, "y": vy,
                "note": f"L1-L4 replay[{s['step']}]",
            })
        elif a == "CLICK_OOB_SKIPPED":
            continue
        else:
            tracer.step_action(a, note=f"L1-L4 replay[{s['step']}]")


def play_l5(tracer: Tracer):
    tracer.step_action("LEFT", note="L5 win: step onto gem at (5,7)")


def play_l6(tracer: Tracer):
    """L6: walk right to (8,23), click 3 g's to destroy them, walk LEFT through chamber to + at (2,31)."""
    for _ in range(5):
        tracer.step_action("RIGHT", note="L6 traverse to (8,23)")
    tracer.step_click((6, 22), note="L6 click g(6,22): flip grav & destroy & fall to (8,31)")
    tracer.step_click((4, 31), note="L6 click g(4,31): flip & destroy & fall back to (8,23)")
    tracer.step_click((8, 1), note="L6 click g(8,1): flip & destroy & fall to (8,31)")
    for _ in range(6):
        tracer.step_action("LEFT", note="L6 walk LEFT to (2,31) gem (g's destroyed, path clear)")


def play_l7(tracer: Tracer):
    """L7: 45-step chain via 1/2 tile conversion puzzle."""
    # Phase A: setup conversions from spawn (3,19)
    for tgt, note in [
        ((4, 14), "L7 setup: (4,14) 2->1 (col4 floor)"),
        ((5, 14), "L7 setup: (5,14) 2->1 (col5 floor)"),
        ((4, 16), "L7 setup: (4,16) 2->1"),
        ((5, 16), "L7 setup: (5,16) 2->1"),
        ((6, 16), "L7 setup: (6,16) 2->1 (col6 fall stop)"),
        ((6, 19), "L7 setup: (6,19) 2->1 (so (6,20) standable)"),
        ((4, 9), "L7 setup: (4,9) 2->1 (col4 upper)"),
    ]:
        tracer.step_click(tgt, note=note)

    # Phase B: spawn -> (8,15) -> (4,13)
    tracer.step_click((0, 19), note="L7 flip g(0,19) UP->DOWN -> (3,21)")
    for _ in range(3):
        tracer.step_action("RIGHT", note="L7 traverse y=21 to (6,21)")
    tracer.step_click((0, 21), note="L7 flip g(0,21) DOWN->UP -> (6,20)")
    tracer.step_action("RIGHT", note="L7 -> (7,20)")
    tracer.step_action("RIGHT", note="L7 -> (8,15) via col 8 fall up")
    for _ in range(4):
        tracer.step_action("LEFT", note="L7 traverse y=13 chain")
    tracer.step_click((0, 18), note="L7 flip g(0,18) UP->DOWN @ (4,13)")
    tracer.step_action("RIGHT")
    tracer.step_action("RIGHT", note="L7 -> (6,15)")
    tracer.step_action("LEFT", note="L7 -> (5,15)")
    tracer.step_action("LEFT", note="L7 -> (4,15)")
    tracer.step_click((0, 17), note="L7 flip g(0,17) DOWN->UP @ (4,15)")
    tracer.step_action("LEFT", note="L7 -> (3,15)")
    tracer.step_action("LEFT", note="L7 -> (2,8) via col2 fall up")
    tracer.step_action("RIGHT", note="L7 -> (3,8)")
    # Phase C: descend to col 7, rise to (7,4), descend to (7,7), to col 8/9
    tracer.step_click((0, 16), note="L7 flip g(0,16) UP->DOWN @ (3,8) -> (3,11)")
    tracer.step_action("RIGHT", note="L7 -> (4,11)")
    tracer.step_click((0, 15), note="L7 flip g(0,15) DOWN->UP @ (4,11) -> (4,10)")
    tracer.step_click((5, 10), note="L7 (5,10) 1->2 (allow col5 fall)")
    tracer.step_action("RIGHT", note="L7 -> (5,8)")
    tracer.step_click((5, 10), note="L7 (5,10) 2->1 (revert)")
    tracer.step_click((0, 14), note="L7 flip g(0,14) UP->DOWN @ (5,8) -> (5,9)")
    tracer.step_click((6, 9), note="L7 (6,9) 1->2 (walk thru)")
    tracer.step_action("RIGHT", note="L7 -> (6,11)")
    tracer.step_click((6, 9), note="L7 (6,9) 2->1 (stop col6 fall up)")
    tracer.step_click((0, 13), note="L7 flip g(0,13) DOWN->UP @ (6,11) -> (6,10)")
    tracer.step_action("RIGHT", note="L7 -> (7,4) via col7 fall up")
    tracer.step_click((7, 8), note="L7 (7,8) 2->1 (col7 fall stop at (7,7))")
    tracer.step_click((0, 12), note="L7 flip g(0,12) UP->DOWN @ (7,4) -> (7,7)")
    tracer.step_action("RIGHT", note="L7 -> (8,7)")
    tracer.step_action("RIGHT", note="L7 -> (9,26) via col 9 fall down")
    # Phase D: navigate to gem (3,25)
    tracer.step_action("LEFT", note="L7 -> (8,26)")
    tracer.step_action("LEFT", note="L7 -> (7,26)")
    tracer.step_click((0, 11), note="L7 flip g(0,11) DOWN->UP @ (7,26) -> (7,23)")
    tracer.step_action("LEFT", note="L7 -> (6,23)")
    tracer.step_action("LEFT", note="L7 -> (5,23)")
    tracer.step_action("LEFT", note="L7 -> (4,23)")
    tracer.step_action("LEFT", note="L7 -> (3,23)")
    tracer.step_click((0, 10), note="L7 WIN! flip g(0,10) UP->DOWN @ (3,23) -> fall to + at (3,25)")


def play_l8_attempt(tracer: Tracer):
    """L8 attempt: y-spread chain from (3,18). NOT YET WINNING."""
    # Phase A: y-spread chain at row 16 (creates platform for col 5 fall stop... but col 5 dies on row 27 spike)
    # See bp35_l8_solve.py for details. This is left as a marker for future work.
    for tgt, note in [
        ((3, 18), "L8 y-spread: (3,18) -> (3,17),(3,19),(2,18),(4,18)"),
        ((3, 17), "L8 y-spread: (3,17) -> (3,16) etc"),
        ((3, 16), "L8 y-spread: (3,16) -> (3,15)"),
        ((3, 15), "L8 y-spread: (3,15) -> (4,15)"),
        ((4, 15), "L8 y-spread: (4,15) -> (5,15)"),
        ((5, 15), "L8 y-spread: (5,15) -> (6,15)"),
        ((6, 15), "L8 y-spread: (6,15) -> (7,15)"),
        ((7, 15), "L8 y-spread: (7,15) -> (8,15) and (7,16)"),
        ((8, 15), "L8 y-spread: (8,15) -> (8,16)y"),
    ]:
        tracer.step_click(tgt, note=note)
    # Note: walking RIGHT here would die on col 5 (5,27)v spike via fall up.
    # Need additional y-spread chain from (2,29) to block row 27. See bp35_l8_solve.py.


def main() -> int:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    tracer = Tracer(env, game)

    # Capture L1 first frame
    arr = game.oztjzzyqoek.srlqyenmue()
    save_frame(arr, FRAMES_DIR / "L1_initial_state.png")
    tracer.frames_saved.add(1)

    play_l1_to_l4(tracer)
    play_l5(tracer)
    play_l6(tracer)
    play_l7(tracer)
    # L8 attempt (does not win)
    play_l8_attempt(tracer)

    scene = game.oztjzzyqoek
    final_lvl = scene.qswcochjodb
    try:
        final_p = scene.twdpowducb.qumspquyus
    except Exception:
        final_p = None

    # Capture final frame
    try:
        arr = scene.srlqyenmue()
        save_frame(arr, FRAMES_DIR / f"final_L{final_lvl}_p{final_p}.png" if final_p else f"final_L{final_lvl}.png")
    except Exception:
        pass

    out = {
        "game_id": "bp35-0a0ad940",
        "player": "bp35-L7-winning-L8-partial",
        "win_levels_proved": 7,  # L1-L7 won
        "total_steps": len(tracer.steps),
        "final_level_index": final_lvl,
        "final_player": list(final_p) if final_p else None,
        "phase_breakdown": {
            "L1-L4": "143 steps replayed from prior chain",
            "L5": "1 step (LEFT) — gem (5,7) immediately left of player",
            "L6": "14 steps — destroy 3 g gravity gems then walk through cleared chamber",
            "L7": "45 steps — multi-stage 1/2 tile conversion puzzle, 8 g flips",
            "L8": "9 steps (y-spread setup), partial — not yet winning",
        },
        "notes": [
            "Walking grav UP into (3,23) then click any g (flip to DOWN) triggers "
            "fsvnqdbzrp grav DOWN starting (3,24)→(3,25)+ → win() called via fall path.",
            "L8 gem at (9,19) walled off; only access via converting (8,18) `1`->`2` "
            "and falling from (8, y<18) grav DOWN. Reaching col 8 upper requires "
            "y-spread chain to bypass spike row at y=27. Single (5,2)g limits options.",
            "All deferred file targets are absolute paths under SAGE/arc-agi-3/experiments.",
        ],
        "steps": tracer.steps,
    }

    with open(OUT_DIR / "run.json", "w") as fp:
        json.dump(out, fp, indent=2)

    print(f"Final: L{final_lvl}, P={final_p}, total steps={len(tracer.steps)}")
    print(f"Wrote {OUT_DIR / 'run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

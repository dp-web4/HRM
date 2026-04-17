#!/usr/bin/env python3
"""
L7 winning replay — in progress.

Discord reframe: L10 strategy applies. Red piece = stepping stone like blue in L10.
Transport red across the board so one N can jump over red to reduce to 2 pieces.

Verified phases (as of task entry):
  Phase 1: pushes L,L,U,U,R,R,R — creates landing at (6,3)
  Phase 2: red (6,1) -> (6,3) — red jumps onto block at (6,3)
  Phase 3: pushes L,L,L,D,D,L,L,D — transports red-on-block to (1,6)
  Phase 4: red (1,6) -> (1,8) — red jumps down off block onto walkable
  Phase 5: red (1,8) -> (3,8) -> (5,8) — red hops via pegs along bottom

Grid after Phase 5 (red @ (5,8)):
        012345678901234567890123456789
      0                  #####
      1 N     .    ####  ;   #
      2 o     o    #  # #;#####
      3 #######  ####o. ; # # #
      4    #     ;    .   ; # #
      5  ###### ####  .o.o. # .
      6  ;    # #  #  .o.o.   N
      7  o    o ###o
      8  .o.oR....#.

Pieces now: N(0,1), N(22,6), R(5,8)
Blocks: (1,6), (9,4), (16,3), (17,1), (17,2), (18,4)
Pegs: (0,2), (1,7), (2,8), (4,8), (6,2), (6,7), (11,7), (13,3), (15,5),
      (15,6), (16,3), (17,1), (17,5), (17,6), (18,4)

Next steps TBD — continue transporting R toward (22,6) area.

Actions:
  ACTION1/2/3/4 = push UP/DOWN/LEFT/RIGHT
  ACTION6 with data={'x','y'} = click
  Coords: vp_x = world_x*6 + offset[0] + 3
          vp_y = world_y*6 + offset[1] + 3
  Arrow click: + (dx_half*12, dy_half*12) from piece center
"""
import os, sys, json, time
from pathlib import Path

os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.setrecursionlimit(50000)
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from PIL import Image

from arc_agi import Arcade
from arcengine import GameAction
from lf52_solve_final import solve_level, PALETTE, extract_state

OUT_DIR = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/lf52/run_L7_winning")
FRAMES_DIR = OUT_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
RUN_JSON = OUT_DIR / "run.json"

DIR_ACTIONS = {'U': GameAction.ACTION1, 'D': GameAction.ACTION2,
               'L': GameAction.ACTION3, 'R': GameAction.ACTION4}
DIR_VEC = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}

trace_steps = []
frame_counter = [0]


def save_frame(env, label):
    """Save frame as PNG."""
    try:
        arr = np.array(env.observation_space.frame[0])
        h, w = arr.shape
        s = 6
        img = Image.new('RGB', (w*s, h*s))
        pix = img.load()
        for y in range(h):
            for x in range(w):
                c = PALETTE.get(int(arr[y, x]), (0, 0, 0))
                for dy in range(s):
                    for dx in range(s):
                        pix[x*s+dx, y*s+dy] = c
        fname = f"{frame_counter[0]:03d}_{label}.png"
        img.save(FRAMES_DIR / fname)
        frame_counter[0] += 1
    except Exception as e:
        print(f"  save_frame error: {e}")


def write_trace():
    """Write current trace incrementally."""
    data = {
        "game_id": "lf52-271a04aa",
        "player": "cbp-l7-reframe-replay",
        "target_level": 7,
        "reframe": "red piece is transport/stepping-stone (like blue in L10)",
        "steps": trace_steps,
        "step_count": len(trace_steps),
    }
    RUN_JSON.write_text(json.dumps(data, indent=2))


def do_push(env, game, dir_char, label=""):
    """Execute a push and record."""
    fd = env.step(DIR_ACTIONS[dir_char])
    trace_steps.append({
        "type": "push", "dir": dir_char,
        "step_count": fd.action_count if hasattr(fd, 'action_count') else None,
        "state": fd.state.name, "levels_completed": fd.levels_completed,
        "label": label,
    })
    write_trace()
    return fd


def do_click(env, vp_x, vp_y, label=""):
    """Execute a raw viewport click."""
    fd = env.step(GameAction.ACTION6, data={'x': vp_x, 'y': vp_y})
    trace_steps.append({
        "type": "click", "vp_x": vp_x, "vp_y": vp_y,
        "state": fd.state.name, "levels_completed": fd.levels_completed,
        "label": label,
    })
    write_trace()
    return fd


def do_jump(env, game, src, dst, label=""):
    """Execute a jump: click src, click arrow."""
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg
    sx, sy = src
    dx_, dy_ = dst

    off = grid.cdpcbbnfdp
    sxp = sx*6 + off[0] + 3
    syp = sy*6 + off[1] + 3
    fd = env.step(GameAction.ACTION6, data={'x': sxp, 'y': syp})
    trace_steps.append({
        "type": "click", "vp_x": sxp, "vp_y": syp,
        "sub": "select", "src": list(src), "dst": list(dst),
        "state": fd.state.name, "levels_completed": fd.levels_completed,
        "label": f"{label}: select {src}",
    })

    off = grid.cdpcbbnfdp
    half_dx = (dx_ - sx) // 2
    half_dy = (dy_ - sy) // 2
    axp = sx*6 + off[0] + half_dx*12 + 3
    ayp = sy*6 + off[1] + half_dy*12 + 3
    fd = env.step(GameAction.ACTION6, data={'x': axp, 'y': ayp})
    trace_steps.append({
        "type": "click", "vp_x": axp, "vp_y": ayp,
        "sub": "arrow", "src": list(src), "dst": list(dst),
        "state": fd.state.name, "levels_completed": fd.levels_completed,
        "label": f"{label}: arrow -> {dst}",
    })
    write_trace()
    return fd


def render_grid(eq, label):
    grid = eq.hncnfaqaddg
    print(f"\n--- {label} | offset={grid.cdpcbbnfdp} steps={eq.asqvqzpfdi} ---")
    H, W = 12, 30
    print("    " + "".join(str(x % 10) for x in range(W)))
    for y in range(H):
        row = f"{y:3d} "
        for x in range(W):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if not names:
                row += " "
            elif any(n == 'fozwvlovdui_red' for n in names):
                row += "R"
            elif any(n == 'fozwvlovdui_blue' for n in names):
                row += "B"
            elif any(n == 'fozwvlovdui' for n in names):
                row += "N"
            elif any(n == 'hupkpseyuim2' for n in names):
                if any('kraubslpehi' in nn for nn in names):
                    row += ";"
                else:
                    row += "b"
            elif any('dgxfozncuiz' in n for n in names):
                row += "o"
            elif any('kraubslpehi' in n for n in names):
                row += "#"
            elif any(n == 'hupkpseyuim' for n in names):
                row += "."
            else:
                row += "?"
        print(row)


def main():
    arc = Arcade(operation_mode='offline')
    env = arc.make('lf52-271a04aa')
    fd = env.reset()
    game = env._game

    print("=== Solving L1-L6 ===")
    for lvl in range(6):
        fd = solve_level(env, game, lvl)
        if fd is None or fd.levels_completed <= lvl:
            print(f"FAIL at L{lvl+1}")
            return 1

    eq = game.ikhhdzfmarl
    save_frame(env, "00_L7_start")
    render_grid(eq, "L7 initial")

    # Record pre-L7 step count to compute actions spent on L7 only
    print(f"\n=== Beginning L7 (steps={eq.asqvqzpfdi}) ===")

    # Phase 1: L,L,U,U,R,R,R
    print("\n--- Phase 1: L,L,U,U,R,R,R ---")
    for c in "LLUURRR":
        do_push(env, game, c, f"P1: push {c}")
    save_frame(env, "01_after_phase1")
    render_grid(eq, "after Phase 1")

    # Phase 2: red (6,1) -> (6,3)
    print("\n--- Phase 2: red (6,1) -> (6,3) ---")
    assert eq.qikmikecdf((6, 1), (0, 1)), "Phase 2 jump invalid"
    do_jump(env, game, (6, 1), (6, 3), "P2: red jump")
    save_frame(env, "02_after_phase2")
    render_grid(eq, "after Phase 2")

    # Phase 3: L,L,L,D,D,L,L,D
    print("\n--- Phase 3: L,L,L,D,D,L,L,D ---")
    for c in "LLLDDLLD":
        do_push(env, game, c, f"P3: push {c}")
    save_frame(env, "03_after_phase3")
    render_grid(eq, "after Phase 3")

    # Phase 4: red (1,6) -> (1,8)
    print("\n--- Phase 4: red (1,6) -> (1,8) ---")
    assert eq.qikmikecdf((1, 6), (0, 1)), "Phase 4 jump invalid"
    do_jump(env, game, (1, 6), (1, 8), "P4: red down")
    save_frame(env, "04_after_phase4")
    render_grid(eq, "after Phase 4")

    # Phase 5: red (1,8) -> (3,8) -> (5,8)
    print("\n--- Phase 5: red (1,8) -> (3,8) -> (5,8) ---")
    assert eq.qikmikecdf((1, 8), (1, 0)), "P5 jump 1 invalid"
    do_jump(env, game, (1, 8), (3, 8), "P5a")
    assert eq.qikmikecdf((3, 8), (1, 0)), "P5 jump 2 invalid"
    do_jump(env, game, (3, 8), (5, 8), "P5b")
    save_frame(env, "05_after_phase5")
    render_grid(eq, "after Phase 5 (red @ (5,8))")

    # === Phase 6+ — this is where we need to find the path ===
    # TODO: from (5,8) continue to (22,6) area. Try stuff.

    # Print summary of state
    state = extract_state(eq)
    print(f"\nFinal pieces: {state['pieces']}")
    print(f"Blocks: {sorted(state['pushable'])}")
    print(f"Step count: {eq.asqvqzpfdi}")
    print(f"Levels completed: {fd.levels_completed}")
    print(f"State: {fd.state.name}")

    write_trace()
    return 0


if __name__ == "__main__":
    sys.exit(main())

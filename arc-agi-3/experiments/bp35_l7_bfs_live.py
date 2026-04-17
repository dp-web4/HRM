#!/usr/bin/env python3
"""BFS L7 using live engine with state snapshot/restore.

We explore over the 4 available actions (L, R, CLICK at each of a few candidate positions, UNDO).
Depth-limit it. Use UNDO to backtrack.

Actually UNDO is limited (one-level). Instead, use vlyikbzinq/svmaaixutx manually.
"""
from __future__ import annotations
import copy
import json, os, sys
from pathlib import Path
from collections import deque
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

def play_through_l7_spawn(env, game):
    for s in json.load(open(TRACE_IN))["steps"]:
        if s["step"] > 143: break
        a = s["action"]
        if a == "CLICK": env.step(ACT_MAP["CLICK"], data={"x": s["x"], "y": s["y"]})
        elif a != "CLICK_OOB_SKIPPED": env.step(ACT_MAP[a])
    env.step(ACT_MAP["LEFT"])
    for _ in range(5): env.step(ACT_MAP["RIGHT"])
    scene = game.oztjzzyqoek
    for tgt in [(6,22), (4,31), (8,1)]:
        vx, vy = vp_for(scene, tgt)
        env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        scene = game.oztjzzyqoek
    for _ in range(6): env.step(ACT_MAP["LEFT"])

def snapshot_grid(scene):
    """Extract grid state as frozenset of (x, y, tag) for L7 (11 wide, 28 tall)."""
    pl = scene.twdpowducb.qumspquyus
    grav = scene.vivnprldht
    tiles = []
    for y in range(28):
        for x in range(11):
            ents = sorted(e.name for e in scene.hdnrlfmyrj.jhzcxkveiw(x, y) if e.name != "player_right")
            if ents:
                tiles.append((x, y, tuple(ents)))
    return (pl, grav, tuple(tiles))

def build_action_list(scene):
    """Generate a list of actions to try at this state: LEFT, RIGHT, and CLICKs on
    non-wall, non-spike tiles."""
    actions = []
    actions.append(("LEFT", None))
    actions.append(("RIGHT", None))
    # CLICK candidates: all 'g' (lrpkmzabbfa), '2' (oonshderxef), '1' (yuuqpmlxorv), 'X' (qclfkhjnaac)
    for y in range(28):
        for x in range(11):
            ents = [e.name for e in scene.hdnrlfmyrj.jhzcxkveiw(x, y)]
            if any(n in ents for n in ("lrpkmzabbfa", "oonshderxef", "yuuqpmlxorv", "qclfkhjnaac", "etlsaqqtjvn")):
                actions.append(("CLICK", (x, y)))
    return actions

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game
    play_through_l7_spawn(env, game)
    scene = game.oztjzzyqoek
    assert scene.qswcochjodb == 7

    # BFS over (player, grav, tile-state). Store path of actions.
    root_state = snapshot_grid(scene)
    # Map state -> (parent_state, action)
    parents = {root_state: (None, None)}
    queue = deque([root_state])
    goal_state = None
    depth_map = {root_state: 0}
    max_depth = 20

    print(f"Root: P={root_state[0]} grav={root_state[1]} tiles={len(root_state[2])}")

    total_explored = 0
    while queue:
        state = queue.popleft()
        depth = depth_map[state]
        if depth >= max_depth:
            continue
        # Restore engine to this state — reset and replay path
        path = []
        s = state
        while parents.get(s, (None, None))[0] is not None:
            parent, action = parents[s]
            path.append(action)
            s = parent
        path.reverse()
        # Reset env
        env.reset()
        play_through_l7_spawn(env, game)
        for action_name, data in path:
            if action_name == "CLICK":
                vx, vy = vp_for(game.oztjzzyqoek, data)
                env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
            else:
                env.step(ACT_MAP[action_name])
        scene = game.oztjzzyqoek
        # Try each candidate action
        actions = build_action_list(scene)
        for action_name, data in actions:
            # Snapshot via vlyikbzinq? Too expensive. Just reset-replay.
            # Actually let's NOT replay here — just run action and observe, then continue
            # BFS works from a fresh replay each time. We'll batch next-state discovery per depth.
            # For now, expand from a single replay and revert via UNDO.
            if action_name == "CLICK":
                vx, vy = vp_for(scene, data)
                fd = env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
            else:
                fd = env.step(ACT_MAP[action_name])
            scene2 = game.oztjzzyqoek
            if scene2.qswcochjodb != 7:
                # Won or lost — check
                if scene2.qswcochjodb > 7:
                    # WIN
                    new_state = (action_name, data)
                    parents[("WON", id(path))] = (state, (action_name, data))
                    print(f"*** WON at depth {depth+1}! path len {len(path)+1} ***")
                    goal_state = (state, (action_name, data))
                    # Print winning path
                    final_path = path + [(action_name, data)]
                    print("Winning path:")
                    for i, a in enumerate(final_path):
                        print(f"  {i}: {a}")
                    return 0
                else:
                    # DIED / reset somehow — rewind via UNDO
                    env.step(ACT_MAP["UNDO"])
            elif scene2.jrhqdvdwpsb:
                env.step(ACT_MAP["UNDO"])
            else:
                new_state = snapshot_grid(scene2)
                if new_state not in parents:
                    parents[new_state] = (state, (action_name, data))
                    depth_map[new_state] = depth + 1
                    queue.append(new_state)
                # Rewind via UNDO
                env.step(ACT_MAP["UNDO"])
            total_explored += 1
            if total_explored % 100 == 0:
                print(f"... explored {total_explored}, queue={len(queue)}, depth={depth}")
        if total_explored > 10000:
            print("Exploration limit hit")
            break
    print(f"BFS finished. total_explored={total_explored}, parents={len(parents)}")
    return 1

if __name__ == "__main__":
    sys.exit(main())

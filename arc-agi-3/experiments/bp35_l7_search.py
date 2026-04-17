#!/usr/bin/env python3
"""Iterative deepening search L7. Uses restart-and-replay for state exploration."""
from __future__ import annotations
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

def replay(env, game, path):
    """Replay L1-L6 then apply path. Returns (scene, won, died)."""
    env.reset()
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
    scene = game.oztjzzyqoek
    if scene.qswcochjodb != 7:
        return None, False, True
    for action_name, data in path:
        if action_name == "CLICK":
            vx, vy = vp_for(scene, data)
            env.step(ACT_MAP["CLICK"], data={"x": vx, "y": vy})
        else:
            env.step(ACT_MAP[action_name])
        scene = game.oztjzzyqoek
        if scene.qswcochjodb > 7:
            return scene, True, False
        if scene.jrhqdvdwpsb:
            return scene, False, True
    return scene, False, False

def state_key(scene):
    try:
        pl = scene.twdpowducb.qumspquyus
        grav = scene.vivnprldht
    except Exception:
        return None
    tiles = []
    for y in range(28):
        for x in range(11):
            ents = tuple(sorted(e.name for e in scene.hdnrlfmyrj.jhzcxkveiw(x, y) if e.name != "player_right"))
            if ents:
                tiles.append((x, y, ents))
    return (pl, grav, tuple(tiles))

def main():
    arc = Arcade(operation_mode="offline", environments_dir="environment_files")
    env = arc.make("bp35-0a0ad940")
    env.reset()
    game = env._game

    # Initial state
    scene, _, _ = replay(env, game, [])
    if scene is None:
        print("Failed to reach L7")
        return 1
    print(f"Start state: P={scene.twdpowducb.qumspquyus} grav={scene.vivnprldht}")

    # Restrict CLICK candidates to relevant tiles (non-wall interactive)
    click_candidates = set()
    for y in range(28):
        for x in range(11):
            ents = [e.name for e in scene.hdnrlfmyrj.jhzcxkveiw(x, y)]
            if any(n in ents for n in ("lrpkmzabbfa", "oonshderxef", "yuuqpmlxorv")):
                click_candidates.add((x, y))
    print(f"{len(click_candidates)} click candidates")

    # BFS with state key deduplication
    root_key = state_key(scene)
    visited = {root_key}
    # Each node: (path, state_key)
    queue = deque([([], root_key)])
    depth = 0
    best_depth = 0
    MAX_NODES = 5000

    nodes_expanded = 0
    while queue and nodes_expanded < MAX_NODES:
        path, key = queue.popleft()
        d = len(path)
        if d > best_depth:
            best_depth = d
            print(f"depth={best_depth}, nodes={nodes_expanded}, queue={len(queue)}")
        if d >= 12:  # depth limit
            continue
        # Generate candidate actions
        # Limit: LEFT, RIGHT, and a subset of CLICK candidates
        current_scene = key[0], key[1]  # player, grav; tiles from key[2]
        actions = [("LEFT", None), ("RIGHT", None)]
        # Click the live candidates from the current state's grid
        # Rebuild scene knowledge
        tiles_at = {}
        for x, y, ents in key[2]:
            tiles_at[(x, y)] = ents
        for (cx, cy) in click_candidates:
            if (cx, cy) in tiles_at:
                ents = tiles_at[(cx, cy)]
                if any(n in ents for n in ("lrpkmzabbfa", "oonshderxef", "yuuqpmlxorv")):
                    actions.append(("CLICK", (cx, cy)))
        nodes_expanded += 1
        for action_name, data in actions:
            new_path = path + [(action_name, data)]
            scene2, won, died = replay(env, game, new_path)
            if won:
                print(f"*** WON at depth {d+1}! ***")
                print("Path:")
                for i, a in enumerate(new_path):
                    print(f"  {i}: {a}")
                return 0
            if died or scene2 is None:
                continue
            new_key = state_key(scene2)
            if new_key and new_key not in visited:
                visited.add(new_key)
                queue.append((new_path, new_key))
    print(f"Exhausted search. nodes={nodes_expanded}, visited={len(visited)}, max_depth={best_depth}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
